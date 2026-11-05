"""
Strategy: Structured rank-reduction via term-merging with linear constraint solving.
Start from schoolbook (rank 90). For each pair of terms, check if they can be merged
into a single term by solving the resulting linear system. Also try a more general
approach: find a rank-r decomposition by treating it as a system of polynomial equations
and using iterative refinement from a near-solution.

Key structural difference: we don't randomly sample; we solve the constraints that
define valid decompositions, using the bilinear structure to reduce the search space.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition using the trace convention W[k*n+i, r]."""
    rank = U.shape[1]
    # Build the tensor check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Build all required products
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                if abs(np.sum(prod) - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_merge_pair(n, m, p, idx_a, idx_b):
    """
    Try to merge schoolbook terms idx_a and idx_b into a single term.
    Term idx_a corresponds to (i_a, j_a, k_a), term idx_b to (i_b, j_b, k_b).
    
    A single term (u, v, w) with entries U[*,r], V[*,r], W[*,r] must satisfy:
    - For (i_a, j_a, k_a): sum over other terms + u[i_a*m+j_a]*v[j_a*p+k_a]*w[k_a*n+i_a] = 1
    - For (i_b, j_b, k_b): sum over other terms + u[i_b*m+j_b]*v[j_b*p+k_b]*w[k_b*n+i_b] = 1
    - For all other (i,j,k): sum over other terms = 1 (unchanged if other terms are schoolbook)
    
    If we keep all other 88 terms as schoolbook, then:
    - (i_a,j_a,k_a) gets 0 from other terms (since that term was removed), so we need:
      u[i_a*m+j_a] * v[j_a*p+k_a] * w[k_a*n+i_a] = 1
    - (i_b,j_b,k_b) gets 0 from other terms, so we need:
      u[i_b*m+j_b] * v[j_b*p+k_b] * w[k_b*n+i_b] = 1
    - All other (i,j,k) get 1 from their own schoolbook term, so the merged term must contribute 0:
      u[i*m+j] * v[j*p+k] * w[k*n+i] = 0 for all (i,j,k) != (i_a,j_a,k_a) and != (i_b,j_b,k_b)
    
    This means the merged term's support must be such that:
    - It has non-zero entries at positions covering (i_a,j_a,k_a) and (i_b,j_b,k_b)
    - It has zero contribution at all other (i,j,k) positions
    
    The simplest case: if (i_a,j_a,k_a) and (i_b,j_b,k_b) share exactly one index,
    we might be able to find a solution. Let's check systematically.
    """
    i_a, j_a, k_a = idx_a // (m * p), (idx_a // p) % m, idx_a % p
    i_b, j_b, k_b = idx_b // (m * p), (idx_b // p) % m, idx_b % p
    
    # For the merged term to work with all other terms being schoolbook:
    # We need a rank-1 tensor (u,v,w) such that:
    #   u[i_a*m+j_a] * v[j_a*p+k_a] * w[k_a*n+i_a] = 1
    #   u[i_b*m+j_b] * v[j_b*p+k_b] * w[k_b*n+i_b] = 1
    #   u[i*m+j] * v[j*p+k] * w[k*n+i] = 0 for all other (i,j,k)
    
    # The zero constraints are very restrictive. The product u[i*m+j]*v[j*p+k]*w[k*n+i]
    # is zero unless all three factors are non-zero. So we need the supports to be "disjoint"
    # in a specific way.
    
    # Let's try: set most entries to 0, and only set the necessary ones.
    # For (i_a,j_a,k_a): need u[i_a*m+j_a] != 0, v[j_a*p+k_a] != 0, w[k_a*n+i_a] != 0
    # For (i_b,j_b,k_b): need u[i_b*m+j_b] != 0, v[j_b*p+k_b] != 0, w[k_b*n+i_b] != 0
    
    # If (i_a,j_a,k_a) == (i_b,j_b,k_b), they're the same term, skip.
    if idx_a == idx_b:
        return None
    
    # Check if the two terms can be covered by a single sparse rank-1 tensor.
    # The key constraint: for any (i,j,k) where i != i_a,i_b OR j != j_a,j_b OR k != k_a,k_b,
    # we need at least one of u[i*m+j], v[j*p+k], w[k*n+i] to be zero.
    
    # Strategy: try to construct u, v, w with minimal support.
    # u has non-zeros at positions {i_a*m+j_a, i_b*m+j_b} (if distinct)
    # v has non-zeros at positions {j_a*p+k_a, j_b*p+k_b} (if distinct)
    # w has non-zeros at positions {k_a*n+i_a, k_b*n+i_b} (if distinct)
    
    # Let's check if this minimal support works.
    u_nz = set([i_a * m + j_a, i_b * m + j_b])
    v_nz = set([j_a * p + k_a, j_b * p + k_b])
    w_nz = set([k_a * n + i_a, k_b * n + i_b])
    
    # For all (i,j,k) not in {(i_a,j_a,k_a), (i_b,j_b,k_b)}, check if the product is zero.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) == (i_a, j_a, k_a) or (i, j, k) == (i_b, j_b, k_b):
                    continue
                # Check if u[i*m+j], v[j*p+k], w[k*n+i] could all be non-zero
                if (i * m + j) in u_nz and (j * p + k) in v_nz and (k * n + i) in w_nz:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Now solve for the values.
    # Let a = u[i_a*m+j_a], b = v[j_a*p+k_a], c = w[k_a*n+i_a]
    # Let d = u[i_b*m+j_b], e = v[j_b*p+k_b], f = w[k_b*n+i_b]
    # Need: a*b*c = 1 and d*e*f = 1
    # We have 6 unknowns and 2 equations. Many solutions exist.
    # Choose simple values: a=1, b=1, c=1, d=1, e=1, f=1
    # But we need to check that the zero constraints are satisfied with these values.
    
    # Actually, the zero constraints are about support, not values. If the support check passed,
    # then setting all non-zero entries to 1 works.
    
    # Build the merged term
    U = np.zeros(n * m)
    V = np.zeros(m * p)
    W = np.zeros(p * n)
    
    U[i_a * m + j_a] = 1.0
    V[j_a * p + k_a] = 1.0
    W[k_a * n + i_a] = 1.0
    
    if i_b * m + j_b != i_a * m + j_a:
        U[i_b * m + j_b] = 1.0
    if j_b * p + k_b != j_a * p + k_a:
        V[j_b * p + k_b] = 1.0
    if k_b * n + i_b != k_a * n + i_a:
        W[k_b * n + i_b] = 1.0
    
    return U, V, W

def _try_structured_rank89(n, m, p, deadline):
    """
    Try to find rank 89 by merging pairs of schoolbook terms.
    Also try a more general approach: start from schoolbook, remove one term,
    and use iterative refinement to adjust the remaining terms.
    """
    # First, try all pairs for merging
    for idx_a in range(n * m * p):
        if time.time() > deadline - 30:
            break
        for idx_b in range(idx_a + 1, n * m * p):
            if time.time() > deadline - 30:
                break
            merged = _try_merge_pair(n, m, p, idx_a, idx_b)
            if merged is not None:
                U_m, V_m, W_m = merged
                # Build the full rank-89 decomposition
                U = np.zeros((n * m, 89), dtype=float)
                V = np.zeros((m * p, 89), dtype=float)
                W = np.zeros((p * n, 89), dtype=float)
                
                # Copy all schoolbook terms except idx_a and idx_b
                col = 0
                for idx in range(n * m * p):
                    if idx == idx_a or idx == idx_b:
                        continue
                    i = idx // (m * p)
                    j = (idx // p) % m
                    k = idx % p
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                
                # Add the merged term
                U[:, col] = U_m
                V[:, col] = V_m
                W[:, col] = W_m
                
                if _verify_decomposition_fast(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_iterative_refinement(n, m, p, target_rank, deadline, max_iter=2000):
    """
    Try to find a rank-target_rank decomposition by iterative refinement.
    Start from a random initialization with the right sparsity pattern,
    then use gradient-like updates to reduce the residual.
    
    The objective is: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    if time.time() > deadline - 10:
        return None
    
    best_residual = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(seed * 7919 + 13)
        
        # Initialize with sparse random values
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)
        
        # Initialize each column with a few non-zero entries
        for r in range(target_rank):
            # Pick 2-4 non-zero entries for each factor
            nz_u = np.random.choice(n * m, size=min(n * m, np.random.randint(2, 5)), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            nz_v = np.random.choice(m * p, size=min(m * p, np.random.randint(2, 5)), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            nz_w = np.random.choice(p * n, size=min(p * n, np.random.randint(2, 5)), replace=False)
            for idx in nz_w:
                W[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        # Iterative refinement: coordinate descent on each entry
        lr = 0.1
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute residual for each (i,j,k)
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] = np.dot(
                            U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]
                        ) - 1.0
            
            res_norm = np.sqrt(np.sum(residual**2))
            if res_norm < best_residual:
                best_residual = res_norm
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if res_norm < 1e-9:
                    return {"U": U, "V": V, "W": W}
            
            # Update U entries
            for i in range(n):
                for j in range(m):
                    for r in range(target_rank):
                        # Gradient of loss w.r.t. U[i*m+j, r]
                        grad = 0.0
                        for k in range(p):
                            grad += 2 * residual[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, r] -= lr * grad
            
            # Update V entries
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        grad = 0.0
                        for i in range(n):
                            grad += 2 * residual[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, r] -= lr * grad
            
            # Update W entries
            for k in range(p):
                for i in range(n):
                    for r in range(target_rank):
                        grad = 0.0
                        for j in range(m):
                            grad += 2 * residual[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                        W[k*n+i, r] -= lr * grad
            
            # Decay learning rate
            if iteration % 50 == 0 and iteration > 0:
                lr *= 0.8
    
    # Check best found
    if best_U is not None and best_residual < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def _try_block_structure(n, m, p, deadline):
    """
    Try a block-structured decomposition. For 3x5x6, consider partitioning
    into blocks and finding low-rank decompositions within blocks.
    
    Alternative: use the fact that matrix multiplication can be decomposed
    as (A*B) where we factor through an intermediate space.
    """
    # This is a placeholder for more sophisticated structural approaches.
    # For now, return None to fall back to other methods.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank = 90)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured rank-89 via term merging
    result89 = _try_structured_rank89(n, m, p, deadline)
    if result89 is not None:
        best_result = result89
        best_rank = 89
    
    # If we have time, try iterative refinement for rank 89
    if time.time() < deadline - 20 and best_rank > 89:
        result_ref = _try_iterative_refinement(n, m, p, 89, deadline)
        if result_ref is not None:
            best_result = result_ref
            best_rank = 89
    
    # If we have time, try rank 88 via iterative refinement
    if time.time() < deadline - 15 and best_rank > 88:
        result_ref = _try_iterative_refinement(n, m, p, 88, deadline)
        if result_ref is not None:
            best_result = result_ref
            best_rank = 88
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
