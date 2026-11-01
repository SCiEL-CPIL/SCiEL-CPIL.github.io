"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 150s kill

    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Fast verification using vectorized operations
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification: check all (i,j,k) at once."""
        # Reshape for broadcasting
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # For each c, we need: sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Build the full tensor and check
        rank = U.shape[1]
        # Create tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c]  # shape (n*m,)
            v_col = V[:, c]  # shape (m*p,)
            w_col = W[:, c]  # shape (p*n,)
            # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is: sum_k (v_mat[j,k] * w_mat[k,i]) * u_mat[i,j]
            # = u_mat[i,j] * (v_mat @ w_mat)[j,i]
            vw = v_mat @ w_mat  # shape (m, n)
            T += u_mat * vw.T  # T[i,j] += u_mat[i,j] * vw[j,i]
        # Check T == 1 everywhere
        if np.max(np.abs(T - 1.0)) > tol:
            return False
        return True

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if not verify_fast(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")

    # STRATEGY: Systematic algebraic block decomposition
    # Key insight: For 2x3x7, we can try to merge terms by finding
    # linear combinations of rank-1 tensors that still span the space.
    #
    # Approach: Enumerate "merge patterns" - groups of schoolbook terms
    # that can be replaced by fewer terms using Strassen-like identities.
    #
    # A Strassen-style merge: if we have terms for (i,j,k1) and (i,j,k2),
    # we can sometimes replace them with 1 term using a linear combination,
    # provided the W entries allow it.
    #
    # More concretely: We look for decompositions of the form:
    #   T = sum_{c=1}^{r} U_c * V_c * W_c
    # where each term is a rank-1 tensor (outer product of vectors).
    #
    # For small dimensions, we can try to find such decompositions by:
    # 1. Starting from schoolbook (rank 42)
    # 2. Trying to merge pairs of terms that share structure
    # 3. Using the fact that T[i,j,k]=1 for all (i,j,k) gives 42 constraints
    #    on r*18 variables (r terms * 18 entries per term set)
    #    So we need r*18 >= 42, i.e., r >= 3 (theoretically)
    #    But the constraints are nonlinear, so practical minimum is higher.
    #
    # Practical approach: Try ranks from 30 to 41, and for each rank,
    # use a structured initialization based on known matrix multiplication
    # algorithms (Strassen for 2x2 blocks, then extend to 3x7).

    # Build a Strassen-based decomposition for 2x3x7
    # Strategy: Split the 3x7 into blocks and use Strassen for 2x2x2 subproblems
    
    def build_strassen_hybrid():
        """
        Build a hybrid Strassen/Schoolbook decomposition.
        
        For 2x3x7:
        - The output is 2x7
        - We can think of this as multiplying 2x3 by 3x7
        
        Split the 3x7 matrix into: [3x2 | 3x2 | 3x2 | 3x1]
        Or better: use the structure directly.
        
        Actually, let's use a different approach:
        - We have T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,6}
        - This is the tensor for A(2x3) * B(3x7) = C(2x7)
        
        Strassen for 2x2x2: rank 7 instead of 8.
        We can partition the 3x7 into overlapping 2x2 blocks.
        
        Let's try: partition j dimension into {0,1} and {1,2} (overlapping)
        and k dimension into pairs.
        
        Actually, a cleaner approach:
        - Use Strassen to reduce a 2x2x2 block from rank 8 to rank 7
        - We have 2*3*7 = 42 terms in schoolbook
        - If we can find 5 independent 2x2x2 blocks and apply Strassen to each,
          we save 5 terms: rank 37.
        
        But the blocks overlap in the j dimension (only 3 values), so this is tricky.
        
        Alternative: Use the fact that for fixed i, the problem is 3x7 matrix-vector
        multiplication (viewing B as 3x7 and multiplying by a 3-vector).
        
        Let me try a direct algebraic approach:
        - Express the tensor as a sum of rank-1 terms
        - Use the constraint that T[i,j,k] = 1 for all (i,j,k)
        """
        
        # Try to build a decomposition with rank < 42
        # by merging terms that share the same (i,j) or same (j,k) or same (k,i)
        
        # Key observation: If two terms c1 and c2 have:
        #   U[i1*m+j1, c1] = U[i1*m+j1, c2] = 1 (same i,j)
        #   V[j1*p+k1, c1] = 1, V[j1*p+k2, c2] = 1 (different k)
        #   W[k1*n+i1, c1] = 1, W[k2*n+i1, c2] = 1 (different k)
        # Then we can't simply merge them.
        
        # But if we allow linear combinations:
        #   Term c1: U[i*m+j,c1]=1, V[j*p+k1,c1]=1, W[k1*n+i,c1]=1
        #   Term c2: U[i*m+j,c2]=1, V[j*p+k2,c2]=1, W[k2*n+i,c2]=1
        # We can replace these 2 terms with 3 terms using Strassen-like formulas
        # only if we also have other terms to "absorb" the cross terms.
        
        # For now, let's try a simpler approach:
        # Random search with structured initialization, but much faster verification.
        
        return None  # Placeholder
    
    # FASTER APPROACH: Use the fact that we can verify in O(rank * n*m*p)
    # with vectorized ops, and do a smarter search.
    
    # Strategy: For each target rank, try multiple random initializations
    # but use a much faster verification and a smarter initialization.
    
    np.random.seed(42)
    
    # Try to find a decomposition with rank < 42
    # Use a "perturbation + snap" approach but with faster verification
    
    def try_find_decomposition(target_rank, n_attempts=50):
        """Try to find a valid decomposition of given rank."""
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize
            if target_rank >= 35:
                # Start from schoolbook, drop random terms, perturb
                U_c = np.zeros((n * m, target_rank))
                V_c = np.zeros((m * p, target_rank))
                W_c = np.zeros((p * n, target_rank))
                
                terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                # Use first target_rank terms
                for c in range(target_rank):
                    i, j, k = terms[c]
                    U_c[i * m + j, c] = 1.0
                    V_c[j * p + k, c] = 1.0
                    W_c[k * n + i, c] = 1.0
                
                # Perturb
                U_c += np.random.randn(n * m, target_rank) * 0.05
                V_c += np.random.randn(m * p, target_rank) * 0.05
                W_c += np.random.randn(p * n, target_rank) * 0.05
            else:
                # More aggressive random init
                U_c = np.random.randn(n * m, target_rank) * 0.3
                V_c = np.random.randn(m * p, target_rank) * 0.3
                W_c = np.random.randn(p * n, target_rank) * 0.3
            
            # Quick check: if verification passes, we're done
            if verify_fast(U_c, V_c, W_c):
                return (U_c, V_c, W_c)
            
            # Try snapping to half-integers
            U_snap = np.round(U_c * 2.0) / 2.0
            V_snap = np.round(V_c * 2.0) / 2.0
            W_snap = np.round(W_c * 2.0) / 2.0
            if verify_fast(U_snap, V_snap, W_snap):
                return (U_snap, V_snap, W_snap)
            
            # Try a few gradient steps
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute tensor error
                T = np.zeros((n, m, p))
                for c in range(target_rank):
                    u_col = U_c[:, c]
                    v_col = V_c[:, c]
                    w_col = W_c[:, c]
                    u_mat = u_col.reshape(n, m)
                    v_mat = v_col.reshape(m, p)
                    w_mat = w_col.reshape(p, n)
                    vw = v_mat @ w_mat
                    T += u_mat * vw.T
                
                err = T - 1.0
                max_err = np.max(np.abs(err))
                
                if max_err < 1e-9:
                    if verify_fast(U_c, V_c, W_c):
                        return (U_c, V_c, W_c)
                    U_snap = np.round(U_c * 2.0) / 2.0
                    V_snap = np.round(V_c * 2.0) / 2.0
                    W_snap = np.round(W_c * 2.0) / 2.0
                    if verify_fast(U_snap, V_snap, W_snap):
                        return (U_snap, V_snap, W_snap)
                    break
                
                # Gradient descent step (pick random matrix and entry)
                mat = np.random.choice(['U', 'V', 'W'])
                lr = 0.01
                
                if mat == 'U':
                    row = np.random.randint(0, n * m)
                    col = np.random.randint(0, target_rank)
                    i_val = row // m
                    j_val = row % m
                    # Gradient: 2 * sum_k (T[i,j,k]-1) * V[j*p+k,col] * W[k*n+i,col]
                    grad = 0.0
                    for k in range(p):
                        grad += (T[i_val, j_val, k] - 1.0) * V_c[j_val * p + k, col] * W_c[k * n + i_val, col]
                    U_c[row, col] -= lr * 2.0 * grad
                    
                elif mat == 'V':
                    row = np.random.randint(0, m * p)
                    col = np.random.randint(0, target_rank)
                    j_val = row // p
                    k_val = row % p
                    grad = 0.0
                    for i in range(n):
                        grad += (T[i, j_val, k_val] - 1.0) * U_c[i * m + j_val, col] * W_c[k_val * n + i, col]
                    V_c[row, col] -= lr * 2.0 * grad
                    
                else:
                    row = np.random.randint(0, p * n)
                    col = np.random.randint(0, target_rank)
                    k_val = row // n
                    i_val = row % n
                    grad = 0.0
                    for j in range(m):
                        grad += (T[i_val, j, k_val] - 1.0) * U_c[i_val * m + j, col] * V_c[j * p + k_val, col]
                    W_c[row, col] -= lr * 2.0 * grad
            
            # Final check
            if verify_fast(U_c, V_c, W_c):
                return (U_c, V_c, W_c)
            U_snap = np.round(U_c * 2.0) / 2.0
            V_snap = np.round(V_c * 2.0) / 2.0
            W_snap = np.round(W_c * 2.0) / 2.0
            if verify_fast(U_snap, V_snap, W_snap):
                return (U_snap, V_snap, W_snap)
        
        return None
    
    # Try ranks from 41 down to 30
    for target_rank in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomposition(target_rank, n_attempts=30)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_fast(U_r, V_r, W_r):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                # Try to go lower
                continue
        else:
            # If we couldn't find this rank, we might still try lower
            # but it's unlikely
            if target_rank <= 38:
                break
    
    # Final verification
    if not verify_fast(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
