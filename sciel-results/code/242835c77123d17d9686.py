import numpy as np
import time
from itertools import product, combinations, permutations
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_fixed_UV_search(n, m, p, deadline=None):
    """
    Fix sparse patterns for U and V, then solve for W via linear systems.
    
    Strategy:
    - Each row of U corresponds to an input A[i,j]. In schoolbook, each row has one 1.
    - Each row of V corresponds to an input B[j,k]. In schoolbook, each row has one 1.
    - We try to merge terms by allowing some rows to have multiple non-zero entries
      or by having different (i,j) pairs map to the same rank dimension.
    
    Key insight: If we fix U and V, the constraints on W are linear:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is linear in W if we define C[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r],
    then sum_r C[i,j,k,r] * W[k*n+i, r] = 1.
    
    But this is still coupled across r. Instead, let's think differently:
    
    For a given rank r decomposition, each "term" r contributes:
    U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    The constraint is that the sum of these rank-1 tensors equals the matmul tensor.
    
    Alternative approach: Try specific structured decompositions.
    
    For 2x3x6, note that the output is 2x6. Each output C[i,k] = sum_j A[i,j]*B[j,k].
    
    Let's try a decomposition where we group terms by output position (i,k).
    For each (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k].
    
    In tensor form: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    
    Let's try rank-1 patterns where:
    - U has a 1 at position (i,j) for some subset of (i,j) pairs
    - V has a 1 at position (j,k) for some subset of (j,k) pairs  
    - W has a 1 at position (k,i) for some subset of (k,i) pairs
    
    And the product U[a,r]*V[b,r]*W[c,r] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i.
    
    This means for each term r, we pick a "path" i->j->k such that:
    - U[a,r]=1 only if a corresponds to (i,j)
    - V[b,r]=1 only if b corresponds to (j,k)
    - W[c,r]=1 only if c corresponds to (k,i)
    
    But this is just schoolbook. To get lower rank, we need to combine multiple paths
    into one term, which requires the contributions to add up correctly.
    
    Let's try a different approach: assume W has a specific structure and solve for U,V.
    
    Actually, let's try a more direct approach:
    1. Assume each row of U has at most 2 non-zeros (instead of 1 in schoolbook)
    2. Assume each row of V has at most 2 non-zeros
    3. Solve for W linearly
    
    This is still complex. Let's try an even simpler approach:
    
    Try to find a decomposition of rank 35, 34, ... by starting from schoolbook
    and trying to merge pairs of terms.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    n, m, p = 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    # Try to find a decomposition with rank < schoolbook_rank
    # by starting from schoolbook and merging terms.
    
    # In schoolbook, term (i,j,k) has:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r is the index of (i,j,k) in the enumeration.
    
    # To merge two terms r1 and r2 into one term, we need:
    # U[:, r_new] = U[:, r1] + U[:, r2] (or some combination)
    # V[:, r_new] = V[:, r1] + V[:, r2]
    # W[:, r_new] = W[:, r1] + W[:, r2]
    # But this doesn't work because the tensor product is not linear in this way.
    
    # Instead, let's try a random search approach:
    # 1. Pick a target rank r < 36
    # 2. Initialize U, V, W with random values
    # 3. Use gradient descent or Newton's method to minimize the error
    
    # But this is computationally expensive. Let's try a more structured approach.
    
    # Key observation: The matmul tensor T has the property that:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # and T[a,b,c] = 0 otherwise.
    
    # For a rank-r decomposition, we need:
    # sum_{r=0}^{r-1} U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Let's try to find a decomposition where U, V, W have integer entries
    # (or half-integers) by solving a system of polynomial equations.
    
    # For small ranks, we can try to enumerate possible sparse patterns.
    
    # Approach: For each candidate rank r from 35 down to some minimum,
    # try to find U, V, W with small integer entries.
    
    # This is a hard problem. Let's try a heuristic:
    # 1. Start with schoolbook (rank 36)
    # 2. Try to eliminate one term at a time by adjusting the others
    
    # For simplicity, let's try rank 35 first.
    
    # Actually, let's try a completely different approach:
    # Use the fact that for matrix multiplication, there are known lower bounds
    # and constructions. For 2x3x6, the rank is at most 36 (schoolbook).
    # It's known that the rank of 2x2x2 matmul is 7 (Strassen), which is less than 8.
    # For larger matrices, the rank can be less than n*m*p.
    
    # Let's try to find a Strassen-like decomposition for 2x3x6.
    
    # This is a research-level problem. For now, let's implement a basic search
    # that tries to find a rank-35 decomposition by perturbing the schoolbook.
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = None, None, None
    
    # Try to find a rank-35 decomposition
    # We'll use a random search with local optimization
    
    def _random_search_for_rank(target_rank, n, m, p, max_iterations=1000, deadline=None):
        """Random search for a decomposition of given rank."""
        if deadline is None:
            deadline = time.time() + 50
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Compute the error
        def compute_error(U, V, W):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            T = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1.0
            return np.linalg.norm(T_dec - T)
        
        initial_error = compute_error(U, V, W)
        best_error = initial_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try a small perturbation
            if np.random.random() < 0.5:
                U[np.random.randint(n * m), np.random.randint(target_rank)] += np.random.randn() * 0.1
            elif np.random.random() < 0.5:
                V[np.random.randint(m * p), np.random.randint(target_rank)] += np.random.randn() * 0.1
            else:
                W[np.random.randint(p * n), np.random.randint(target_rank)] += np.random.randn() * 0.1
            
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        return best_U, best_V, best_W, best_error
    
    # Try random search for rank 35
    if time.time() < deadline - 10:
        U35, V35, W35, err35 = _random_search_for_rank(35, n, m, p, max_iterations=500, deadline=deadline - 10)
        if err35 < 1e-6:
            # Verify and return
            if _verify_decomposition(U35, V35, W35, n, m, p, tol=1e-5):
                return U35, V35, W35, 35
    
    # Try rank 34
    if time.time() < deadline - 10:
        U34, V34, W34, err34 = _random_search_for_rank(34, n, m, p, max_iterations=300, deadline=deadline - 10)
        if err34 < 1e-6:
            if _verify_decomposition(U34, V34, W34, n, m, p, tol=1e-5):
                return U34, V34, W34, 34
    
    # Fallback: return None to use schoolbook
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, try to find a rank-deficient decomposition
    result = _try_fixed_UV_search(n, m, p, deadline=deadline)
    
    if result is not None:
        U, V, W, rank = result
        if time.time() < deadline - 15:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook decomposition
    U, V, W, rank = _schoolbook(n, m, p)
    
    if time.time() < deadline - 15:
        if not _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
