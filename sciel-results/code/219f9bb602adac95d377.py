import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _build_strassen_rank39(n, m, p):
    """Build the rank-39 Strassen-based decomposition."""
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None:
            u_coeffs = [1] * len(u_indices)
        if v_coeffs is None:
            v_coeffs = [1] * len(v_indices)
        if w_coeffs is None:
            w_coeffs = [1] * len(w_indices)
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        """Add a 2x2x2 Strassen block for output columns k_start, k_start+1."""
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([a00, a11], [b00, b11], [c00, c11])
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
    
    return {"U": U, "V": V, "W": W}

def _find_rank_reduction(U, V, W, n, m, p, tol=1e-9):
    """
    Find the minimal rank by identifying linearly dependent columns.
    Uses the structure of the tensor decomposition to find dependencies.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each column r, we have a term U_r ⊗ V_r ⊗ W_r
    # The decomposition is valid if sum_r U_r ⊗ V_r ⊗ W_r = T (the multiplication tensor)
    # 
    # To find dependencies, we can look for subsets of columns where
    # sum_{r in S} c_r * (U_r ⊗ V_r ⊗ W_r) = 0
    # 
    # This is equivalent to finding the kernel of the map from coefficients
    # to the tensor space.
    
    # Build the tensor T from the decomposition
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Our decomposition gives sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r] = 1
    
    # To find linear dependencies, we need to find vectors c in R^rank such that
    # sum_r c_r * U_t[:,:,r] ⊗ V_t[:,:,r] ⊗ W_t[:,:,r] = 0
    # 
    # This is a 3-linear constraint. We can flatten it.
    # Each term contributes to T[i,j,k] the value sum_r c_r * U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
    # 
    # Let's build a linear system. For each (i,j,k), we have:
    # sum_r c_r * (U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]) = 0
    # 
    # This is linear in c! So we can build an (n*m*p) x rank matrix and find its null space.
    
    nmp = n * m * p
    A = np.zeros((nmp, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    A[idx, r] = U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
    
    # Find the null space of A
    # Use SVD
    try:
        U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    
    # The null space is spanned by the right singular vectors corresponding to zero singular values
    # Find the rank of A
    tol_sv = max(A.shape) * np.finfo(np.float64).eps * s[0] if len(s) > 0 else 0
    matrix_rank = np.sum(s > tol_sv)
    
    null_dim = rank - matrix_rank
    
    if null_dim <= 0:
        return None
    
    # The null space basis is the last null_dim rows of Vt_svd
    null_basis = Vt_svd[matrix_rank:, :].T  # shape (rank, null_dim)
    
    # Now we need to find a subset of columns that can be removed while maintaining validity.
    # If there's a null vector c, we can modify the decomposition by:
    # U_new = U - sum_r c_r * U_r (but this doesn't make sense directly)
    # 
    # Actually, the null space tells us that certain linear combinations of the terms sum to zero.
    # We can use this to eliminate one column at a time.
    # 
    # For a null vector c, we have:
    # sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0
    # 
    # This means we can replace one term with a linear combination of others.
    # Specifically, if c_j != 0, we can write:
    # U_j ⊗ V_j ⊗ W_j = -sum_{r != j} c_r/c_j * (U_r ⊗ V_r ⊗ W_r)
    # 
    # But this doesn't directly give us a way to reduce rank unless we can merge terms.
    # 
    # Alternative approach: try greedy elimination with verification.
    
    # Greedy approach: try removing each column and check if the remaining decomposition
    # can be adjusted to still be valid.
    
    best_result = None
    best_rank = rank
    
    for target_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time.time() > time.time() + 10:  # safety
            break
        
        # Try to find a valid decomposition of rank target_rank
        # by removing (rank - target_rank) columns
        
        # Simple greedy: try removing columns one at a time
        U_curr = [row[:] for row in U]
        V_curr = [row[:] for row in V]
        W_curr = [row[:] for row in W]
        curr_rank = len(U_curr[0])
        
        for _ in range(rank - target_rank):
            removed = False
            for col in range(curr_rank):
                # Try removing column col
                U_test = [row[:col] + row[col+1:] for row in U_curr]
                V_test = [row[:col] + row[col+1:] for row in V_curr]
                W_test = [row[:col] + row[col+1:] for row in W_curr]
                
                if _verify(U_test, V_test, W_test, n, m, p, tol):
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    curr_rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        if len(U_curr[0]) <= target_rank and _verify(U_curr, V_curr, W_curr, n, m, p, tol):
            return {"U": U_curr, "V": V_curr, "W": W_curr}
    
    return None

def _try_advanced_rank_reduction(U, V, W, n, m, p, max_time=60):
    """
    Advanced rank reduction using null space analysis and iterative refinement.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the constraint matrix
    nmp = n * m * p
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    A = np.zeros((nmp, rank), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    A[idx, r] = U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
    
    # Find null space
    try:
        U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    
    tol_sv = max(A.shape) * np.finfo(np.float64).eps * s[0] if len(s) > 0 else 0
    matrix_rank = np.sum(s > tol_sv)
    null_dim = rank - matrix_rank
    
    if null_dim <= 0:
        return None
    
    # Get null space basis
    null_basis = Vt_svd[matrix_rank:, :].T  # (rank, null_dim)
    
    # Try to use null vectors to eliminate columns
    # For each null vector, find which columns can be eliminated
    
    U_curr = [row[:] for row in U]
    V_curr = [row[:] for row in V]
    W_curr = [row[:] for row in W]
    curr_rank = len(U_curr[0])
    
    # Iteratively try to eliminate columns using null space information
    for iteration in range(null_dim):
        if time.time() > deadline:
            break
        
        # Rebuild A with current matrices
        U_arr_curr = np.array(U_curr, dtype=np.float64)
        V_arr_curr = np.array(V_curr, dtype=np.float64)
        W_arr_curr = np.array(W_curr, dtype=np.float64)
        
        curr_rank = len(U_curr[0])
        U_t_curr = U_arr_curr.reshape(n, m, curr_rank)
        V_t_curr = V_arr_curr.reshape(m, p, curr_rank)
        W_t_curr = W_arr_curr.reshape(p, n, curr_rank)
        
        A_curr = np.zeros((nmp, curr_rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    for r in range(curr_rank):
                        A_curr[idx, r] = U_t_curr[i, j, r] * V_t_curr[j, k, r] * W_t_curr[k, i, r]
        
        # Check if any column can be removed
        removed = False
        for col in range(curr_rank):
            if time.time() > deadline:
                break
            
            U_test = [row[:col] + row[col+1:] for row in U_curr]
            V_test = [row[:col] + row[col+1:] for row in V_curr]
            W_test = [row[:col] + row[col+1:] for row in W_curr]
            
            if _verify(U_test, V_test, W_test, n, m, p):
                U_curr = U_test
                V_curr = V_test
                W_curr = W_test
                removed = True
                break
        
        if not removed:
            break
    
    if len(U_curr[0]) < rank and _verify(U_curr, V_curr, W_curr, n, m, p):
        return {"U": U_curr, "V": V_curr, "W": W_curr}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    result = _build_strassen_rank39(n, m, p)
    
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # First, try simple greedy rank reduction
        if time.time() < deadline - 20:
            reduced = _find_rank_reduction(result["U"], result["V"], result["W"], n, m, p)
            if reduced is not None:
                result = reduced
        
        # Then, try advanced null-space-based reduction
        if time.time() < deadline - 10:
            reduced = _try_advanced_rank_reduction(result["U"], result["V"], result["W"], n, m, p, max_time=min(50, deadline - time.time() - 5))
            if reduced is not None:
                result = reduced
        
        return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
