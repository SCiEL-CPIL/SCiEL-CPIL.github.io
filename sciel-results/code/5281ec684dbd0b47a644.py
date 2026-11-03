import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """
    Verify if U, V, W form a valid decomposition of T.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i, r] V[j, r] W[k, r] == T[i, j, k]
    """
    try:
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if not np.allclose(T_recon, T, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _is_half_integer(arr, tol=1e-9):
    """
    Check if all entries in arr are half-integers (i.e., 2*entry is an integer).
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return np.allclose(scaled, rounded, atol=tol)

def _round_to_half_integer(arr, tol=1e-9):
    """
    Round entries to nearest half-integer.
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return rounded / 2.0

def _strassen_2x2x2_decomposition():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    The decomposition is of the form:
    C = sum_{r=1}^7 U_r * V_r * W_r
    where U_r, V_r, W_r are 2x2 matrices.
    
    The standard Strassen formulas for (A, B) -> C where C = A*B:
    M1 = (A11 - A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to map this to tensor form T[i, j, k] where:
    i indexes A (row-major), j indexes B (row-major), k indexes C (row-major)
    T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    
    Let's define the 2x2 matrices for each term r.
    """
    # We'll construct U, V, W as 4x7 matrices (flattened 2x2)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    
    # Term 1: M1 = (A11 - A22) * (B11 + B22)
    # A part: A11 - A22 -> U[:, 0] = [1, 0, 0, -1]
    # B part: B11 + B22 -> V[:, 0] = [1, 0, 0, 1]
    # C part: M1 contributes to C11 (+1) and C22 (+1)
    # W[:, 0] = [1, 0, 0, 1]
    
    # Term 2: M2 = (A21 + A22) * B11
    # A part: A21 + A22 -> U[:, 1] = [0, 0, 1, 1]
    # B part: B11 -> V[:, 1] = [1, 0, 0, 0]
    # C part: M2 contributes to C21 (+1) and C22 (-1)
    # W[:, 1] = [0, 0, 1, -1]
    
    # Term 3: M3 = A11 * (B12 - B22)
    # A part: A11 -> U[:, 2] = [1, 0, 0, 0]
    # B part: B12 - B22 -> V[:, 2] = [0, 1, 0, -1]
    # C part: M3 contributes to C12 (+1) and C22 (+1)
    # W[:, 2] = [0, 1, 0, 1]
    
    # Term 4: M4 = A22 * (B21 - B11)
    # A part: A22 -> U[:, 3] = [0, 0, 0, 1]
    # B part: B21 - B11 -> V[:, 3] = [-1, 0, 1, 0]
    # C part: M4 contributes to C11 (+1) and C21 (+1)
    # W[:, 3] = [1, 0, 1, 0]
    
    # Term 5: M5 = (A11 + A12) * B22
    # A part: A11 + A12 -> U[:, 4] = [1, 1, 0, 0]
    # B part: B22 -> V[:, 4] = [0, 0, 0, 1]
    # C part: M5 contributes to C11 (-1) and C12 (+1)
    # W[:, 4] = [-1, 1, 0, 0]
    
    # Term 6: M6 = (A21 - A11) * (B11 + B12)
    # A part: A21 - A11 -> U[:, 5] = [-1, 0, 1, 0]
    # B part: B11 + B12 -> V[:, 5] = [1, 1, 0, 0]
    # C part: M6 contributes to C22 (+1)
    # W[:, 5] = [0, 0, 0, 1]
    
    # Term 7: M7 = (A12 - A22) * (B21 + B22)
    # A part: A12 - A22 -> U[:, 6] = [0, 1, 0, -1]
    # B part: B21 + B22 -> V[:, 6] = [0, 0, 1, 1]
    # C part: M7 contributes to C11 (+1)
    # W[:, 6] = [1, 0, 0, 0]
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [-1, 1, 0, 1, 0, 0, -1]  # A22
    ], dtype=np.float64)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1]   # B22
    ], dtype=np.float64)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ], dtype=np.float64)
    
    return U, V, W

def _construct_block_decomposition(n, m, p):
    """
    Construct a block-based decomposition using Strassen for 2x2 blocks
    and schoolbook for the remaining parts.
    
    For 2x4x7:
    We can view A as 2x4 (1x2 blocks of 2x2), B as 4x7 (2x4 blocks of 2x2, 
    but 7 is odd, so we handle 4 columns with Strassen and 3 columns with schoolbook).
    
    Actually, let's be more systematic. We'll decompose the multiplication
    into independent block multiplications.
    
    A is 2x4: split into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    B is 4x7: split into B_left (rows 0-3, cols 0-3) as 2x2x2 blocks, 
             and B_right (rows 0-3, cols 4-6) as 2x3 blocks.
    
    This is getting complicated. Let's use a simpler approach:
    We know Strassen gives rank 7 for 2x2x2.
    We can use this to multiply 2x2 * 2x2 blocks.
    
    For 2x4 * 4x7:
    - Multiply A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]
    - Multiply A[0:2, 2:4] * B[2:4, 0:7] -> C[0:2, 0:7]
    
    For the first term: A is 2x2, B is 2x7.
    We can split B into B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:7].
    - A * B[:, 0:2]: 2x2 * 2x2, use Strassen (rank 7)
    - A * B[:, 2:4]: 2x2 * 2x2, use Strassen (rank 7)
    - A * B[:, 4:6]: 2x2 * 2x2, use Strassen (rank 7)
    - A * B[:, 6:7]: 2x2 * 2x1, use schoolbook (rank 2)
    
    For the second term: A is 2x2, B is 2x7.
    - A * B[:, 0:2]: 2x2 * 2x2, use Strassen (rank 7)
    - A * B[:, 2:4]: 2x2 * 2x2, use Strassen (rank 7)
    - A * B[:, 4:6]: 2x2 * 2x2, use Strassen (rank 7)
    - A * B[:, 6:7]: 2x2 * 2x1, use schoolbook (rank 2)
    
    Total rank: 4*7 + 2 + 4*7 + 2 = 28 + 2 + 28 + 2 = 60.
    This is worse than schoolbook (56). So this approach doesn't help.
    
    Let's try a different strategy: use known low-rank decompositions
    from the literature for small matrix multiplication tensors.
    """
    pass

def _als_decompose(T, R, max_iter=1000, tol=1e-10, seed=0):
    """
    Alternating Least Squares (ALS) for tensor decomposition.
    T: (I, J, K)
    Returns U, V, W of shapes (I, R), (J, R), (K, R) or None if fails.
    """
    I, J, K = T.shape
    
    # Initialize randomly
    np.random.seed(seed)
    U = np.random.randn(I, R)
    V = np.random.randn(J, R)
    W = np.random.randn(K, R)
    
    # Normalize to avoid divergence
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    T_flat = T.reshape(I * J, K)
    
    for it in range(max_iter):
        # Update U
        A_mat = np.zeros((J * K, R))
        for r in range(R):
            A_mat[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
        try:
            U_new, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        B_mat = np.zeros((I * K, R))
        for r in range(R):
            B_mat[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
        T_perm_V = np.transpose(T, (1, 0, 2)).reshape(J, I * K)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat, T_perm_V.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        C_mat = np.zeros((I * J, R))
        for r in range(R):
            C_mat[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
        T_perm_W = np.transpose(T, (2, 0, 1)).reshape(K, I * J)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(C_mat, T_perm_W.T, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        if it % 5 == 0:
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                break
                
    # Final check
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.linalg.norm(T_recon - T)
    if err < 1e-8:
        return U, V, W
    return None

def _try_rank_with_als(T, R, max_restarts=30, time_budget=10.0, start_time=None):
    """
    Try to find a rank-R decomposition using ALS with multiple restarts.
    Returns (U, V, W) if found, None otherwise.
    """
    if start_time is None:
        start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        seed = restart * 7919 + R
        np.random.seed(seed)
        res = _als_decompose(T, R, max_iter=500, tol=1e-10, seed=seed)
        if res is not None:
            U, V, W = res
            # Verify
            if _verify(U, V, W, T, tol=1e-6):
                # Try to round to half-integers
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                if _verify(U_r, V_r, W_r, T, tol=1e-9):
                    return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 130.0  # Leave 20s margin for safety
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    T = _build_tensor(n, m, p)
    
    # For 2x4x7, known bounds suggest rank is between 35 and 56.
    # Let's search from 55 down to 35.
    # We'll use ALS with multiple restarts for each rank.
    
    # Start from a reasonable upper bound.
    # The flattening rank gives an upper bound.
    T_flat = T.reshape(n * m, m * p * p * n)
    U_svd, s, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
    rank_upper = np.sum(s > 1e-10)
    
    # The tensor rank is at most the flattening rank.
    # For this tensor, the flattening rank is likely 56 or close.
    # Let's start from min(rank_upper, 55) down to 35.
    
    current_rank = min(rank_upper, schoolbook_rank - 1)
    min_rank_to_try = 35
    
    # We'll allocate time budget for each rank attempt.
    # Higher ranks are easier, so we can spend less time.
    # Lower ranks are harder, so we spend more time.
    
    while current_rank >= min_rank_to_try:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Time budget for this rank: proportional to how close we are to the lower bound
        # More time for lower ranks.
        time_for_rank = 3.0 + 2.0 * (current_rank - min_rank_to_try) / max(1, (schoolbook_rank - 1 - min_rank_to_try))
        time_for_rank = min(time_for_rank, 15.0)  # Cap at 15s per rank
        
        res = _try_rank_with_als(T, current_rank, max_restarts=20, 
                                  time_budget=time_for_rank, start_time=start_time)
        
        if res is not None:
            U_r, V_r, W_r = res
            best_result = {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            best_rank = current_rank
        
        current_rank -= 1
        
        # If we found a solution, we can be more aggressive with lower ranks
        # by reducing the time per rank to try more ranks.
        if best_rank < schoolbook_rank:
            # Found improvement, reduce time per subsequent rank
            time_for_rank = 2.0
    
    return best_result
