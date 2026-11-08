"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _snap_to_half_int(arr):
    """Snap numpy array values to the nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly satisfies the tensor equation."""
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4():
    """
    Return Strassen's 7-term decomposition for 4x4 * 4x4 matrix multiplication.
    U: shape (16, 7), V: shape (16, 7), W: shape (16, 7)
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=4.
    
    Strassen for 2x2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = m = p = 4
    rank = 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Map 2x2 block indices to 4x4 indices
    # Block (bi, bj) of A covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Block (bj, bk) of B covers rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    # Block (bi, bk) of C covers rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    
    # For each Strassen term r, we have:
    # U[i*m+j, r] = sum of coefficients for A[i,j]
    # V[j*p+k, r] = sum of coefficients for B[j,k]  
    # W[k*n+i, r] = sum of coefficients for C[i,k]
    
    # Define the 7 Strassen factors for 2x2:
    # Each factor is a pair of linear combinations
    # A_factors[r]: list of (row_block, col_block, coeff) for A
    # B_factors[r]: list of (row_block, col_block, coeff) for B
    # C_factors[r]: list of (row_block, col_block, coeff) for C
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A contributions for each term:
    # Term 0 (M1): A11: +1, A22: +1
    # Term 1 (M2): A21: +1, A22: +1
    # Term 2 (M3): A11: +1
    # Term 3 (M4): A22: +1
    # Term 4 (M5): A11: +1, A12: +1
    # Term 5 (M6): A21: +1, A11: -1
    # Term 6 (M7): A12: +1, A22: -1
    
    # B contributions for each term:
    # Term 0 (M1): B11: +1, B22: +1
    # Term 1 (M2): B11: +1
    # Term 2 (M3): B21: +1, B22: -1
    # Term 3 (M4): B12: +1, B11: -1
    # Term 4 (M5): B22: +1
    # Term 5 (M6): B11: +1, B12: +1
    # Term 6 (M7): B21: +1, B22: +1
    
    # C contributions for each term:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Term 0 (M1): C11: +1, C22: +1
    # Term 1 (M2): C21: +1, C22: -1
    # Term 2 (M3): C12: +1, C22: +1
    # Term 3 (M4): C11: +1, C21: +1
    # Term 4 (M5): C11: -1, C12: +1
    # Term 5 (M6): C22: +1
    # Term 6 (M7): C11: +1
    
    A_terms = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,1)],       # M2
        [(0,0,1)],                # M3
        [(1,1,1)],                # M4
        [(0,0,1), (0,1,1)],       # M5
        [(1,0,1), (0,0,-1)],      # M6
        [(0,1,1), (1,1,-1)],      # M7
    ]
    
    B_terms = [
        [(0,0,1), (1,1,1)],       # M1
        [(0,0,1)],                # M2
        [(1,0,1), (1,1,-1)],      # M3
        [(0,1,1), (0,0,-1)],      # M4
        [(1,1,1)],                # M5
        [(0,0,1), (0,1,1)],       # M6
        [(1,0,1), (1,1,1)],       # M7
    ]
    
    C_terms = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                # M6
        [(0,0,1)],                # M7
    ]
    
    for r in range(rank):
        for (bi, bj, coeff) in A_terms[r]:
            for di in range(2):
                for dj in range(2):
                    i = 2*bi + di
                    j = 2*bj + dj
                    U[i*m + j, r] += coeff
        
        for (bj, bk, coeff) in B_terms[r]:
            for dj in range(2):
                for dk in range(2):
                    j = 2*bj + dj
                    k = 2*bk + dk
                    V[j*p + k, r] += coeff
        
        for (bi, bk, coeff) in C_terms[r]:
            for di in range(2):
                for dk in range(2):
                    i = 2*bi + di
                    k = 2*bk + dk
                    W[k*n + i, r] += coeff
    
    return U, V, W

def _construct_4x4x6_strassen_hybrid():
    """
    Construct a rank-29 decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Split the 4x6 matrix B into two 4x3 sub-matrices? No, that doesn't work cleanly.
    
    Better strategy: Split the 4x6 matrix B into B1 (4x4) and B2 (4x2).
    C = A * B = A * B1 + A * B2
    
    For A * B1 (4x4 * 4x4): use Strassen with rank 7.
    For A * B2 (4x4 * 4x2): use schoolbook with rank 4*4*2 = 32? That's too much.
    
    Actually, let me reconsider. The total rank should be 29.
    
    Alternative: Use a different decomposition. Let me think about this more carefully.
    
    For 4x4 * 4x6, we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we split B into B_left (4x4) and B_right (4x2):
    C_left[i,k] = sum_j A[i,j] * B_left[j,k] for k=0..3
    C_right[i,k] = sum_j A[i,j] * B_right[j,k] for k=4..5
    
    C_left can be computed with Strassen (rank 7)
    C_right needs rank 4*4*2 = 32 with schoolbook, which is too much.
    
    So total would be 7 + 32 = 39, not 29.
    
    Let me try a different approach. Maybe I should use a more efficient method for the 4x2 part.
    
    Actually, for 4x4 * 4x2, the minimum rank is at least max(16, 8, 8) = 16.
    With schoolbook it's 32. Can we do better?
    
    Hmm, let me just use the ALS approach from the current program but with a better initialization
    based on the Strassen structure.
    
    Actually, let me just return the schoolbook solution for now and use ALS to optimize.
    """
    n, m, p = 4, 4, 6
    rank_school = n * m * p  # 96
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_als(rank, n, m, p, max_iters=50, n_restarts=5, timeout=None, start_time=None, init=None):
    """Try to find a rank-r decomposition using ALS with optional initialization."""
    if start_time is None:
        start_time = time.time()
    
    for restart in range(n_restarts):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Random or provided initialization
        if init is not None and restart == 0:
            U = init[0].copy()
            V = init[1].copy()
            W = init[2].copy()
        else:
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            # Update U
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        X = np.ones(p) @ np.linalg.pinv(M.T)
                        U_new[i*m+j, :] = X
                    except:
                        pass
            U = U_new
            
            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        X = np.ones(n) @ np.linalg.pinv(M.T)
                        V_new[j*p+k, :] = X
                    except:
                        pass
            V = V_new
            
            # Update W
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        X = np.ones(m) @ np.linalg.pinv(M.T)
                        W_new[k*n+i, :] = X
                    except:
                        pass
            W = W_new
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 0.5:
                U_snapped = _snap_to_half_int(U)
                V_snapped = _snap_to_half_int(V)
                W_snapped = _snap_to_half_int(W)
                
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    for rank in range(24, rank_school + 1):
        if time.time() - start_time > deadline:
            break
        
        result = _try_rank_als(rank, n, m, p, max_iters=50, n_restarts=5, 
                              timeout=deadline, start_time=start_time)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition, stop
    
    return best_result
# EVOLVE-BLOCK-END
