import numpy as np
import time
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the mult tensor."""
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    rank = U.shape[1]
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_decomp, T, atol=tol)

def _solve_block_2x3x3():
    """
    Solve for a rank-14 decomposition of 2x3x3 matrix multiplication.
    Returns U_block (6,14), V_block (9,14), W_block (6,14) or None.
    """
    n, m, p = 2, 3, 3
    
    # We'll try to find U, V, W with integer/half-integer entries
    # such that the tensor decomposition is exact.
    
    # Strategy: Use the known structure. For 2x3x3, the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W index = k*n+i = k*2+i
    # U index = i*m+j = i*3+j
    # V index = j*p+k = j*3+k
    
    # Let me try a systematic approach: fix U and V with simple patterns,
    # then solve for W.
    
    # Approach: Use 14 rank-1 terms. Let's try to construct them.
    
    # Known result: rank(2x3x3) = 14.
    # Let me try the following construction:
    
    # Products M0..M13:
    # M0 = (a0+a4)(b0+b4)
    # M1 = (a0-a4)(b0-b4)
    # M2 = (a0+a3)(b0+b3)
    # M3 = (a0-a3)(b0-b3)
    # M4 = (a3+a5)(b3+b7)
    # M5 = (a3-a5)(b3-b7)
    # M6 = (a4+a5)(b4+b8)
    # M7 = (a4-a5)(b4-b8)
    # M8 = a1*b1
    # M9 = a2*b2
    # M10 = a1*b4
    # M11 = a2*b7
    # M12 = a5*b8
    # M13 = a0*b6
    
    U_block = np.zeros((6, 14))
    V_block = np.zeros((9, 14))
    
    # M0 = (a0+a4)(b0+b4)
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[4, 0] = 1
    # M1 = (a0-a4)(b0-b4)
    U_block[0, 1] = 1; U_block[4, 1] = -1
    V_block[0, 1] = 1; V_block[4, 1] = -1
    # M2 = (a0+a3)(b0+b3)
    U_block[0, 2] = 1; U_block[3, 2] = 1
    V_block[0, 2] = 1; V_block[3, 2] = 1
    # M3 = (a0-a3)(b0-b3)
    U_block[0, 3] = 1; U_block[3, 3] = -1
    V_block[0, 3] = 1; V_block[3, 3] = -1
    # M4 = (a3+a5)(b3+b7)
    U_block[3, 4] = 1; U_block[5, 4] = 1
    V_block[3, 4] = 1; V_block[7, 4] = 1
    # M5 = (a3-a5)(b3-b7)
    U_block[3, 5] = 1; U_block[5, 5] = -1
    V_block[3, 5] = 1; V_block[7, 5] = -1
    # M6 = (a4+a5)(b4+b8)
    U_block[4, 6] = 1; U_block[5, 6] = 1
    V_block[4, 6] = 1; V_block[8, 6] = 1
    # M7 = (a4-a5)(b4-b8)
    U_block[4, 7] = 1; U_block[5, 7] = -1
    V_block[4, 7] = 1; V_block[8, 7] = -1
    # M8 = a1*b1
    U_block[1, 8] = 1
    V_block[1, 8] = 1
    # M9 = a2*b2
    U_block[2, 9] = 1
    V_block[2, 9] = 1
    # M10 = a1*b4
    U_block[1, 10] = 1
    V_block[4, 10] = 1
    # M11 = a2*b7
    U_block[2, 11] = 1
    V_block[7, 11] = 1
    # M12 = a5*b8
    U_block[5, 12] = 1
    V_block[8, 12] = 1
    # M13 = a0*b6
    U_block[0, 13] = 1
    V_block[6, 13] = 1
    
    # Now solve for W_block.
    # For each output c (W index = k*2+i), we need:
    # C[c] = sum_j A[i*3+j] * B[j*3+k]
    # where i = c % 2, k = c // 2
    
    # The bilinear form: sum_r W[c,r] * M_r where M_r = (U[:,r]·A)(V[:,r]·B)
    # = sum_r W[c,r] * sum_a U[a,r] A[a] * sum_b V[b,r] B[b]
    # = sum_a,b (sum_r W[c,r] U[a,r] V[b,r]) A[a] B[b]
    
    # So we need: sum_r W[c,r] U[a,r] V[b,r] = delta_{a,b} where
    # delta_{a,b} = 1 if a%3==b%3 and a//3==i and b//3==k, else 0
    # i.e., a = i*3+j, b = j*3+k for some j
    
    # This is a linear system in W[c,:].
    # Let's form the matrix: for each c, the system is:
    # sum_r W[c,r] * (U[:,r] ⊗ V[:,r]) = target_c
    
    # target_c is a 54-dim vector (6*9) with 1s at positions (a,b) where
    # a = i*3+j, b = j*3+k for j=0,1,2, and i=c%2, k=c//2
    
    W_block = np.zeros((6, 14))
    
    for c in range(6):
        i = c % 2
        k = c // 2
        
        # Build target vector
        target = np.zeros(54)
        for j in range(3):
            a = i * 3 + j
            b = j * 3 + k
            target[a * 9 + b] = 1.0
        
        # Build system matrix: each row corresponds to (a,b), column to r
        # Entry (a*9+b, r) = U[a,r] * V[b,r]
        sys_matrix = np.zeros((54, 14))
        for r in range(14):
            for a in range(6):
                for b in range(9):
                    sys_matrix[a * 9 + b, r] = U_block[a, r] * V_block[b, r]
        
        # Solve: sys_matrix @ W[c,:] = target
        W_c, residuals, rank_sys, sv = np.linalg.lstsq(sys_matrix, target, rcond=None)
        
        # Check if solution is close to integer/half-integer
        W_rounded = np.round(W_c * 2) / 2
        if np.allclose(W_c, W_rounded, atol=1e-6):
            W_block[c, :] = W_rounded
        else:
            # Check if the residual is small enough
            if residuals.size > 0 and residuals[0] < 1e-8:
                W_block[c, :] = W_c
            else:
                # This decomposition doesn't work
                return None
    
    # Verify the block decomposition
    U_b = U_block
    V_b = V_block
    W_b = W_block
    
    # Check: for random A, B, does C = A*B match?
    rng = np.random.RandomState(12345)
    for trial in range(20):
        A = rng.rand(6)
        B = rng.rand(9)
        C_direct = A.reshape(2, 3) @ B.reshape(3, 3)
        
        # Compute via decomposition
        C_decomp = np.zeros(6)
        for r in range(14):
            M_r = sum(U_b[a, r] * A[a] for a in range(6)) * sum(V_b[b, r] * B[b] for b in range(9))
            for c in range(6):
                C_decomp[c] += W_b[c, r] * M_r
        
        if not np.allclose(C_decomp, C_direct.flatten(), atol=1e-8):
            return None
    
    return U_block, V_block, W_block

def _construct_rank42(n, m, p):
    """Construct rank-42 decomposition for 2x3x9 using 3 blocks of 2x3x3."""
    if (n, m, p) != (2, 3, 9):
        return None
    
    block_result = _solve_block_2x3x3()
    if block_result is None:
        return None
    
    U_block, V_block, W_block = block_result
    
    U_full = np.zeros((6, 42))
    V_full = np.zeros((27, 42))
    W_full = np.zeros((18, 42))
    
    for b in range(3):
        cols = slice(b * 14, (b + 1) * 14)
        
        # U is the same for all blocks
        U_full[:, cols] = U_block
        
        # V: for block b, B columns 3b, 3b+1, 3b+2
        # V index = j*9 + k, where k = 3b + k_local
        # V_block index = j*3 + k_local
        for j in range(3):
            for k_local in range(3):
                v_full_idx = j * 9 + (3 * b + k_local)
                v_block_idx = j * 3 + k_local
                V_full[v_full_idx, cols] = V_block[v_block_idx, :]
        
        # W: for block b, output columns 3b, 3b+1, 3b+2
        # W index = k*2 + i, where k = 3b + k_local
        # W_block index = k_local*2 + i
        for k_local in range(3):
            for i in range(2):
                w_full_idx = (3 * b + k_local) * 2 + i
                w_block_idx = k_local * 2 + i
                W_full[w_full_idx, cols] = W_block[w_block_idx, :]
    
    # Verify full decomposition
    if not _verify_decomposition(U_full, V_full, W_full, 2, 3, 9):
        return None
    
    # Convert to list of lists
    U_list = [[float(U_full[i, r]) for r in range(42)] for i in range(6)]
    V_list = [[float(V_full[i, r]) for r in range(42)] for i in range(27)]
    W_list = [[float(W_full[i, r]) for r in range(42)] for i in range(18)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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

def construct():
    start_time = time.time()
    
    # Try rank-42
    result = _construct_rank42(2, 3, 9)
    if result is not None:
        return result
    
    # Fallback to schoolbook rank 54
    return _schoolbook(2, 3, 9)
