import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    Falls back to schoolbook (rank 64) if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen's algorithm for 2x2x2 matrix multiplication
    # Standard Strassen: 7 multiplications
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U factors (for A): each row corresponds to A element [A00, A01, A10, A11]
    # V factors (for B): each row corresponds to B element [B00, B01, B10, B11]
    # W factors (for C): each row corresponds to C element [C00, C01, C10, C11]
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1]    # M7: B01+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, -1],  # M2: C10+1, C11-1
        [0, 1, 0, 1],   # M3: C01+1, C11+1
        [1, 0, 1, 0],   # M4: C00+1, C10+1
        [-1, 1, 0, 0],  # M5: C00-1, C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0]    # M7: C00+1
    ]
    
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7  # 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for l in range(num_blocks):
        for s in range(7):
            col = 7 * l + s
            
            # U: A indices are [A00, A01, A10, A11] = [0, 1, 2, 3]
            for a_idx in range(4):
                U[a_idx, col] = strassen_U[s][a_idx]
            
            # V: B indices for block l:
            # B00 -> row 0*p + 2*l = 2*l
            # B01 -> row 0*p + 2*l+1 = 2*l+1
            # B10 -> row 1*p + 2*l = 16+2*l
            # B11 -> row 1*p + 2*l+1 = 16+2*l+1
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            for b_idx in range(4):
                V[b_rows[b_idx], col] = strassen_V[s][b_idx]
            
            # W: C indices for block l (trace convention W[k*n+i, r]):
            # C[i,k] where i=0,1 and k=2*l, 2*l+1
            # C[0,2l] -> W[2l*2+0] = W[4l]
            # C[0,2l+1] -> W[(2l+1)*2+0] = W[4l+2]
            # C[1,2l] -> W[2l*2+1] = W[4l+1]
            # C[1,2l+1] -> W[(2l+1)*2+1] = W[4l+3]
            c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]  # [C00, C01, C10, C11]
            for c_idx in range(4):
                W[c_rows[c_idx], col] = strassen_W[s][c_idx]
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank_val = U_mat.shape[1]
        # Check T[u,v,w] = sum_s U[u,s]*V[v,s]*W[w,s] == 1 for all valid (u,v,w)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.dot(U_mat[u, :], V_mat[v, :] * W_mat[w, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school), dtype=float)
    V_fb = np.zeros((m * p, rank_school), dtype=float)
    W_fb = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
