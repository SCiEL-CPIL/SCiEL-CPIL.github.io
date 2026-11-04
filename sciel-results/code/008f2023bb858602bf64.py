"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Return exact tensor decomposition for 4x5 * 5x6.
    
    Strategy: Partition the multiplication into 2x2x2 Strassen blocks + schoolbook remainder.
    Split middle dim 5 into 2+3, split rows 4 into 2+2, split cols 6 into 2+2+2.
    
    12 Strassen blocks of 2x2x2 (rank 7 each) = 84
    1 schoolbook block of 4x1x6 (rank 24) = 24
    Total rank: 108
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_offset = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        -> C[i0:i0+2, k0:k0+2]
        """
        nonlocal r_offset
        
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += 1, C01 -= 1, C10 += 1, C11 -= 1
        U[u_idx(0, 0)][r_offset] += 1
        U[u_idx(1, 1)][r_offset] += 1
        V[v_idx(0, 0)][r_offset] += 1
        V[v_idx(1, 1)][r_offset] += 1
        W[w_idx(0, 0)][r_offset] += 1
        W[w_idx(1, 0)][r_offset] += -1
        W[w_idx(0, 1)][r_offset] += 1
        W[w_idx(1, 1)][r_offset] += -1
        r_offset += 1
        
        # M2 = (A10 + A11)B00
        # C10 += 1, C11 -= 1
        U[u_idx(1, 0)][r_offset] += 1
        U[u_idx(1, 1)][r_offset] += 1
        V[v_idx(0, 0)][r_offset] += 1
        W[w_idx(0, 1)][r_offset] += 1
        W[w_idx(1, 1)][r_offset] += -1
        r_offset += 1
        
        # M3 = A00(B01 - B11)
        # C01 += 1, C11 += 1
        U[u_idx(0, 0)][r_offset] += 1
        V[v_idx(0, 1)][r_offset] += 1
        V[v_idx(1, 1)][r_offset] += -1
        W[w_idx(1, 0)][r_offset] += 1
        W[w_idx(1, 1)][r_offset] += 1
        r_offset += 1
        
        # M4 = A11(B10 - B00)
        # C00 += 1, C10 += 1
        U[u_idx(1, 1)][r_offset] += 1
        V[v_idx(1, 0)][r_offset] += 1
        V[v_idx(0, 0)][r_offset] += -1
        W[w_idx(0, 0)][r_offset] += 1
        W[w_idx(0, 1)][r_offset] += 1
        r_offset += 1
        
        # M5 = (A00 + A01)B11
        # C00 -= 1, C01 += 1
        U[u_idx(0, 0)][r_offset] += 1
        U[u_idx(0, 1)][r_offset] += 1
        V[v_idx(1, 1)][r_offset] += 1
        W[w_idx(0, 0)][r_offset] += -1
        W[w_idx(1, 0)][r_offset] += 1
        r_offset += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += 1
        U[u_idx(1, 0)][r_offset] += 1
        U[u_idx(0, 0)][r_offset] += -1
        V[v_idx(0, 0)][r_offset] += 1
        V[v_idx(0, 1)][r_offset] += 1
        W[w_idx(1, 1)][r_offset] += 1
        r_offset += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # C00 += 1
        U[u_idx(0, 1)][r_offset] += 1
        U[u_idx(1, 1)][r_offset] += -1
        V[v_idx(1, 0)][r_offset] += 1
        V[v_idx(1, 1)][r_offset] += 1
        W[w_idx(0, 0)][r_offset] += 1
        r_offset += 1
    
    # 12 Strassen blocks:
    # A rows: 0-1, 2-3
    # A cols (middle): 0-1, 2-3
    # B cols: 0-1, 2-3, 4-5
    
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 0, 4)
    # A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    add_strassen_block(2, 0, 0)
    # A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    add_strassen_block(2, 0, 2)
    # A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    add_strassen_block(2, 0, 4)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0)
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2)
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 2, 4)
    # A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    add_strassen_block(2, 2, 0)
    # A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    add_strassen_block(2, 2, 2)
    # A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    add_strassen_block(2, 2, 4)
    
    # Schoolbook block: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]
    # This is 4x1 * 1x6 = 24 terms
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1
    
    if r_offset != rank:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
