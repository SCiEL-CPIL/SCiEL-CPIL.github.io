import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 28 decomposition using Strassen for 2x2 blocks
    # A (2x4) split into A_L (cols 0-1), A_R (cols 2-3)
    # B (4x4) split into 4 2x2 blocks
    # C (2x4) is sum of products
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    def add_strassen_block(a_col_start, b_row_start, b_col_start, c_col_start):
        nonlocal col
        
        # Indices for X (from A)
        # X is 2x2 at A[:, a_col_start:a_col_start+2]
        # X_00 -> U[0*m + a_col_start]
        # X_01 -> U[0*m + a_col_start + 1]
        # X_10 -> U[1*m + a_col_start]
        # X_11 -> U[1*m + a_col_start + 1]
        
        u_00 = 0 * m + a_col_start
        u_01 = 0 * m + a_col_start + 1
        u_10 = 1 * m + a_col_start
        u_11 = 1 * m + a_col_start + 1
        
        # Indices for Y (from B)
        # Y is 2x2 at B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        # Y_00 -> V[b_row_start*p + b_col_start]
        # Y_01 -> V[b_row_start*p + b_col_start + 1]
        # Y_10 -> V[(b_row_start+1)*p + b_col_start]
        # Y_11 -> V[(b_row_start+1)*p + b_col_start + 1]
        
        v_00 = b_row_start * p + b_col_start
        v_01 = b_row_start * p + b_col_start + 1
        v_10 = (b_row_start + 1) * p + b_col_start
        v_11 = (b_row_start + 1) * p + b_col_start + 1
        
        # Indices for Z (to C)
        # Z is 2x2 at C[:, c_col_start:c_col_start+2]
        # Z_00 -> W[c_col_start*n + 0]
        # Z_01 -> W[(c_col_start+1)*n + 0]
        # Z_10 -> W[c_col_start*n + 1]
        # Z_11 -> W[(c_col_start+1)*n + 1]
        
        w_00 = c_col_start * n + 0
        w_01 = (c_col_start + 1) * n + 0
        w_10 = c_col_start * n + 1
        w_11 = (c_col_start + 1) * n + 1
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[u_00, col] += 1.0
        U[u_11, col] += 1.0
        V[v_00, col] += 1.0
        V[v_11, col] += 1.0
        W[w_00, col] += 1.0
        W[w_11, col] += 1.0
        col += 1
        
        # M2 = (X10 + X11)Y00
        U[u_10, col] += 1.0
        U[u_11, col] += 1.0
        V[v_00, col] += 1.0
        W[w_10, col] += 1.0
        W[w_11, col] -= 1.0
        col += 1
        
        # M3 = X00(Y01 - Y11)
        U[u_00, col] += 1.0
        V[v_01, col] += 1.0
        V[v_11, col] -= 1.0
        W[w_01, col] += 1.0
        W[w_11, col] += 1.0
        col += 1
        
        # M4 = X11(Y10 - Y00)
        U[u_11, col] += 1.0
        V[v_10, col] += 1.0
        V[v_00, col] -= 1.0
        W[w_00, col] += 1.0
        W[w_10, col] += 1.0
        col += 1
        
        # M5 = (X00 + X01)Y11
        U[u_00, col] += 1.0
        U[u_01, col] += 1.0
        V[v_11, col] += 1.0
        W[w_00, col] -= 1.0
        W[w_01, col] += 1.0
        col += 1
        
        # M6 = (X10 - X00)(Y00 + Y01)
        U[u_10, col] += 1.0
        U[u_00, col] -= 1.0
        V[v_00, col] += 1.0
        V[v_01, col] += 1.0
        W[w_11, col] += 1.0
        col += 1
        
        # M7 = (X01 - X11)(Y10 + Y11)
        U[u_01, col] += 1.0
        U[u_11, col] -= 1.0
        V[v_10, col] += 1.0
        V[v_11, col] += 1.0
        W[w_00, col] += 1.0
        col += 1

    # Block 1: A_L @ B_TL -> C_L
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A_R @ B_BL -> C_L
    add_strassen_block(2, 2, 0, 0)
    
    # Block 3: A_L @ B_TR -> C_R
    add_strassen_block(0, 0, 2, 2)
    
    # Block 4: A_R @ B_BR -> C_R
    add_strassen_block(2, 2, 2, 2)
    
    # Verify
    if time.time() - start_time < 10:
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for j in range(m):
                    acc += U[i*m+j] @ (V[j*p+k] * W[k*n+i])
                Cf[i, k] = acc
        if not np.allclose(C, Cf, atol=1e-8):
            # Fallback to rank 32
            rank32 = n * m * p
            U32 = np.zeros((n*m, rank32))
            V32 = np.zeros((m*p, rank32))
            W32 = np.zeros((p*n, rank32))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U32[i*m+j, idx] = 1
                        V32[j*p+k, idx] = 1
                        W32[k*n+i, idx] = 1
                        idx += 1
            return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
