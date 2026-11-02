# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen decompositions.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = 2, 4, 8  # A: (n, m) = (2,4), B: (m, p) = (4,8), C: (n, p) = (2,8)
    
    # Split into 8 2x2 block products:
    # A is 2x4: split columns into A_left (cols 0-1) and A_right (cols 2-3)
    # B is 4x8: split rows into B_top (rows 0-1) and B_bottom (rows 2-3),
    #           split columns into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # C[i, 2d:2d+2] = A_left[:, 0:2] * B[0:2, 2d:2d+2] + A_right[:, 2:4] * B[2:4, 2d:2d+2]
    # That's 2 * 4 = 8 2x2 products, each with rank 7 => total rank 56
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for d in range(4):  # B column pair: 0->(0,1), 1->(2,3), 2->(4,5), 3->(6,7)
        for h in range(2):  # h=0: A left half, B top rows; h=1: A right half, B bottom rows
            base = (d * 2 + h) * 7
            if h == 0:
                # A block: A[i, 0:2], B block: B[0:2, 2d:2d+2], C block: C[i, 2d:2d+2]
                def a_idx(r, c):
                    return r * m + c
                def b_idx(r, c):
                    return r * p + (2 * d + c)
                def c_idx(r, c):
                    return (2 * d + c) * n + r
            else:
                # A block: A[i, 2:4], B block: B[2:4, 2d:2d+2], C block: C[i, 2d:2d+2]
                def a_idx(r, c):
                    return r * m + (2 + c)
                def b_idx(r, c):
                    return (2 + r) * p + (2 * d + c)
                def c_idx(r, c):
                    return (2 * d + c) * n + r
            
            # M1 = (A00 + A11) * (B00 + B11)
            col = base + 0
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M2 = (A10 + A11) * B00
            col = base + 1
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            W[c_idx(1, 1)][col] -= 1.0
            
            # M3 = A00 * (B01 - B11)
            col = base + 2
            U[a_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M4 = A11 * (B10 - B00)
            col = base + 3
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            
            # M5 = (A00 + A01) * B11
            col = base + 4
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            
            # M6 = (A10 - A00) * (B00 + B01)
            col = base + 5
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(0, 0)][col] -= 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M7 = (A01 - A11) * (B10 + B11)
            col = base + 6
            U[a_idx(0, 1)][col] += 1.0
            U[a_idx(1, 1)][col] -= 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0
    
    # Quick verification with random matrices
    try:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            M = np.zeros(rank)
            for r in range(rank):
                M[r] = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] = np.dot(W_arr[k * n + i, :], M)
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                break
    except Exception:
        pass
    
    if time.time() > deadline:
        # Fallback to schoolbook if we somehow ran out of time (shouldn't happen)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
