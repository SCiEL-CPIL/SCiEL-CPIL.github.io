"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t_start = time.time()
    deadline = t_start + 110  # safety margin before 120s
    
    # Strategy: Use structured bilinear forms
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We decompose into rank-1 terms: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Use a block-based approach with Strassen for 2x2 subblocks
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] each 2x2 (since p=6, 3 blocks of 2)
    # C = [[C00, C01, C02], [C10, C11, C12]] each 2x2
    
    # For each 2x2 product X*Y, use Strassen (rank 7):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    rank_terms = []  # list of (U_vec, V_vec, W_vec)
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(3):
                # Strassen terms for A[i1 block, j1 block] * B[j1 block, k1 block]
                
                terms = [
                    # M1: (A[i1*2,j1*2] + A[i1*2+1,j1*2+1]) * (B[j1*2,k1*2] + B[j1*2+1,k1*2+1])
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2+1): 1}),
                    # M2: (A[i1*2+1,j1*2] + A[i1*2+1,j1*2+1]) * B[j1*2,k1*2]
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1},
                     {(i1*2+1, k1*2): 1, (i1*2+1, k1*2+1): -1}),
                    # M3: A[i1*2,j1*2] * (B[j1*2,k1*2+1] - B[j1*2+1,k1*2+1])
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k1*2+1): 1, (j1*2+1, k1*2+1): -1},
                     {(i1*2, k1*2+1): 1, (i1*2+1, k1*2+1): 1}),
                    # M4: A[i1*2+1,j1*2+1] * (B[j1*2+1,k1*2] - B[j1*2,k1*2])
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k1*2): 1, (j1*2, k1*2): -1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2): 1}),
                    # M5: (A[i1*2,j1*2] + A[i1*2,j1*2+1]) * B[j1*2+1,k1*2+1]
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): -1, (i1*2, k1*2+1): 1}),
                    # M6: (A[i1*2+1,j1*2] - A[i1*2,j1*2]) * (B[j1*2,k1*2] + B[j1*2,k1*2+1])
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k1*2): 1, (j1*2, k1*2+1): 1},
                     {(i1*2+1, k1*2+1): 1}),
                    # M7: (A[i1*2,j1*2+1] - A[i1*2+1,j1*2+1]) * (B[j1*2+1,k1*2] + B[j1*2+1,k1*2+1])
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1}),
                ]
                
                for u_dict, v_dict, w_dict in terms:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    # Now build U, V, W matrices
    rank = len(rank_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][r] = coeff
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for r in range(rank):
        u_vec = U_arr[:, r].reshape(n, m)
        v_vec = V_arr[:, r].reshape(m, p)
        w_vec = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
