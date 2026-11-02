"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use block-based Strassen decomposition
    # A is 4x4, B is 4x6, C is 4x6
    # Split A into 2x2 blocks of 2x2 submatrices
    # Split B into 2x3 blocks of 2x2 submatrices
    # Split C into 2x3 blocks of 2x2 submatrices
    
    # C[i1, k1] = A[i1, 0] * B[0, k1] + A[i1, 1] * B[1, k1]
    # where i1 in {0,1}, k1 in {0,1,2}
    # Each product is a 2x2 * 2x2 = 2x2 matrix product
    
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
    
    # Total: 6 C blocks * 2 products per block * 7 Strassen terms = 84 rank-1 terms
    
    rank_terms = []  # list of (U_dict, V_dict, W_dict)
    
    for i1 in range(2):  # block row of A and C
        for j1 in range(2):  # block col of A / block row of B
            for k1 in range(3):  # block col of B and C
                # X = A[i1 block, j1 block] (2x2)
                # Y = B[j1 block, k1 block] (2x2)
                # Z = C[i1 block, k1 block] (2x2)
                
                # Indices in full matrices:
                # X00 = A[i1*2, j1*2], X01 = A[i1*2, j1*2+1]
                # X10 = A[i1*2+1, j1*2], X11 = A[i1*2+1, j1*2+1]
                # Y00 = B[j1*2, k1*2], Y01 = B[j1*2, k1*2+1]
                # Y10 = B[j1*2+1, k1*2], Y11 = B[j1*2+1, k1*2+1]
                # Z00 = C[i1*2, k1*2], Z01 = C[i1*2, k1*2+1]
                # Z10 = C[i1*2+1, k1*2], Z11 = C[i1*2+1, k1*2+1]
                
                terms = [
                    # M1: (X00+X11)(Y00+Y11)
                    # Contributes to: Z00(+1), Z11(+1)
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2+1): 1}),
                    # M2: (X10+X11)Y00
                    # Contributes to: Z10(+1), Z11(-1)
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1},
                     {(i1*2+1, k1*2): 1, (i1*2+1, k1*2+1): -1}),
                    # M3: X00(Y01-Y11)
                    # Contributes to: Z01(+1), Z11(+1)
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k1*2+1): 1, (j1*2+1, k1*2+1): -1},
                     {(i1*2, k1*2+1): 1, (i1*2+1, k1*2+1): 1}),
                    # M4: X11(Y10-Y00)
                    # Contributes to: Z00(+1), Z10(+1)
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k1*2): 1, (j1*2, k1*2): -1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2): 1}),
                    # M5: (X00+X01)Y11
                    # Contributes to: Z00(-1), Z01(+1)
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): -1, (i1*2, k1*2+1): 1}),
                    # M6: (X10-X00)(Y00+Y01)
                    # Contributes to: Z11(+1)
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k1*2): 1, (j1*2, k1*2+1): 1},
                     {(i1*2+1, k1*2+1): 1}),
                    # M7: (X01-X11)(Y10+Y11)
                    # Contributes to: Z00(+1)
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1}),
                ]
                
                for u_dict, v_dict, w_dict in terms:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    # Build U, V, W matrices
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, r] = coeff
    
    # Verify correctness with random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        # C[i,k] += sum_j u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
        C_computed += np.einsum('ij,jk,ki->ik', u_vec, v_vec, w_vec)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
