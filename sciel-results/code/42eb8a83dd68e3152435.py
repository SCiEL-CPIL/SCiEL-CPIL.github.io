"""Matrix multiplication 4x4x6 - tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 120
    
    # Build rank-1 tensor terms using Strassen decomposition on 2x2 blocks
    # A: 4x4, B: 4x6, C: 4x6
    # Split into 2x2 blocks: A has 2x2 blocks (2 row-blocks x 2 col-blocks)
    # B has 2x3 blocks (2 row-blocks x 3 col-blocks)
    # C has 2x3 blocks (2 row-blocks x 3 col-blocks)
    
    # For each block product C[i1,k1] = A[i1,j1]*B[j1,k1] + A[i1,j1+1]*B[j1+1,k1]
    # We use Strassen for each 2x2 * 2x2 product, giving 7 terms per product.
    # Each C block needs 2 such products, so 14 terms per C block.
    # 6 C blocks * 14 = 84 total rank-1 terms.
    
    rank_terms = []  # list of (u_dict, v_dict, w_dict)
    
    for i1 in range(2):  # block row of A and C
        for j1 in range(2):  # block col of A = block row of B
            for k1 in range(3):  # block col of B and C
                # 7 Strassen terms for X * Y where X = A[i1*2:(i1+1)*2, j1*2:(j1+1)*2]
                # and Y = B[j1*2:(j1+1)*2, k1*2:(k1+1)*2]
                
                # X00=A[i1*2, j1*2], X01=A[i1*2, j1*2+1]
                # X10=A[i1*2+1, j1*2], X11=A[i1*2+1, j1*2+1]
                # Y00=B[j1*2, k1*2], Y01=B[j1*2, k1*2+1]
                # Y10=B[j1*2+1, k1*2], Y11=B[j1*2+1, k1*2+1]
                
                # M1 = (X00+X11)*(Y00+Y11) -> contributes to Z00(+1), Z11(+1)
                # M2 = (X10+X11)*Y00 -> contributes to Z10(+1), Z11(-1)
                # M3 = X00*(Y01-Y11) -> contributes to Z01(+1), Z11(+1)
                # M4 = X11*(Y10-Y00) -> contributes to Z00(+1), Z10(+1)
                # M5 = (X00+X01)*Y11 -> contributes to Z00(-1), Z01(+1)
                # M6 = (X10-X00)*(Y00+Y01) -> contributes to Z11(+1)
                # M7 = (X01-X11)*(Y10+Y11) -> contributes to Z00(+1)
                
                terms = [
                    # M1
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2+1): 1}),
                    # M2
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1},
                     {(i1*2+1, k1*2): 1, (i1*2+1, k1*2+1): -1}),
                    # M3
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k1*2+1): 1, (j1*2+1, k1*2+1): -1},
                     {(i1*2, k1*2+1): 1, (i1*2+1, k1*2+1): 1}),
                    # M4
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k1*2): 1, (j1*2, k1*2): -1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2): 1}),
                    # M5
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): -1, (i1*2, k1*2+1): 1}),
                    # M6
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k1*2): 1, (j1*2, k1*2+1): 1},
                     {(i1*2+1, k1*2+1): 1}),
                    # M7
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1}),
                ]
                
                for u_dict, v_dict, w_dict in terms:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
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
    
    # Verify correctness with a test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute using the tensor decomposition
    U_mat = np.array(U, dtype=float).T  # shape (rank, n*m)
    V_mat = np.array(V, dtype=float).T  # shape (rank, m*p)
    W_mat = np.array(W, dtype=float).T  # shape (rank, p*n)
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U_mat[r].reshape(n, m)
        v_vec = V_mat[r].reshape(m, p)
        w_vec = W_mat[r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                c = 0.0
                for j in range(m):
                    c += u_vec[i, j] * v_vec[j, k]
                C_computed[i, k] += c * w_vec[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook rank
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
