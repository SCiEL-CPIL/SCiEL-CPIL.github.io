import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 2, 2, 11
    
    # Rank 39 construction:
    # 5 blocks of 2 columns (Strassen, rank 7 each) = 35
    # 1 column of 1 (Schoolbook, rank 4) = 4
    # Total = 39
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    r = 0

    # Process 5 blocks of 2 columns each (k=0..9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices: B is 2x11, B[i,k] index is i*11 + k
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices: C[i,k] -> W[k*n + i] = W[2k + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3

        # m1 = (A00+A11)(B0k0+B1k1)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1

        # m2 = (A10+A11)B0k0
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1

        # m3 = A00(B0k1-B1k1)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1

        # m4 = A11(B1k0-B0k0)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1

        # m5 = (A00+A01)B1k1
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1

        # m6 = (A10-A00)(B0k0+B0k1)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1

        # m7 = (A01-A11)(B1k0+B1k1)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook (4 multiplications)
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # A00 * B0,10 -> C0,10
        U[0][r] += 1
        V[v_e2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # A01 * B1,10 -> C0,10
        U[1][r] += 1
        V[v_g2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # A10 * B0,10 -> C1,10
        U[2][r] += 1
        V[v_e2][r] += 1
        W[w_c12][r] += 1
        r += 1

        # A11 * B1,10 -> C1,10
        U[3][r] += 1
        V[v_g2][r] += 1
        W[w_c12][r] += 1
        r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
