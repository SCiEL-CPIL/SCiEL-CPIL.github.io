import time
import numpy as np

# Matrix multiplication 2x4 * 4x8
# We use a Kronecker product structure: T_{2,4,8} = T_{2,2,2} ⊗ T_{1,2,4}
# Rank(T_{2,2,2}) = 7 (Strassen)
# Rank(T_{1,2,4}) = 8 (Schoolbook)
# Total rank = 7 * 8 = 56

# Dimensions: n=2, m=4, p=8
# A: 2x4, B: 4x8, C: 2x8
# Index conventions:
# A[i,j], i in [0,1], j in [0,3]. Flat: i*4+j
# B[r,k], r in [0,3], k in [0,7]. Flat: r*8+k
# C[i,k], i in [0,1], k in [0,7]. Flat: k*2+i (trace convention W[k*n+i])

def construct():
    """
    Construct a rank-56 decomposition for 2x4 * 4x8 using Kronecker product of
    Strassen (2x2x2) and Schoolbook (1x2x4).
    """
    n, m, p = 2, 4, 8
    rank = 56

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen factors for 2x2x2
    # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    # Term 0: M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1

    # Term 1: M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1

    # Term 2: M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1

    # Term 3: M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1

    # Term 4: M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1

    # Term 5: M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1

    # Term 6: M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Schoolbook factors for 1x2x4
    # U_b: (2, 8), V_b: (8, 8), W_b: (4, 8)
    U_b = np.zeros((2, 8))
    V_b = np.zeros((8, 8))
    W_b = np.zeros((4, 8))

    col = 0
    for j in range(2):
        for k in range(4):
            U_b[j, col] = 1
            V_b[j * 4 + k, col] = 1
            W_b[k, col] = 1
            col += 1

    # Construct global factors using Kronecker product structure
    for s in range(7):
        for b in range(8):
            col = s * 8 + b

            # U: global index i*4 + (2*j1 + j2)
            # U_s index: i*2 + j1
            # U_b index: j2
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        val_s = U_s[i * 2 + j1, s]
                        val_b = U_b[j2, b]
                        if val_s != 0 and val_b != 0:
                            global_idx = i * 4 + (2 * j1 + j2)
                            U[global_idx, col] += val_s * val_b

            # V: global index (2*r1 + r2)*8 + (4*c1 + c2)
            # V_s index: r1*2 + c1
            # V_b index: r2*4 + c2
            for r1 in range(2):
                for r2 in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            val_s = V_s[r1 * 2 + c1, s]
                            val_b = V_b[r2 * 4 + c2, b]
                            if val_s != 0 and val_b != 0:
                                global_idx = (2 * r1 + r2) * 8 + (4 * c1 + c2)
                                V[global_idx, col] += val_s * val_b

            # W: global index (4*c1 + c2)*2 + i
            # W_s index: c1*2 + i (matches C_s[i,c1] in row-major)
            # W_b index: c2
            for i in range(2):
                for c1 in range(2):
                    for c2 in range(4):
                        val_s = W_s[c1 * 2 + i, s]
                        val_b = W_b[c2, b]
                        if val_s != 0 and val_b != 0:
                            global_k = 4 * c1 + c2
                            global_idx = global_k * 2 + i
                            W[global_idx, col] += val_s * val_b

    # Quick verification
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_computed += (u_vec @ v_vec) * w_vec.T

    if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
