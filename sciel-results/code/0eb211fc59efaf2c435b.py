import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145.0

    # Build a decomposition using Strassen blocks and direct terms
    # We'll use a hybrid approach to minimize rank

    def strassen_decomp(A_r, A_c, B_r, B_c, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2 * 2x2 block."""
        # 7 Strassen terms for 2x2 * 2x2
        a = (A_r, A_c)
        b = (A_r, A_c+1)
        c = (A_r+1, A_c)
        d = (A_r+1, A_c+1)
        e = (B_r, B_c)
        f = (B_r, B_c+1)
        g = (B_r+1, B_c)
        h = (B_r+1, B_c+1)
        
        C00 = (A_r, B_c)
        C01 = (A_r, B_c+1)
        C10 = (A_r+1, B_c)
        C11 = (A_r+1, B_c+1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a,1),(d,1)])
        V_t.append([(e,1),(h,1)])
        W_t.append([(C00,1),(C11,1)])
        
        # m2 = (c+d)e
        U_t.append([(c,1),(d,1)])
        V_t.append([(e,1)])
        W_t.append([(C10,1),(C11,-1)])
        
        # m3 = a(f-h)
        U_t.append([(a,1)])
        V_t.append([(f,1),(h,-1)])
        W_t.append([(C01,1),(C11,1)])
        
        # m4 = d(g-e)
        U_t.append([(d,1)])
        V_t.append([(g,1),(e,-1)])
        W_t.append([(C00,1),(C10,1)])
        
        # m5 = (a+b)h
        U_t.append([(a,1),(b,1)])
        V_t.append([(h,1)])
        W_t.append([(C00,-1),(C01,1)])
        
        # m6 = (c-a)(e+f)
        U_t.append([(c,1),(a,-1)])
        V_t.append([(e,1),(f,1)])
        W_t.append([(C11,1)])
        
        # m7 = (b-d)(g+h)
        U_t.append([(b,1),(d,-1)])
        V_t.append([(g,1),(h,1)])
        W_t.append([(C00,1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i,j),1)])
        V_t.append([((j,k),1)])
        W_t.append([((i,k),1)])

    U_t = []
    V_t = []
    W_t = []

    # Strategy: Use Strassen for the 2x2 blocks where possible
    # A is 3x3, B is 3x8
    # We can partition as:
    # - A[0:2, 0:2] * B[0:2, 0:8] using Strassen for 2x2 * 2x4
    # - A[0:2, 2] * B[2, 0:8] direct
    # - A[2, 0:2] * B[0:2, 0:8] direct
    # - A[2, 2] * B[2, 0:8] direct

    # For 2x2 * 2x4, we can use Strassen twice (for columns 0:2 and 2:4)
    # Actually, let's use a 2x2 * 2x4 decomposition
    # This can be done with 2 Strassen blocks (each 2x2*2x2)

    # Process A[0:2, 0:2] * B[0:2, 0:8]
    # Use Strassen for columns 0:2, 2:4, 4:6, 6:8
    for c in range(0, 8, 2):
        strassen_decomp(0, 0, 0, c, U_t, V_t, W_t)

    # Direct terms for A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)

    # Direct terms for A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)

    # Direct terms for A[2, 2] * B[2, 0:8]
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)

    rank = len(W_t)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
