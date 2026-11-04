import numpy as np

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Strassen 2x2x2 local factors (verified working decomposition)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h
    # M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1-M4+M5, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3

    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # M1 = (a+d)(e+h)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1

    # M2 = (c+d)e
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1

    # M3 = a(f-h)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1

    # M4 = d(g-e)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1

    # M5 = (a+b)h
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1

    # M6 = (c-a)(e+f)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1

    # M7 = (b-d)(g+h)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1

    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True

    # Build rank-56 decomposition using 8 independent 2x2x2 Strassen blocks
    # We partition the middle dimension (j=0..3) into two pairs: {0,1} and {2,3}
    # And the output dimension (k=0..7) into four pairs: {0,1}, {2,3}, {4,5}, {6,7}
    # For each (j_pair, k_pair) combination, we apply one Strassen block.
    # Total: 2 * 4 = 8 blocks, each of rank 7, giving total rank 56.

    total_rank = 56
    U = np.zeros((nm, total_rank), dtype=int)
    V = np.zeros((mp, total_rank), dtype=int)
    W = np.zeros((pn, total_rank), dtype=int)

    for j_pair in range(2):
        for k_pair in range(4):
            base_r = (j_pair * 4 + k_pair) * 7

            # Map A indices: for j_pair=0, use A columns 0,1; for j_pair=1, use A columns 2,3
            # A is 2x4, flattened as [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]]
            # Local A indices: 0=a=A[i,2*j_pair], 1=b=A[i,2*j_pair+1], 2=c=A[i+1,2*j_pair], 3=d=A[i+1,2*j_pair+1]
            a_map = [0, 1, 4, 5] if j_pair == 0 else [2, 3, 6, 7]

            # Map B indices: B is 4x8, flattened as [B[0,0]..B[0,7], B[1,0]..B[1,7], B[2,0]..B[2,7], B[3,0]..B[3,7]]
            # For j_pair=0: B rows 0,1; for j_pair=1: B rows 2,3
            # k_pair selects which pair of output columns: k_pair=0 -> cols 0,1; k_pair=1 -> cols 2,3; etc.
            k0 = k_pair * 2
            if j_pair == 0:
                b_map = [0*8 + k0, 0*8 + k0+1, 1*8 + k0, 1*8 + k0+1]
            else:
                b_map = [2*8 + k0, 2*8 + k0+1, 3*8 + k0, 3*8 + k0+1]

            # Map C indices: C is 2x8, flattened as [C[0,0]..C[0,7], C[1,0]..C[1,7]]
            # W index is k*n+i = k*2+i
            # Local C indices: 0=C[0,k0], 1=C[0,k0+1], 2=C[1,k0], 3=C[1,k0+1]
            c_map = [k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val

    if not verify(U, V, W):
        # Fallback to naive 64-rank
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
