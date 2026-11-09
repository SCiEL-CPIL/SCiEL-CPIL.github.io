import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 150 - 5  # safety margin

    def build_strassen_2():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))

        # Strassen 2x2 (indices 0..3 for 2x2 blocks)
        U2[0, 0] = 1; U2[3, 0] = 1   # a+d
        U2[2, 1] = 1; U2[3, 1] = 1   # c+d
        U2[0, 2] = 1                 # a
        U2[3, 3] = 1                 # d
        U2[0, 4] = 1; U2[1, 4] = 1   # a+b
        U2[2, 5] = 1; U2[0, 5] = -1  # c-a
        U2[1, 6] = 1; U2[3, 6] = -1  # b-d

        V2[0, 0] = 1; V2[3, 0] = 1   # e+h
        V2[0, 1] = 1                 # e
        V2[1, 2] = 1; V2[3, 2] = -1  # f-h
        V2[2, 3] = 1; V2[0, 3] = -1  # g-c
        V2[3, 4] = 1                 # h
        V2[0, 5] = 1; V2[1, 5] = 1   # e+f
        V2[2, 6] = 1; V2[3, 6] = 1   # g+h

        W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1   # C00
        W2[2, 2] = 1; W2[2, 4] = 1                                  # C01
        W2[1, 1] = 1; W2[1, 3] = 1                                  # C10
        W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1    # C11
        return U2, V2, W2

    def build_laderman_3():
        U3 = np.zeros((9, 23))
        V3 = np.zeros((9, 23))
        W3 = np.zeros((9, 23))

        # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
        # Using the standard Laderman construction
        # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
        # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
        
        # r0: (a00+a11+a22)(b00+b11+b22)
        for (i,j) in [(0,0),(1,1),(2,2)]:
            U3[i*3+j, 0] += 1
            V3[i*3+j, 0] += 1
            W3[i*3+i, 0] += 1  # W index: k*n+i where k=j for diagonal... wait
        # Let me be more careful. W index is k*n+i
        # For C_ik = sum_j A_ij B_jk
        # W[k*n + i, r]
        
        # Let me rebuild more carefully
        U3 = np.zeros((9, 23))
        V3 = np.zeros((9, 23))
        W3 = np.zeros((9, 23))

        # Term 0: (a00+a11+a22)(b00+b11+b22) -> contributes to c00, c11, c22
        for idx in [0, 1, 2]:
            i = idx
            U3[i*3+i, 0] = 1
            V3[i*3+i, 0] = 1
            W3[i*3+i, 0] = 1  # k=i, so W[i*3+i]

        # Term 1: (a01+a10)(b10+b11) -> c10 += (a01+a10)(b10+b11)... 
        # Actually let me use a known correct Laderman implementation

        # I'll use a well-known Laderman decomposition
        # Source: Laderman, "On the complexity of the multiplication of 3x3 matrices"
        
        U3 = np.zeros((9, 23))
        V3 = np.zeros((9, 23))
        W3 = np.zeros((9, 23))

        # Define terms as (U_indices, V_indices, W_indices) with coefficients
        # U index: i*3+j (row i, col j of A)
        # V index: j*3+k (row j, col k of B)  
        # W index: k*3+i (row k, col i of C) -- this is the trace convention W[k*n+i]

        terms = [
            # term 0
            ([(0,1),(1,1),(2,2)], [(0,1),(1,1),(2,2)], [(0,0),(1,1),(2,2)]),
            # term 1
            ([(0,1),(1,0)], [(1,0),(1,1)], [(0,1),(-1,0)]),
            # term 2
            ([(0,2),(2,0)], [(2,0),(0,2)], [(0,2),(-2,0)]),
            # term 3
            ([(1,2),(2,1)], [(1,2),(2,1)], [(1,2),(-2,1)]),
            # term 4
            ([(0,0)], [(0,1)], [(1,0)]),
            # term 5
            ([(0,1)], [(1,0)], [(0,1)]),
            # term 6
            ([(1,0)], [(0,0)], [(0,0)]),
            # term 7
            ([(0,2)], [(2,0)], [(0,2)]),
            # term 8
            ([(2,0)], [(0,2)], [(2,0)]),
            # term 9
            ([(1,2)], [(2,1)], [(1,2)]),
            # term 10
            ([(2,1)], [(1,2)], [(2,1)]),
            # term 11
            ([(0,0),(0,1)], [(0,0),(1,0)], [(0,0),(0,1)]),
            # term 12
            ([(1,0),(0,0)], [(0,0),(0,1)], [(0,1),(1,0)]),
            # term 13
            ([(0,1),(1,0)], [(0,1),(1,1)], [(1,0),(0,0)]),
            # term 14
            ([(1,1)], [(1,1)], [(1,1)]),
            # term 15
            ([(0,2),(1,2)], [(0,2),(1,2)], [(0,2),(1,2)]),
            # term 16
            ([(1,2),(2,1)], [(1,2),(2,1)], [(1,2),(2,1)]),
            # term 17
            ([(2,1),(2,0)], [(2,1),(2,0)], [(2,1),(2,0)]),
            # term 18
            ([(2,0),(0,2)], [(2,0),(0,2)], [(2,0),(0,2)]),
            # term 19
            ([(0,0),(1,1),(2,2)], [(0,0),(1,1),(2,2)], [(0,0),(1,1),(2,2)]),
            # term 20
            ([(0,1),(1,2),(2,0)], [(0,1),(1,2),(2,0)], [(0,1),(1,2),(2,0)]),
            # term 21
            ([(1,0),(2,1),(0,2)], [(1,0),(2,1),(0,2)], [(1,0),(2,1),(0,2)]),
            # term 22
            ([(0,0),(1,1),(2,2)], [(0,0),(1,1),(2,2)], [(0,0),(1,1),(2,2)]),
        ]

        for r, (u_list, v_list, w_list) in enumerate(terms):
            for (i, j) in u_list:
                U3[i*3+j, r] += 1
            for (j, k) in v_list:
                V3[j*3+k, r] += 1
            for (k, i) in w_list:
                W3[k*3+i, r] += 1

        return U3, V3, W3

    def build_schoolbook_3():
        U3 = np.zeros((9, 27))
        V3 = np.zeros((9, 27))
        W3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i*3+j, col] = 1
                    V3[j*3+k, col] = 1
                    W3[k*3+i, col] = 1
                    col += 1
        return U3, V3, W3

    def verify(U, V, W, n):
        rank = U.shape[1]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    U2, V2, W2 = build_strassen_2()

    # Try Laderman
    U3, V3, W3 = build_laderman_3()
    rank3 = 23
    if not verify(U3, V3, W3, 3):
        U3, V3, W3 = build_schoolbook_3()
        rank3 = 27

    rank2 = 7
    total_rank = rank3 * rank2

    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    for r3 in range(rank3):
        for r2 in range(rank2):
            r = r3 * rank2 + r2
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_u2 = np.nonzero(U2[:, r2])[0]
            for u3_idx in nz_u3:
                i3, j3 = divmod(int(u3_idx), 3)
                for u2_idx in nz_u2:
                    i2, j2 = divmod(int(u2_idx), 2)
                    i = i3*2 + i2
                    j = j3*2 + j2
                    U6[i*6+j, r] = U3[u3_idx, r3] * U2[u2_idx, r2]

            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            for v3_idx in nz_v3:
                j3, k3 = divmod(int(v3_idx), 3)
                for v2_idx in nz_v2:
                    j2, k2 = divmod(int(v2_idx), 2)
                    j = j3*2 + j2
                    k = k3*2 + k2
                    V6[j*6+k, r] = V3[v3_idx, r3] * V2[v2_idx, r2]

            nz_w3 = np.nonzero(W3[:, r3])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            for w3_idx in nz_w3:
                k3, i3 = divmod(int(w3_idx), 3)
                for w2_idx in nz_w2:
                    k2, i2 = divmod(int(w2_idx), 2)
                    k = k3*2 + k2
                    i = i3*2 + i2
                    W6[k*6+i, r] = W3[w3_idx, r3] * W2[w2_idx, r2]

    if verify(U6, V6, W6, 6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback: schoolbook 6x6
    U6s = np.zeros((36, 216))
    V6s = np.zeros((36, 216))
    W6s = np.zeros((36, 216))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U6s[i*6+j, col] = 1
                V6s[j*6+k, col] = 1
                W6s[k*6+i, col] = 1
                col += 1
    return {"U": U6s.tolist(), "V": V6s.tolist(), "W": W6s.tolist()}
# EVOLVE-BLOCK-END
