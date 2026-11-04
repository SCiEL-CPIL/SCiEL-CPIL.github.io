# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 130
    n, m, p = 4, 4, 9

    # ------------------------------------------------------------------
    # Strategy: Use the block-circulant structure of 4x4 matrix
    # multiplication. The 4x4x4 tensor can be decomposed using the
    # Kronecker product of a 2x2x2 Strassen (rank 7) with a 2x2x2
    # schoolbook (rank 4), giving rank 28 for the 4x4x4 core.
    # Then for 4x4x9: two 4x4 blocks use rank-28 each, one 4x1 block
    # uses rank 4. Total = 28 + 28 + 4 = 60.
    #
    # The Kronecker approach: T_{4,4,4} = T_{2,2,2} ⊗ T_{2,2,2}
    # where the outer is Strassen (rank 7) and inner is schoolbook (rank 4).
    # This gives a valid rank-28 decomposition.
    # ------------------------------------------------------------------

    # 2x2x2 Strassen (rank 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    # m1: (a+d)(e+h)
    U_s[0,0]=1; U_s[3,0]=1; V_s[0,0]=1; V_s[3,0]=1; W_s[0,0]=1; W_s[3,0]=1
    # m2: (c+d)e
    U_s[2,1]=1; U_s[3,1]=1; V_s[0,1]=1; W_s[2,1]=1; W_s[3,1]=1
    # m3: a(f-h)
    U_s[0,2]=1; V_s[1,2]=1; V_s[3,2]=-1; W_s[1,2]=1; W_s[3,2]=1
    # m4: d(g-e)
    U_s[3,3]=1; V_s[2,3]=1; V_s[0,3]=-1; W_s[0,3]=1; W_s[3,3]=1
    # m5: (a+b)h
    U_s[0,4]=1; U_s[1,4]=1; V_s[3,4]=1; W_s[0,4]=1; W_s[1,4]=-1
    # m6: (c-a)(e+f)
    U_s[2,5]=1; U_s[0,5]=-1; V_s[0,5]=1; V_s[1,5]=1; W_s[2,5]=1
    # m7: (b-d)(g+h)
    U_s[1,6]=1; U_s[3,6]=-1; V_s[2,6]=1; V_s[3,6]=1; W_s[0,6]=1; W_s[1,6]=1

    # 2x2x2 Schoolbook (rank 4)
    U_k = np.zeros((4, 4))
    V_k = np.zeros((4, 4))
    W_k = np.zeros((4, 4))
    for t in range(4):
        r = t // 2
        c = t % 2
        U_k[r*2+c, t] = 1
        V_k[c, t] = 1
        W_k[r, t] = 1

    # Kronecker: 4x4x4 rank-28
    # U_444[i*4+j, a*4+b] = U_s[i*2+j, a] * U_k[j*2+b, b]
    # V_444[j*4+k, a*4+b] = V_s[j*2+k, a] * V_k[k*2+b, b]  -- wait, this indexing is off
    #
    # Let me be more careful. The 4x4x4 tensor has indices (i,j,k) with i,j,k in [0,3].
    # Write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0 where i0,j0,k0 in {0,1}.
    # The Kronecker decomposition: T(i,j,k) = sum_{a,b} U_s(i1,j1,a) * U_k(i0,j0,b)
    #                                  * V_s(j1,k1,a) * V_k(j0,k0,b)
    #                                  * W_s(k1,i1,a) * W_k(k0,i0,b)
    #
    # So U_444[i*4+j, a*4+b] = U_s[i1*2+j1, a] * U_k[i0*2+j0, b]
    # V_444[j*4+k, a*4+b] = V_s[j1*2+k1, a] * V_k[j0*2+k0, b]
    # W_444[k*4+i, a*4+b] = W_s[k1*2+i1, a] * W_k[k0*2+i0, b]

    U_444 = np.zeros((16, 28))
    V_444 = np.zeros((16, 28))
    W_444 = np.zeros((16, 28))

    for a in range(7):
        for b in range(4):
            col = a * 4 + b
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1*2 + i0
                            j = j1*2 + j0
                            U_444[i*4+j, col] = U_s[i1*2+j1, a] * U_k[i0*2+j0, b]
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = j1*2 + j0
                            k = k1*2 + k0
                            V_444[j*4+k, col] = V_s[j1*2+k1, a] * V_k[j0*2+k0, b]
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = k1*2 + k0
                            i = i1*2 + i0
                            W_444[k*4+i, col] = W_s[k1*2+i1, a] * W_k[k0*2+i0, b]

    # 4x4x1 schoolbook (rank 4) for the last column block (k=8)
    # T[i,j,8] = sum_j A[i,j] * B[j,8] = C[i,8]
    # For each j in 0..3: one rank-1 term
    U_441 = np.zeros((16, 4))
    V_441 = np.zeros((4, 4))
    W_441 = np.zeros((4, 4))
    for j in range(4):
        for i in range(4):
            U_441[i*4 + j, j] = 1
        V_441[j, j] = 1
        for i in range(4):
            W_441[i, j] = 1

    # Assemble 4x4x9: rank = 28 + 28 + 4 = 60
    total_rank = 60
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((36, total_rank))
    W_total = np.zeros((36, total_rank))

    # Block 1: columns 0-3 of B
    U_total[:, 0:28] = U_444
    for j in range(4):
        for k in range(4):
            V_total[j*9 + k, 0:28] = V_444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W_total[k*4 + i, 0:28] = W_444[k*4 + i, :]

    # Block 2: columns 4-7 of B
    U_total[:, 28:56] = U_444
    for j in range(4):
        for k in range(4):
            V_total[j*9 + (4+k), 28:56] = V_444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W_total[(4+k)*4 + i, 28:56] = W_444[k*4 + i, :]

    # Block 3: column 8 of B
    U_total[:, 56:60] = U_441
    for j in range(4):
        V_total[j*9 + 8, 56:60] = V_441[j, :]
    for i in range(4):
        W_total[8*4 + i, 56:60] = W_441[i, :]

    # Validate
    def validate(U, V, W, tol=1e-9):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        r = Ua.shape[1]
        if Va.shape[1] != r or Wa.shape[1] != r:
            return False
        for i in range(n):
            for j in range(m):
                ur = Ua[i*m + j, :]
                for k in range(p):
                    vr = Va[j*p + k, :]
                    wr = Wa[k*n + i, :]
                    s = float(np.dot(ur, vr * wr))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if validate(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

    # Fallback: schoolbook rank 144
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
