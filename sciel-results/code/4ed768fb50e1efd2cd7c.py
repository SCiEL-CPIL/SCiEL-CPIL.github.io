import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = 145

    def time_left():
        return deadline - (time.time() - start_time)

    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        U = np.zeros((4, 7), dtype=object)
        V = np.zeros((4, 7), dtype=object)
        W = np.zeros((4, 7), dtype=object)
        terms = [
            ([(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,1)]),
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        for r, (u_list, v_list, w_list) in enumerate(terms):
            for (i, j, c) in u_list:
                U[i * 2 + j, r] += c
            for (j, k, c) in v_list:
                V[j * 2 + k, r] += c
            for (k, i, c) in w_list:
                W[k * 2 + i, r] += c
        return U, V, W

    def verify_decomp(U, V, W, nn, mm, pp, tol=1e-9):
        r = U.shape[1]
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        U3 = Uf.T.reshape(nn, mm, r)
        V3 = Vf.T.reshape(mm, pp, r)
        W3 = Wf.T.reshape(pp, nn, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        return err < tol

    # Build 4x4x4 using tensor product of Strassen 2x2x2
    def build_4x4x4():
        sU, sV, sW = strassen_2x2x2()
        r1, r2 = 7, 7
        U = np.zeros((16, 49), dtype=object)
        V = np.zeros((16, 49), dtype=object)
        W = np.zeros((16, 49), dtype=object)
        for r1_idx in range(7):
            for r2_idx in range(7):
                col = r1_idx * 7 + r2_idx
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U[i * 4 + j, col] = sU[i1 * 2 + j1, r1_idx] * sU[i2 * 2 + j2, r2_idx]
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                V[j * 4 + k, col] = sV[j1 * 2 + k1, r1_idx] * sV[j2 * 2 + k2, r2_idx]
                for k1 in range(2):
                    for k2 in range(2):
                        for i1 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                W[k * 4 + i, col] = sW[k1 * 2 + i1, r1_idx] * sW[k2 * 2 + i2, r2_idx]
        return U, V, W

    # Build 4x4x2: 2 blocks of 4x2x2 (rank 4 each) = rank 8
    # 4x2x2: i in 0..3, j in 0..1, k in 0..1. Rank 8.
    # Actually, 4x2x2: can be done with rank 8 (schoolbook) or less?
    # Schoolbook: 4*2*2 = 16. But we can do better.
    # Split i into 2+2: two 2x2x2 blocks, each rank 7. Total 14.
    # Or: i in 0..3, j in 0..1, k in 0..1.
    # T[i,j,k] = 1. This is a 4x2x2 tensor.
    # We can use: U[i, j, r], V[j, k, r], W[k, i, r].
    # For j=0: T[i,0,k] = 1 for all i,k. That's a 4x2 tensor (i,k).
    # For j=1: T[i,1,k] = 1 for all i,k.
    # We can do: for each j, use a 4x2 matrix multiplication? No, it's just a product.
    # Actually, T[i,j,k] = 1 means C[i,k] = sum_j A[i,j] B[j,k].
    # For fixed j, the contribution is A[i,j] B[j,k]. This is rank 1 in (i,k) for each j.
    # So we can use: r = 2 (one for each j). U[i,j,r=j] = 1 if r==j, V[j,k,r=j] = 1 if r==j, W[k,i,r=j] = 1.
    # Wait: sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # If r corresponds to j: U[i,j,r] = 1 if r==j else 0. V[j,k,r] = 1 if r==j else 0.
    # Then sum_r U[i,j,r] V[j,k,r] W[k,i,r] = U[i,j,j] V[j,k,j] W[k,i,j] = 1*1*W[k,i,j].
    # We need W[k,i,j] = 1 for all k,i. So W[k,i,r] = 1 for all k,i,r? No, W is (pp*nn, r) = (2*4, 2).
    # W[k*4+i, r]. If W[k*4+i, r] = 1 for all k,i,r, then it works.
    # Let's verify: sum_r U[i*2+j, r] V[j*2+k, r] W[k*4+i, r].
    # For r=0: U[i*2+j, 0] = 1 if j==0, V[j*2+k, 0] = 1 if j==0, W[k*4+i, 0] = 1.
    #   Product = 1 if j==0, else 0.
    # For r=1: U[i*2+j, 1] = 1 if j==1, V[j*2+k, 1] = 1 if j==1, W[k*4+i, 1] = 1.
    #   Product = 1 if j==1, else 0.
    # Sum = 1 if j==0 or j==1, which is always 1. Correct!
    # So 4x2x2 has rank 2.
    def build_4x2x2():
        U = np.zeros((8, 2), dtype=object)
        V = np.zeros((4, 2), dtype=object)
        W = np.zeros((8, 2), dtype=object)
        for i in range(4):
            U[i*2+0, 0] = 1
            U[i*2+1, 1] = 1
        for k in range(2):
            V[0*2+k, 0] = 1
            V[1*2+k, 1] = 1
        for k in range(2):
            for i in range(4):
                W[k*4+i, 0] = 1
                W[k*4+i, 1] = 1
        return U, V, W

    # 4x4x2 = two 4x2x2 blocks (j=0,1 and j=2,3). Each rank 2. Total rank 4.
    def build_4x4x2():
        U1, V1, W1 = build_4x2x2()
        U2, V2, W2 = build_4x2x2()
        U = np.zeros((16, 4), dtype=object)
        V = np.zeros((8, 4), dtype=object)
        W = np.zeros((8, 4), dtype=object)
        # Block 1: j in 0..1
        for i in range(4):
            for j in range(2):
                for r in range(2):
                    U[i*4+j, r] = U1[i*2+j, r]
        for j in range(2):
            for k in range(2):
                for r in range(2):
                    V[j*2+k, r] = V1[j*2+k, r]
        for k in range(2):
            for i in range(4):
                for r in range(2):
                    W[k*4+i, r] = W1[k*4+i, r]
        # Block 2: j in 2..3
        for i in range(4):
            for j in range(2):
                for r in range(2):
                    U[i*4+(j+2), 2+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                for r in range(2):
                    V[(j+2)*2+k, 2+r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(4):
                for r in range(2):
                    W[k*4+i, 2+r] = W2[k*4+i, r]
        return U, V, W

    # 4x1x4: rank 1. U[i*1+j, 0] = 1, V[j*4+k, 0] = 1, W[k*4+i, 0] = 1.
    # Actually: T[i,0,k] = 1. sum_r U[i,0,r] V[0,k,r] W[k,i,r] = 1.
    # With r=1: U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1. Product = 1. Correct.
    def build_4x1x4():
        U = np.zeros((4, 1), dtype=object)
        V = np.zeros((4, 1), dtype=object)
        W = np.zeros((16, 1), dtype=object)
        for i in range(4):
            U[i, 0] = 1
        for k in range(4):
            V[k, 0] = 1
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0] = 1
        return U, V, W

    # 4x1x2: rank 1. T[i,0,k] = 1 for i in 0..3, k in 0..1.
    def build_4x1x2():
        U = np.zeros((4, 1), dtype=object)
        V = np.zeros((2, 1), dtype=object)
        W = np.zeros((8, 1), dtype=object)
        for i in range(4):
            U[i, 0] = 1
        for k in range(2):
            V[k, 0] = 1
        for k in range(2):
            for i in range(4):
                W[k*4+i, 0] = 1
        return U, V, W

    # Combine: 4x5x6 = 4x4x4 + 4x1x4 + 4x4x2 + 4x1x2
    # j split: 0..3 (4), 4 (1). k split: 0..3 (4), 4..5 (2).
    # Block 1: i=0..3, j=0..3, k=0..3 -> 4x4x4, rank 49
    # Block 2: i=0..3, j=4, k=0..3 -> 4x1x4, rank 1
    # Block 3: i=0..3, j=0..3, k=4..5 -> 4x4x2, rank 4
    # Block 4: i=0..3, j=4, k=4..5 -> 4x1x2, rank 1
    # Total rank: 49 + 1 + 4 + 1 = 55

    def build_full():
        U1, V1, W1 = build_4x4x4()   # rank 49
        U2, V2, W2 = build_4x1x4()   # rank 1
        U3, V3, W3 = build_4x4x2()   # rank 4
        U4, V4, W4 = build_4x1x2()   # rank 1
        r_total = 49 + 1 + 4 + 1     # 55
        n, m, p = 4, 5, 6
        U = np.zeros((n * m, r_total), dtype=object)
        V = np.zeros((m * p, r_total), dtype=object)
        W = np.zeros((p * n, r_total), dtype=object)

        # Block 1: j=0..3, k=0..3, cols 0..48
        for i in range(4):
            for j in range(4):
                for r in range(49):
                    U[i*m+j, r] = U1[i*4+j, r]
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    V[j*p+k, r] = V1[j*4+k, r]
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W[k*n+i, r] = W1[k*4+i, r]

        # Block 2: j=4, k=0..3, cols 49..49
        for i in range(4):
            for r in range(1):
                U[i*m+4, 49+r] = U2[i, r]
        for k in range(4):
            for r in range(1):
                V[4*p+k, 49+r] = V2[k, r]
        for k in range(4):
            for i in range(4):
                for r in range(1):
                    W[k*n+i, 49+r] = W2[k*4+i, r]

        # Block 3: j=0..3, k=4..5, cols 50..53
        for i in range(4):
            for j in range(4):
                for r in range(4):
                    U[i*m+j, 50+r] = U3[i*4+j, r]
        for j in range(4):
            for k in range(2):
                for r in range(4):
                    V[j*p+(k+4), 50+r] = V3[j*2+k, r]
        for k in range(2):
            for i in range(4):
                for r in range(4):
                    W[(k+4)*n+i, 50+r] = W3[k*4+i, r]

        # Block 4: j=4, k=4..5, cols 54..54
        for i in range(4):
            for r in range(1):
                U[i*m+4, 54+r] = U4[i, r]
        for k in range(2):
            for r in range(1):
                V[4*p+(k+4), 54+r] = V4[k, r]
        for k in range(2):
            for i in range(4):
                for r in range(1):
                    W[(k+4)*n+i, 54+r] = W4[k*4+i, r]

        return U, V, W, r_total

    U, V, W, rank = build_full()
    ok = verify_decomp(U, V, W, 4, 5, 6)

    if ok and rank < 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    U, V, W = schoolbook(4, 5, 6)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
