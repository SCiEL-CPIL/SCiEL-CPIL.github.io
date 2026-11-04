import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12

    def strassen_2x2x2():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def schoolbook_1x4x2():
        """Rank-8 schoolbook for 1x4 * 4x2 -> 1x2."""
        U = np.zeros((4, 8))
        V = np.zeros((8, 8))
        W = np.zeros((2, 8))
        for j in range(4):
            for k in range(2):
                r = j * 2 + k
                U[j, r] = 1
                V[j * 2 + k, r] = 1
                W[k, r] = 1
        return U, V, W

    def build_rank44():
        rank = 44
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for mid_start in [0, 2]:
            for right_start in [0, 2]:
                sU, sV, sW = strassen_2x2x2()
                for lr in range(7):
                    for li in range(4):
                        lr_row = li // 2
                        lr_col = li % 2
                        gI = lr_row * m + (mid_start + lr_col)
                        U[gI, col + lr] = sU[li, lr]
                    for lj in range(4):
                        lj_row = lj // 2
                        lj_col = lj % 2
                        gJ = (mid_start + lj_row) * p + (right_start + lj_col)
                        V[gJ, col + lr] = sV[lj, lr]
                    for lk in range(4):
                        lk_row = lk // 2
                        lk_col = lk % 2
                        gK = (right_start + lk_row) * n + lk_col
                        W[gK, col + lr] = sW[lk, lr]
                col += 7
        for right_start in [0, 2]:
            sU, sV, sW = schoolbook_1x4x2()
            for lr in range(8):
                for li in range(4):
                    gI = 2 * m + li
                    U[gI, col + lr] = sU[li, lr]
                for lj in range(8):
                    lj_row = lj // 2
                    lj_col = lj % 2
                    gJ = lj_row * p + (right_start + lj_col)
                    V[gJ, col + lr] = sV[lj, lr]
                for lk in range(2):
                    gK = (right_start + lk) * n + 2
                    W[gK, col + lr] = sW[lk, lr]
            col += 8
        return U, V, W

    def verify(U, V, W, R):
        Ui = np.round(np.array(U) * 2).astype(int)
        Vi = np.round(np.array(V) * 2).astype(int)
        Wi = np.round(np.array(W) * 2).astype(int)
        T_approx = np.einsum('ir,jr,kr->ijk', Ui, Vi, Wi)
        T_full = np.zeros((dim_U, dim_V, dim_W), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i*m+j, j*p+k, k*n+i] = 8
        return np.array_equal(T_approx, T_full)

    U44, V44, W44 = build_rank44()
    if verify(U44, V44, W44, 44):
        return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}

    start_time = time.time()
    deadline = start_time + 110
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    for R in [43, 42, 41, 40]:
        if time.time() > deadline:
            break
        np.random.seed(42 + R)
        U = np.zeros((dim_U, R))
        V = np.zeros((dim_V, R))
        W = np.zeros((dim_W, R))
        U[:, :min(R,44)] = U44[:, :min(R,44)]
        V[:, :min(R,44)] = V44[:, :min(R,44)]
        W[:, :min(R,44)] = W44[:, :min(R,44)]
        U += np.random.randn(dim_U, R) * 0.05
        V += np.random.randn(dim_V, R) * 0.05
        W += np.random.randn(dim_W, R) * 0.05
        for it in range(50):
            if time.time() > deadline:
                break
            for i in range(dim_U):
                AU = np.einsum('jr,kr->jkr', V, W).reshape(-1, R)
                try: U[i,:] = np.linalg.lstsq(AU, T[i].flatten(), rcond=None)[0]
                except: pass
            for j in range(dim_V):
                AV = np.einsum('ir,kr->ikr', U, W).reshape(-1, R)
                try: V[j,:] = np.linalg.lstsq(AV, T[:,j,:].flatten(), rcond=None)[0]
                except: pass
            for k in range(dim_W):
                AW = np.einsum('ir,jr->ijr', U, V).reshape(-1, R)
                try: W[k,:] = np.linalg.lstsq(AW, T[:,:,k].flatten(), rcond=None)[0]
                except: pass
            if it % 5 == 0:
                Ur = np.round(U*2)/2.0; Vr = np.round(V*2)/2.0; Wr = np.round(W*2)/2.0
                if verify(Ur, Vr, Wr, R):
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}

    rank = n*m*p
    U = [[0]*rank for _ in range(dim_U)]
    V = [[0]*rank for _ in range(dim_V)]
    W = [[0]*rank for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
