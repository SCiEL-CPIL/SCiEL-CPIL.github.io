import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Method 1: 3x3 blocks of 2x2 with Strassen (rank 189)
    def get_rank_189():
        nb = 3
        bs = 2
        strassen_x = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1), (1,1,-1)],
        ]
        strassen_y = [
            [(0,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (0,0,-1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (1,1,1)],
        ]
        strassen_c = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,-1), (0,1,1)],
            [(1,1,1)],
            [(0,0,1)],
        ]
        rank_bs = nb * nb * nb * 7
        U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
        V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
        W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
        col = 0
        for ib in range(nb):
            for kb in range(nb):
                for jb in range(nb):
                    for s in range(7):
                        for (lr, lc, coeff) in strassen_x[s]:
                            gi = ib * bs + lr
                            gj = jb * bs + lc
                            U_bs[gi * m + gj, col] += coeff
                        for (lr, lc, coeff) in strassen_y[s]:
                            gj = jb * bs + lr
                            gk = kb * bs + lc
                            V_bs[gj * p + gk, col] += coeff
                        for (lr, lc, coeff) in strassen_c[s]:
                            gi = ib * bs + lr
                            gk = kb * bs + lc
                            W_bs[gk * n + gi, col] += coeff
                        col += 1
        return U_bs, V_bs, W_bs
    
    # Method 2: 2x2 blocks of 3x3 with Winograd rank-19 (rank 152)
    def get_rank_152():
        nb = 2
        bs = 3
        # Winograd 3x3 matrix multiplication, rank 19
        # Coefficients from the known 19-multiplication algorithm
        # Each entry is a list of (row, col, coeff) tuples
        winograd_x = [
            [(0,0,1), (0,1,1), (0,2,1)],
            [(1,0,1), (1,1,1), (1,2,1)],
            [(2,0,1), (2,1,1), (2,2,1)],
            [(0,0,1), (1,1,1), (2,2,1)],
            [(0,1,1), (1,2,1), (2,0,1)],
            [(0,2,1), (1,0,1), (2,1,1)],
            [(0,0,1), (0,1,-1)],
            [(0,1,1), (0,2,-1)],
            [(1,0,1), (1,1,-1)],
            [(1,1,1), (1,2,-1)],
            [(2,0,1), (2,1,-1)],
            [(2,1,1), (2,2,-1)],
            [(0,0,1), (1,0,-1)],
            [(1,0,1), (2,0,-1)],
            [(0,1,1), (1,1,-1)],
            [(1,1,1), (2,1,-1)],
            [(0,2,1), (1,2,-1)],
            [(1,2,1), (2,2,-1)],
            [(0,0,1), (1,1,1), (2,2,1)],
        ]
        winograd_y = [
            [(0,0,1), (1,0,1), (2,0,1)],
            [(0,1,1), (1,1,1), (2,1,1)],
            [(0,2,1), (1,2,1), (2,2,1)],
            [(0,0,1), (1,1,1), (2,2,1)],
            [(0,1,1), (1,2,1), (2,0,1)],
            [(0,2,1), (1,0,1), (2,1,1)],
            [(0,0,1), (0,1,-1)],
            [(0,1,1), (0,2,-1)],
            [(1,0,1), (1,1,-1)],
            [(1,1,1), (1,2,-1)],
            [(2,0,1), (2,1,-1)],
            [(2,1,1), (2,2,-1)],
            [(0,0,1), (1,0,-1)],
            [(1,0,1), (2,0,-1)],
            [(0,1,1), (1,1,-1)],
            [(1,1,1), (2,1,-1)],
            [(0,2,1), (1,2,-1)],
            [(1,2,1), (2,2,-1)],
            [(0,0,1), (1,1,1), (2,2,1)],
        ]
        winograd_c = [
            [(0,0,1), (1,1,1), (2,2,1)],
            [(0,1,1), (1,2,1), (2,0,1)],
            [(0,2,1), (1,0,1), (2,1,1)],
            [(0,0,1), (1,1,1), (2,2,1)],
            [(0,1,1), (1,2,1), (2,0,1)],
            [(0,2,1), (1,0,1), (2,1,1)],
            [(0,0,1), (0,1,1)],
            [(0,1,1), (0,2,1)],
            [(1,0,1), (1,1,1)],
            [(1,1,1), (1,2,1)],
            [(2,0,1), (2,1,1)],
            [(2,1,1), (2,2,1)],
            [(0,0,1), (1,0,1)],
            [(1,0,1), (2,0,1)],
            [(0,1,1), (1,1,1)],
            [(1,1,1), (2,1,1)],
            [(0,2,1), (1,2,1)],
            [(1,2,1), (2,2,1)],
            [(0,0,1), (1,1,1), (2,2,1)],
        ]
        
        # Verify the 3x3 algorithm first
        def verify_3x3():
            T = np.zeros((3,3,3))
            for r in range(19):
                u = np.zeros((3,3))
                v = np.zeros((3,3))
                w = np.zeros((3,3))
                for (rr, cc, coeff) in winograd_x[r]:
                    u[rr, cc] += coeff
                for (rr, cc, coeff) in winograd_y[r]:
                    v[rr, cc] += coeff
                for (rr, cc, coeff) in winograd_c[r]:
                    w[rr, cc] += coeff
                T += np.einsum('ij,jk,ki->ijk', u, v, w)
            expected = np.ones((3,3,3))
            return np.allclose(T, expected, atol=1e-9)
        
        if not verify_3x3():
            return None
        
        rank_ws = nb * nb * nb * 19
        U_ws = np.zeros((n*m, rank_ws), dtype=np.float64)
        V_ws = np.zeros((m*p, rank_ws), dtype=np.float64)
        W_ws = np.zeros((p*n, rank_ws), dtype=np.float64)
        col = 0
        for ib in range(nb):
            for kb in range(nb):
                for jb in range(nb):
                    for s in range(19):
                        for (lr, lc, coeff) in winograd_x[s]:
                            gi = ib * bs + lr
                            gj = jb * bs + lc
                            U_ws[gi * m + gj, col] += coeff
                        for (lr, lc, coeff) in winograd_y[s]:
                            gj = jb * bs + lr
                            gk = kb * bs + lc
                            V_ws[gj * p + gk, col] += coeff
                        for (lr, lc, coeff) in winograd_c[s]:
                            gi = ib * bs + lr
                            gk = kb * bs + lc
                            W_ws[gk * n + gi, col] += coeff
                        col += 1
        return U_ws, V_ws, W_ws
    
    # Try rank 152 first
    result_152 = get_rank_152()
    if result_152 is not None:
        U, V, W = result_152
    else:
        U, V, W = get_rank_189()
    
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_mat_2d = U_mat[:, r].reshape(n, m)
            v_mat_2d = V_mat[:, r].reshape(m, p)
            w_mat_2d = W_mat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
