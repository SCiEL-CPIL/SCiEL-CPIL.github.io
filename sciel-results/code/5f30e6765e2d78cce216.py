import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T = build_tensor()

    def cols_to_matrices(cols):
        rank = len(cols)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r, (u, v, w) in enumerate(cols):
            for idx, val in u.items():
                U[idx, r] = val
            for idx, val in v.items():
                V[idx, r] = val
            for idx, val in w.items():
                W[idx, r] = val
        return U, V, W

    def verify(U, V, W):
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(D, T, atol=1e-9)

    def strassen_2x2x2_cols(i0, j0, k0):
        return [
            ({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
             {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
             {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}),
            ({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
             {j0*p+k0: 1},
             {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}),
            ({i0*m+j0: 1},
             {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
             {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}),
            ({(i0+1)*m+(j0+1): 1},
             {(j0+1)*p+k0: 1, j0*p+k0: -1},
             {k0*n+i0: 1, k0*n+(i0+1): 1}),
            ({i0*m+j0: 1, i0*m+(j0+1): 1},
             {(j0+1)*p+(k0+1): 1},
             {k0*n+i0: -1, (k0+1)*n+i0: 1}),
            ({(i0+1)*m+j0: 1, i0*m+j0: -1},
             {j0*p+k0: 1, j0*p+(k0+1): 1},
             {(k0+1)*n+(i0+1): 1}),
            ({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
             {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
             {k0*n+i0: 1}),
        ]

    def schoolbook_cols(i, j, k):
        return [({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})]

    def decompose(nc, mc, pc, i0, j0, k0):
        cols = []
        def recurse(nc, mc, pc, i0, j0, k0):
            if time.time() > deadline:
                return
            if nc == 1 and mc == 1 and pc == 1:
                cols.extend(schoolbook_cols(i0, j0, k0))
                return
            if nc >= 2 and mc >= 2 and pc >= 2:
                cols.extend(strassen_2x2x2_cols(i0, j0, k0))
                if nc > 2:
                    recurse(nc-2, mc, pc, i0+2, j0, k0)
                if mc > 2:
                    recurse(2, mc-2, pc, i0, j0+2, k0)
                if pc > 2:
                    recurse(2, 2, pc-2, i0, j0, k0+2)
                return
            if nc >= 2:
                recurse(1, mc, pc, i0, j0, k0)
                recurse(nc-1, mc, pc, i0+1, j0, k0)
            elif mc >= 2:
                recurse(nc, 1, pc, i0, j0, k0)
                recurse(nc, mc-1, pc, i0, j0+1, k0)
            elif pc >= 2:
                recurse(nc, mc, 1, i0, j0, k0)
                recurse(nc, mc, pc-1, i0, j0, k0+1)
        recurse(nc, mc, pc, i0, j0, k0)
        return cols

    # Standard Strassen decomposition
    cols = decompose(n, m, p, 0, 0, 0)
    U, V, W = cols_to_matrices(cols)
    
    if not verify(U, V, W):
        # Fallback: pure schoolbook
        cols = [({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})
                for i in range(n) for j in range(m) for k in range(p)]
        U, V, W = cols_to_matrices(cols)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
