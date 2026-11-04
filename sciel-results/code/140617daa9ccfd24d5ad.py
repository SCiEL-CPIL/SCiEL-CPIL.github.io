import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Strategy: Strassen on 2x2x2 blocks for the first 2 columns of B,
    # schoolbook for the 3rd column.
    # A is 2x3, B is 3x10, C is 2x10
    # Split B into B_top (2x10) and B_bottom (1x10)
    # C = A_left @ B_top + A_right @ B_bottom
    # A_left is 2x2, A_right is 2x1
    # For A_left @ B_top: split B_top into 5 blocks of 2 columns, use Strassen (7 terms each) -> 35 terms
    # For A_right @ B_bottom: schoolbook 2*1*10 = 20 terms
    # Total: 55

    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55

    r = 0  # current rank index

    # Strassen's 7 products for 2x2 x 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4,        m1-m2+m3+m6]]

    # A_left entries: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    #   U indices: a->0, b->1, c->3, d->4
    # B_block entries: B[0,2t]=e, B[0,2t+1]=f, B[1,2t]=g, B[1,2t+1]=h
    #   V indices: e->2t, f->2t+1, g->10+2t, h->10+2t+1
    # C entries: C[0,2t], C[0,2t+1], C[1,2t], C[1,2t+1]
    #   W indices: C[0,2t]->4t, C[1,2t]->4t+1, C[0,2t+1]->4t+2, C[1,2t+1]->4t+3

    # Strassen coefficients:
    # m1: U: a+1, d+1; V: e+1, h+1; W: C[0,2t]+1, C[1,2t+1]+1
    # m2: U: c+1, d+1; V: e+1; W: C[1,2t]+1, C[1,2t+1]-1
    # m3: U: a+1; V: f+1, h-1; W: C[0,2t+1]+1, C[1,2t+1]+1
    # m4: U: d+1; V: g+1, e-1; W: C[0,2t]+1, C[1,2t]+1
    # m5: U: a+1, b+1; V: h+1; W: C[0,2t]-1, C[0,2t+1]+1
    # m6: U: c+1, a-1; V: e+1, f+1; W: C[1,2t+1]+1
    # m7: U: b+1, d-1; V: g+1, h+1; W: C[0,2t]+1

    strassen_u = [
        [(0, 1), (4, 1)],   # m1: a + d
        [(3, 1), (4, 1)],   # m2: c + d
        [(0, 1)],           # m3: a
        [(4, 1)],           # m4: d
        [(0, 1), (1, 1)],   # m5: a + b
        [(3, 1), (0, -1)],  # m6: c - a
        [(1, 1), (4, -1)],  # m7: b - d
    ]

    strassen_v = [
        [(0, 1), (11, 1)],  # m1: e + h  (offsets 0=e, 11=h)
        [(0, 1)],           # m2: e
        [(1, 1), (11, -1)], # m3: f - h  (offsets 1=f, 11=h)
        [(10, 1), (0, -1)], # m4: g - e  (offsets 10=g, 0=e)
        [(11, 1)],          # m5: h
        [(0, 1), (1, 1)],   # m6: e + f
        [(10, 1), (11, 1)], # m7: g + h
    ]

    strassen_w = [
        [(0, 1), (3, 1)],   # m1: C[0,2t] + C[1,2t+1]
        [(1, 1), (3, -1)],  # m2: C[1,2t] - C[1,2t+1]
        [(2, 1), (3, 1)],   # m3: C[0,2t+1] + C[1,2t+1]
        [(0, 1), (1, 1)],   # m4: C[0,2t] + C[1,2t]
        [(0, -1), (2, 1)],  # m5: -C[0,2t] + C[0,2t+1]
        [(3, 1)],           # m6: C[1,2t+1]
        [(0, 1)],           # m7: C[0,2t]
    ]

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[u_row][r] += coeff

            for (v_row_offset, coeff) in strassen_v[s]:
                if v_row_offset == 0:
                    v_row = k0
                elif v_row_offset == 1:
                    v_row = k1
                elif v_row_offset == 10:
                    v_row = 10 + k0
                elif v_row_offset == 11:
                    v_row = 10 + k1
                else:
                    raise ValueError(f"Unexpected v_row_offset {v_row_offset}")
                V[v_row][r] += coeff

            for (w_row_offset, coeff) in strassen_w[s]:
                if w_row_offset == 0:
                    w_row = k0 * 2 + 0
                elif w_row_offset == 1:
                    w_row = k0 * 2 + 1
                elif w_row_offset == 2:
                    w_row = k1 * 2 + 0
                elif w_row_offset == 3:
                    w_row = k1 * 2 + 1
                else:
                    raise ValueError(f"Unexpected w_row_offset {w_row_offset}")
                W[w_row][r] += coeff

            r += 1

    # Schoolbook for A_right (2x1) @ B_bottom (1x10)
    # A[0,2]*B[2,k] -> C[0,k], A[1,2]*B[2,k] -> C[1,k]
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    # Fallback to trivial decomposition
                    rank_trivial = n * m * p
                    U_t = [[0] * rank_trivial for _ in range(n * m)]
                    V_t = [[0] * rank_trivial for _ in range(m * p)]
                    W_t = [[0] * rank_trivial for _ in range(p * n)]
                    for rr in range(rank_trivial):
                        ii = rr // (m * p)
                        rem = rr % (m * p)
                        jj = rem // p
                        kk = rem % p
                        U_t[ii * m + jj][rr] = 1
                        V_t[jj * p + kk][rr] = 1
                        W_t[kk * n + ii][rr] = 1
                    return {"U": U_t, "V": V_t, "W": W_t}

    return {"U": U, "V": V, "W": W}
