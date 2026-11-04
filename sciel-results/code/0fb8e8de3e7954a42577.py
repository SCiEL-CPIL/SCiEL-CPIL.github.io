import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        for idx, val in zip(u_idx, u_val):
            u[idx] += val
        for idx, val in zip(v_idx, v_val):
            v[idx] += val
        for idx, val in zip(w_idx, w_val):
            w[idx] += val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def strassen_2x2x2(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        a, b, c, d = i0*m+j0, i1*m+j1, i1*m+j0, i0*m+j1
        e, f, g, h = j0*p+k0, j1*p+k1, j0*p+k1, j1*p+k0
        r, s, t, u_w = k0*n+i0, k1*n+i1, k0*n+i1, k1*n+i0

        add_term([a, b], [1, 1], [e, f], [1, 1], [r, s], [1, 1])
        add_term([d, b], [1, 1], [e], [1], [t, s], [1, -1])
        add_term([a], [1], [h, f], [1, -1], [u_w, s], [1, 1])
        add_term([b], [1], [f, e], [1, -1], [r, t], [1, 1])
        add_term([a, c], [1, 1], [h], [1], [r, u_w], [-1, 1])
        add_term([d, a], [1, -1], [e, h], [1, 1], [t], [1])
        add_term([c, b], [1, -1], [f, g], [1, 1], [r], [1])

    def block(nl, ml, pl, io, jo, ko):
        if nl == 1 and ml == 1 and pl == 1:
            add_term([io*m+jo], [1.0], [jo*p+ko], [1.0], [ko*n+io], [1.0])
            return

        if nl == 1:
            for j in range(ml):
                for k in range(pl):
                    add_term([io*m+(jo+j)], [1.0], [(jo+j)*p+(ko+k)], [1.0], [(ko+k)*n+io], [1.0])
            return

        if ml == 1:
            for i in range(nl):
                for k in range(pl):
                    add_term([(io+i)*m+jo], [1.0], [jo*p+(ko+k)], [1.0], [(ko+k)*n+(io+i)], [1.0])
            return

        if pl == 1:
            if nl == 2 and ml == 2:
                i0, i1, j0, j1, k = io, io+1, jo, jo+1, ko
                add_term([i0*m+j0], [1.0], [j0*p+k], [1.0], [k*n+i0], [1.0])
                add_term([i1*m+j1], [1.0], [j1*p+k], [1.0], [k*n+i1], [1.0])
                return
            for i in range(nl):
                for j in range(ml):
                    add_term([(io+i)*m+(jo+j)], [1.0], [(jo+j)*p+ko], [1.0], [ko*n+(io+i)], [1.0])
            return

        if nl == 2 and ml == 2 and pl == 2:
            strassen_2x2x2(io, jo, ko)
            return

        if nl == 2 and ml == 2 and pl == 3:
            strassen_2x2x2(io, jo, ko)
            add_term([io*m+jo], [1.0], [jo*p+(ko+2)], [1.0], [(ko+2)*n+io], [1.0])
            add_term([(io+1)*m+(jo+1)], [1.0], [(jo+1)*p+(ko+2)], [1.0], [(ko+2)*n+(io+1)], [1.0])
            return

        if nl == 2 and ml == 2 and pl == 4:
            strassen_2x2x2(io, jo, ko)
            strassen_2x2x2(io, jo, ko+2)
            return

        if nl == 3 and ml == 4 and pl == 3:
            block(2, 4, 3, io, jo, ko)
            block(1, 4, 3, io+2, jo, ko)
            return

        if nl == 3 and ml == 4 and pl == 4:
            block(2, 4, 4, io, jo, ko)
            block(1, 4, 4, io+2, jo, ko)
            return

        if nl == 2 and ml == 4 and pl == 3:
            block(2, 2, 3, io, jo, ko)
            block(2, 2, 3, io, jo+2, ko)
            return

        if nl == 2 and ml == 4 and pl == 4:
            block(2, 2, 4, io, jo, ko)
            block(2, 2, 4, io, jo+2, ko)
            return

        # fallback
        for i in range(nl):
            for j in range(ml):
                for k in range(pl):
                    add_term([(io+i)*m+(jo+j)], [1.0], [(jo+j)*p+(ko+k)], [1.0], [(ko+k)*n+(io+i)], [1.0])

    block(n, m, p, 0, 0, 0)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
