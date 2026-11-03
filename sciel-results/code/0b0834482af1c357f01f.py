import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen(j_pairs, j_singles, k_pairs):
        rank = 7 * len(j_pairs) * len(k_pairs) + 2 * len(j_singles) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        for j in j_singles:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        return U, V, W, rank

    def eliminate(U, V, W, rank):
        U_list = [row[:] for row in U]
        V_list = [row[:] for row in V]
        W_list = [row[:] for row in W]
        cr = rank
        while cr > 0 and time_left() > 5:
            best_r = -1
            best_nz = float('inf')
            for r in range(cr):
                nz = sum(1 for i in range(n*m) if U_list[i][r] != 0)
                nz += sum(1 for j in range(m*p) if V_list[j][r] != 0)
                nz += sum(1 for k in range(p*n) if W_list[k][r] != 0)
                if nz < best_nz:
                    best_nz = nz
                    best_r = r
            if best_r < 0:
                break
            U_t = np.array(U_list, dtype=float)
            V_t = np.array(V_list, dtype=float)
            W_t = np.array(W_list, dtype=float)
            U_t[:, best_r] = 0
            V_t[:, best_r] = 0
            W_t[:, best_r] = 0
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_t[i*m+j, :], V_t[j*p+k, :] * W_t[k*n+i, :])
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                mask = [r for r in range(cr) if r != best_r]
                U_list = [[U_list[i][r] for r in mask] for i in range(n*m)]
                V_list = [[V_list[j][r] for r in mask] for j in range(m*p)]
                W_list = [[W_list[k][r] for r in mask] for k in range(p*n)]
                cr -= 1
            else:
                break
        return U_list, V_list, W_list, cr

    # Generate j configs: 2 pairs + 1 single from [0,1,2,3,4]
    j_configs = []
    for single in range(5):
        others = [x for x in range(5) if x != single]
        for a in range(4):
            for b in range(a+1, 4):
                c, d = [x for x in range(4) if x != a and x != b]
                if c < d:
                    j_configs.append(([(others[a], others[b]), (others[c], others[d])], [single]))

    # Generate k configs: 3 pairs from [0,1,2,3,4,5]
    k_configs = []
    def gen_k(idx):
        if len(idx) == 0:
            yield []
            return
        if len(idx) == 2:
            yield [(idx[0], idx[1])]
            return
        first = idx[0]
        for i in range(1, len(idx)):
            rest = [x for x in idx if x != first and x != idx[i]]
            for pm in gen_k(rest):
                yield [(first, idx[i])] + pm
    for pm in gen_k(list(range(6))):
        k_configs.append(pm)

    best_rank = float('inf')
    best_result = None

    for j_pairs, j_singles in j_configs:
        if time_left() < 10:
            break
        for k_pairs in k_configs:
            if time_left() < 5:
                break
            U, V, W, rank = build_strassen(j_pairs, j_singles, k_pairs)
            if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
                continue
            U_list, V_list, W_list, new_rank = eliminate(U.tolist(), V.tolist(), W.tolist(), rank)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}

    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
