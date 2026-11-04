import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    deadline = time.time() + 118

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, tol=1e-6):
        rank = len(U[0])
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        for t in range(rank):
            u = np.array(U[:, t])
            v = np.array(V[:, t])
            w = np.array(W[:, t])
            T += u[:, None, None] * v[None, :, None] * w[None, None, :]
        target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_mask[i*m+j, j*p+k, k*n+i] = True
        max_err = np.max(np.abs(T[target_mask] - 1.0))
        non_target_err = np.max(np.abs(T[~target_mask]))
        return max_err < tol and non_target_err < tol

    best_result = schoolbook()
    best_rank = 72

    targets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    target_idx = {(i, j, k): t for t, (i, j, k) in enumerate(targets)}
    n_targets = len(targets)

    def make_vecs(Iset, Jset, Kset):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for i in Iset:
            for j in Jset:
                u[i*m + j] = 1.0
        for j in Jset:
            for k in Kset:
                v[j*p + k] = 1.0
        for k in Kset:
            for i in Iset:
                w[k*n + i] = 1.0
        cov = np.zeros(n_targets, dtype=int)
        for (i, j, k) in targets:
            if u[i*m+j] and v[j*p+k] and w[k*n+i]:
                cov[target_idx[(i,j,k)]] = 1
        return cov, u, v, w

    candidates = []
    for (i, j, k) in targets:
        cov, u, v, w = make_vecs({i}, {j}, {k})
        candidates.append((cov, u, v, w))
    for j in range(m):
        for k in range(p):
            for i1 in range(n):
                for i2 in range(i1+1, n):
                    cov, u, v, w = make_vecs({i1, i2}, {j}, {k})
                    if np.sum(cov) >= 2:
                        candidates.append((cov, u, v, w))
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    cov, u, v, w = make_vecs({i}, {j}, {k1, k2})
                    if np.sum(cov) >= 2:
                        candidates.append((cov, u, v, w))
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(j1+1, m):
                    cov, u, v, w = make_vecs({i}, {j1, j2}, {k})
                    if np.sum(cov) >= 2:
                        candidates.append((cov, u, v, w))

    n_cand = len(candidates)
    cov_matrix = np.array([c[0] for c in candidates], dtype=float)
    target_vec = np.ones(n_targets)

    def greedy_cover(cov_matrix, target_vec, max_iter=20000):
        remaining = target_vec.copy()
        selected = []
        used = np.zeros(cov_matrix.shape[0], dtype=bool)
        for _ in range(max_iter):
            if np.all(remaining == 0):
                break
            score = cov_matrix @ remaining
            avail = np.where((~used) & (score > 0))[0]
            if len(avail) == 0:
                break
            best = avail[np.argmax(score[avail])]
            new_remaining = remaining - cov_matrix[best]
            if np.any(new_remaining < 0):
                used[best] = True
                continue
            selected.append(best)
            used[best] = True
            remaining = new_remaining
        if np.all(remaining == 0):
            return selected
        return None

    best_selected = None
    for seed in range(200):
        if time.time() > deadline - 15:
            break
        rng = np.random.RandomState(seed)
        order = rng.permutation(n_cand)
        cov_shuffled = cov_matrix[order]
        sel = greedy_cover(cov_shuffled, target_vec)
        if sel is not None:
            orig_sel = list(dict.fromkeys(order[s] for s in sel))
            if best_selected is None or len(orig_sel) < len(best_selected):
                best_selected = orig_sel

    if best_selected is not None:
        rank = len(best_selected)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t, idx in enumerate(best_selected):
            cov, u, v, w = candidates[idx]
            U[:, t] = u
            V[:, t] = v
            W[:, t] = w
        if verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
