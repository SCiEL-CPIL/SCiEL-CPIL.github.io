"""Matrix multiplication 2x3x7 rank-minimal decomposition.

Combines:
  - Current program: randomized integer search over U,V,W with exact verification.
  - Inspiration: tensor-based error via einsum + structured partial-schoolbook
    initialization and greedy column-merging from schoolbook.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]


def construct():
    deadline = time.time() + 115.0  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # A is n x m (2x3), B is m x p (3x7), C is n x p (2x7)
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14

    # Target tensor: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def reconstruct(U, V, W):
        return np.einsum('ac,bc,dc->abd', U, V, W)

    def is_exact(U, V, W, tol=1e-6):
        R = reconstruct(U, V, W)
        return np.max(np.abs(R - target)) < tol

    def to_lists(U, V, W):
        def conv(row):
            out = []
            for x in row:
                hx = round(x * 2) / 2.0
                out.append(int(hx) if abs(hx - round(hx)) < 1e-9 else hx)
            return out
        return {"U": [conv(r) for r in U],
                "V": [conv(r) for r in V],
                "W": [conv(r) for r in W]}

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best = to_lists(U_sb, V_sb, W_sb)
    best_rank = n * m * p

    def try_accept(U, V, W, r):
        nonlocal best, best_rank
        if is_exact(U, V, W):
            best = to_lists(U, V, W)
            best_rank = r
            return True
        return False

    rng = np.random.default_rng(0)

    # Precompute terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i * m + j, j * p + k, k * n + i))
    nterms = len(terms)

    # ---- Phase 1: Partial-schoolbook + perturbation search ----
    for keep_frac in [1.0, 0.9, 0.8, 0.7, 0.6]:
        if time.time() > deadline - 20:
            break
        K = max(1, int(nterms * keep_frac))
        for attempt in range(300):
            if time.time() > deadline - 10:
                break
            idx = rng.choice(nterms, size=K, replace=False)
            U = np.zeros((nm, K))
            V = np.zeros((mp, K))
            W = np.zeros((pn, K))
            for c, t in enumerate(idx):
                a, b, cc = terms[t]
                U[a, c] = 1
                V[b, c] = 1
                W[cc, c] = 1

            n_perturb = rng.integers(1, min(4, K) + 1)
            perm = rng.permutation(K)
            for c in perm[:n_perturb]:
                nz_u = rng.choice(nm, size=rng.integers(1, 3), replace=False)
                nz_v = rng.choice(mp, size=rng.integers(1, 3), replace=False)
                nz_w = rng.choice(pn, size=rng.integers(1, 3), replace=False)
                U[nz_u, c] += rng.integers(-1, 2, size=len(nz_u))
                V[nz_v, c] += rng.integers(-1, 2, size=len(nz_v))
                W[nz_w, c] += rng.integers(-1, 2, size=len(nz_w))

            if try_accept(U, V, W, K):
                return best

    # ---- Phase 2: Pure random search at low ranks ----
    target_rank = 2 * p + 2 * n - 2  # 18
    for r in range(target_rank, best_rank):
        if time.time() > deadline - 5:
            break
        for attempt in range(500):
            if time.time() > deadline - 3:
                break
            U = rng.integers(-2, 3, size=(nm, r))
            V = rng.integers(-2, 3, size=(mp, r))
            W = rng.integers(-2, 3, size=(pn, r))
            if try_accept(U, V, W, r):
                return best

    # ---- Phase 3: Greedy column-merging from schoolbook ----
    if time.time() < deadline - 15:
        U, V, W = schoolbook()
        r = n * m * p
        active = list(range(r))
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for _ in range(300):
                if len(active) < 2:
                    break
                i, j = rng.choice(len(active), size=2, replace=False)
                ci, cj = active[i], active[j]
                U_c1 = U[ci].copy()
                V_c1 = V[ci].copy()
                W_c1 = W[ci].copy()
                # Try merge: ci absorbs cj
                U[ci] = U_c1 + U[cj]
                V[ci] = V_c1 + V[cj]
                W[ci] = W_c1 + W[cj]
                if is_exact(U, V, W):
                    active.remove(cj)
                    r -= 1
                    improved = True
                else:
                    U[ci] = U_c1
                    V[ci] = V_c1
                    W[ci] = W_c1
        if r < best_rank:
            U_final = U[:, active]
            V_final = V[:, active]
            W_final = W[:, active]
            if try_accept(U_final, V_final, W_final, r):
                return best

    return best
