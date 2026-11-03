"""Rewritten program: fast ALS + descending rank search for M_{2,3,7} tensor."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s hard limit
    random.seed(12345)
    np.random.seed(12345)

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14

    # Build target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def err_of(U, V, W):
        Trec = np.einsum('ac,bc,dc->abd', U, V, W)
        return float(np.max(np.abs(Trec - T)))

    # ---- Schoolbook baseline (rank 42) ----
    rank_school = n * m * p
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()

    def als(U, V, W, max_iter=200, tol=1e-11):
        """Alternating least squares; returns (U, V, W, err)."""
        rank = U.shape[1]
        err = 1e9
        for it in range(max_iter):
            if time.time() > deadline - 8.0:
                break
            # Update U
            VW = np.einsum('jc,kc->jkc', V, W).reshape(mp * pn, rank)
            for i in range(nm):
                U[i] = np.linalg.lstsq(VW, T[i].reshape(-1), rcond=None)[0]
            # Update V
            UW = np.einsum('ic,kc->ikc', U, W).reshape(nm * pn, rank)
            for j in range(mp):
                V[j] = np.linalg.lstsq(UW, T[:, j, :].reshape(-1), rcond=None)[0]
            # Update W
            UV = np.einsum('ic,jc->ijc', U, V).reshape(nm * mp, rank)
            for k in range(pn):
                W[k] = np.linalg.lstsq(UV, T[:, :, k].reshape(-1), rcond=None)[0]
            err = err_of(U, V, W)
            if err < tol:
                break
        return U, V, W, err

    def random_init(rank, scale=0.7):
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        return U, V, W

    def structured_init(rank):
        """Seed from schoolbook columns, then random perturbation / subsampling."""
        if rank >= rank_school:
            return U_s.copy(), V_s.copy(), W_s.copy()
        # take a random subset of schoolbook columns plus small noise
        idx = random.sample(range(rank_school), min(rank, rank_school))
        U = U_s[:, idx].copy()
        V = V_s[:, idx].copy()
        W = W_s[:, idx].copy()
        # add noise to escape exact schoolbook subspace
        U += np.random.randn(nm, rank) * 0.3
        V += np.random.randn(mp, rank) * 0.3
        W += np.random.randn(pn, rank) * 0.3
        return U, V, W

    def attempts_for_rank(r):
        if r >= 35:
            return 4
        if r >= 30:
            return 8
        if r >= 25:
            return 16
        return 30

    # ---- Descending rank search ----
    found_at = {}
    for r in range(rank_school - 1, 20, -1):
        if time.time() > deadline - 12.0:
            break
        N = attempts_for_rank(r)
        found = False
        for a in range(N):
            if time.time() > deadline - 8.0:
                break
            # alternate random and structured inits
            if a % 2 == 0:
                U, V, W = random_init(r)
            else:
                U, V, W = structured_init(r)
            Uo, Vo, Wo, e = als(U, V, W, max_iter=180)
            if e < 1e-8:
                best_rank = r
                best_U, best_V, best_W = Uo.copy(), Vo.copy(), Wo.copy()
                found = True
                break
        if found:
            continue
        else:
            # could not fit at this rank -> likely below achievable; stop descending
            break

    # ---- Refinement: try to shave best_rank by 1 repeatedly ----
    def try_reduce(cur_rank, deadline_local):
        """Attempt to produce a valid decomposition of rank cur_rank-1."""
        target = cur_rank - 1
        if target < 1:
            return None
        # (a) drop-column from current best
        for _ in range(6):
            if time.time() > deadline_local:
                break
            c = random.randrange(cur_rank)
            keep = [x for x in range(cur_rank) if x != c]
            U2 = best_U[:, keep].copy()
            V2 = best_V[:, keep].copy()
            W2 = best_W[:, keep].copy()
            Uo, Vo, Wo, e = als(U2, V2, W2, max_iter=200)
            if e < 1e-8:
                return (Uo, Vo, Wo, target)
        # (b) fresh random / structured at target
        for a in range(20):
            if time.time() > deadline_local:
                break
            if a % 2 == 0:
                U, V, W = random_init(target)
            else:
                U, V, W = structured_init(target)
            Uo, Vo, Wo, e = als(U, V, W, max_iter=220)
            if e < 1e-8:
                return (Uo, Vo, Wo, target)
        return None

    # polish current best once more
    if time.time() < deadline - 15.0:
        Uo, Vo, Wo, e = als(best_U.copy(), best_V.copy(), best_W.copy(), max_iter=250)
        if e < 1e-8:
            best_U, best_V, best_W = Uo, Vo, Wo

    while time.time() < deadline - 10.0:
        res = try_reduce(best_rank, deadline - 8.0)
        if res is None:
            break
        Uo, Vo, Wo, nr = res
        best_rank = nr
        best_U, best_V, best_W = Uo, Vo, Wo

    # ---- Cleanup to nice fractions ----
    def clean(x):
        if abs(x) < 1e-9:
            return 0
        for d in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
            v = round(x * d) / d
            if abs(x - v) < 1e-8:
                if d == 1:
                    return int(v)
                return v
        return float(x)

    def to_list(mat):
        return [[clean(x) for x in row] for row in mat]

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)

    # final sanity: rebuild from cleaned values and verify
    Uc = np.array(U_list, dtype=np.float64)
    Vc = np.array(V_list, dtype=np.float64)
    Wc = np.array(W_list, dtype=np.float64)
    if err_of(Uc, Vc, Wc) > 1e-7:
        # fall back to raw (uncleaned) best
        U_list = to_list(best_U)
        V_list = to_list(best_V)
        W_list = to_list(best_W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
