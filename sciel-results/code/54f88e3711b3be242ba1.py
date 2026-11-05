"""Search-based rank-minimal decomposition for 4x5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# W index convention: W[k*n + i, r] ; U[i*m + j, r] ; V[j*p + k, r]
# Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1


def _verify(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ur = U[i * m + j]
                vr = V[j * p + k]
                wr = W[k * n + i]
                for r in range(rank):
                    s += ur[r] * vr[r] * wr[r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _make_term(i, j, k):
    # rank-1 term that is 1 at (i,j,k) and 0 elsewhere (unit tensor)
    n, m, p = DIMENSIONS
    U = [0.0]
    V = [0.0]
    W = [0.0]
    U[i * m + j] = 1.0
    V[j * p + k] = 1.0
    W[k * n + i] = 1.0
    return U, V, W


def _build_candidates(n, m, p, rng, limit=4000):
    # Generate candidate rank-1 tensors from index "circuits".
    # Each candidate is a triple of index sets (I, J, K) with a sign pattern.
    cands = []
    seen = set()

    def add(i, j, k):
        key = (i, j, k)
        if key in seen:
            return
        seen.add(key)
        U, V, W = _make_term(i, j, k)
        cands.append((U, V, W))

    # all unit terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add(i, j, k)

    # sum over one index (e.g. sum over k) -> term = sum_k unit(i,j,k)
    # This is a rank-1 term with U[i*m+j]=1, V[j*p+k]=1 for all k, W[k*n+i]=1 for all k.
    for i in range(n):
        for j in range(m):
            U = [0.0]
            V = [0.0]
            W = [0.0]
            U[i * m + j] = 1.0
            for k in range(p):
                V[j * p + k] = 1.0
                W[k * n + i] = 1.0
            cands.append((U, V, W))
            seen.add(("sumk", i, j))

    for i in range(n):
        for k in range(p):
            U = [0.0]
            V = [0.0]
            W = [0.0]
            W[k * n + i] = 1.0
            for j in range(m):
                U[i * m + j] = 1.0
                V[j * p + k] = 1.0
            cands.append((U, V, W))
            seen.add(("sumj", i, k))

    for j in range(m):
        for k in range(p):
            U = [0.0]
            V = [0.0]
            W = [0.0]
            V[j * p + k] = 1.0
            for i in range(n):
                U[i * m + j] = 1.0
                W[k * n + i] = 1.0
            cands.append((U, V, W))
            seen.add(("sumi", j, k))

    # random sparse products: U on a subset of (i,j), V on subset of (j,k), W on subset of (k,i)
    for _ in range(600):
        # pick a small set of i, j, k
        si = sorted(rng.sample(range(n), rng.randint(1, n)))
        sj = sorted(rng.sample(range(m), rng.randint(1, m)))
        sk = sorted(rng.sample(range(p), rng.randint(1, p)))
        U = [0.0]
        V = [0.0]
        W = [0.0]
        for i in si:
            for j in sj:
                U[i * m + j] = rng.choice([1.0, -1.0])
        for j in sj:
            for k in sk:
                V[j * p + k] = rng.choice([1.0, -1.0])
        for k in sk:
            for i in si:
                W[k * n + i] = rng.choice([1.0, -1.0])
        cands.append((U, V, W))

    # random dense-ish with small support
    for _ in range(400):
        U = [0.0]
        V = [0.0]
        W = [0.0]
        for idx in range(n * m):
            if rng.random() < 0.3:
                U[idx] = rng.choice([1.0, -1.0])
        for idx in range(m * p):
            if rng.random() < 0.3:
                V[idx] = rng.choice([1.0, -1.0])
        for idx in range(p * n):
            if rng.random() < 0.3:
                W[idx] = rng.choice([1.0, -1.0])
        cands.append((U, V, W))

    rng.shuffle(cands)
    return cands[:limit]


def _fit_candidates(cands, n, m, p, max_terms):
    # Solve A x = b where A[:, r] = tensor_r (flattened), b = all ones.
    # We want a sparse solution: pick up to max_terms columns.
    N = n * m * p
    A = np.zeros((N, len(cands)))
    for r, (U, V, W) in enumerate(cands):
        A[:, r] = np.array([U[i * m + j] * V[j * p + k] * W[k * n + i]
                            for i in range(n) for j in range(m) for k in range(p)])
    b = np.ones(N)

    # Greedy forward selection with least-squares solve.
    chosen = []
    residual = b.copy()
    for step in range(max_terms):
        best_r = -1
        best_err = float('inf')
        # try adding each candidate, solve least squares on chosen + candidate
        for r in range(len(cands)):
            if r in chosen:
                continue
            cols = chosen + [r]
            sub = A[:, cols]
            try:
                sol, *_ = np.linalg.lstsq(sub, b, rcond=None)
            except Exception:
                continue
            # check exactness
            pred = sub @ sol
            err = np.max(np.abs(pred - b))
            if err < best_err:
                best_err = err
                best_r = r
                best_sol = sol
                best_cols = cols
        if best_r < 0:
            break
        chosen = best_cols
        sol = best_sol
        if best_err < 1e-9:
            break

    if not chosen:
        return None
    # Verify exactness
    sub = A[:, chosen]
    sol, *_ = np.linalg.lstsq(sub, b, rcond=None)
    pred = sub @ sol
    if np.max(np.abs(pred - b)) > 1e-6:
        return None
    # Build U,V,W from chosen candidates and solution
    rank = len(chosen)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, r in enumerate(chosen):
        U[:, c] = cands[r][0] * sol[c]
        V[:, c] = cands[r][1]
        W[:, c] = cands[r][2]
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _try_block_strassen(n, m, p):
    # Use Strassen on 2x2 blocks where possible to reduce rank.
    # For 4x5x5: split into 2x2 blocks of 2x3 and 3x3 etc. This is complex.
    # Instead, try a simple block approach: treat as (2x2) x (5x5) with Strassen on the 2x2 part.
    # This may not directly help; skip for now and rely on search.
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120  # safety margin

    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    rng = random.Random(12345)

    # Try search with increasing max_terms
    for max_terms in [80, 90, 100, 110, 120]:
        if time.time() > deadline - 30:
            break
        cands = _build_candidates(n, m, p, rng, limit=3000)
        result = _fit_candidates(cands, n, m, p, max_terms)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
            if rank < best_rank and _verify(U, V, W, n, m, p):
                best_rank = rank
                best = (U, V, W)
        if time.time() > deadline - 15:
            break

    U, V, W = best
    # Convert to lists of lists with integer/half-integer rounding
    def to_list(A):
        out = []
        for row in A:
            r = []
            for v in row:
                # round to nearest half-integer
                h = round(v * 2) / 2.0
                # if very close to integer, make it integer
                if abs(h - round(h)) < 1e-9:
                    r.append(int(round(h)))
                else:
                    r.append(h)
            out.append(r)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
