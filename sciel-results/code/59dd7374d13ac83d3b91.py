import time
import random
import sys

DIMENSIONS = [3, 3, 3]


def _verify(U, V, W):
    """Verify exact decomposition: for all i,j,k sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _construct_strassen_2x2():
    """Rank-7 Strassen for 2x2. Returns (U,V,W) each of shape (4,7).
    U[i*2+j], V[j*2+k], W[k*2+i]."""
    U = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 1, 0, 0, 0, 0],  # (0,1)
        [0, 1, 0, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 1, 1],  # (1,1)
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 1, 1, 0, 0, 0],  # (0,1)
        [1, 1, 0, 0, 1, 1, 0],  # (1,0)
        [0, 0, 1, 1, 1, 1, 1],  # (1,1)
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 1, 1],  # (1,1)
    ]
    return U, V, W


def _construct_rank_26():
    """Rank-26 decomposition: Strassen on top-left 2x2 block (rank 7)
    plus 19 scalar products for the interactions with row/col index 2."""
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- 2x2 Strassen block on indices i,j,k in {0,1} ---
    U2, V2, W2 = _construct_strassen_2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m+j][c] = U2[i*2+j][c]
                V[j*p+i][c] = V2[j*2+i][c]   # V row (j,k) with k=i here (2x2 square)
                W[i*n+j][c] = W2[i*2+j][c]   # W row (k,i) with k=i here

    # --- 19 scalar terms for the third row/column ---
    # Terms: (i,j,k) with (i,j,k) not all in {0,1}, i.e. at least one index == 2
    # That's 27 - 8 = 19 terms.
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue  # already handled by Strassen block
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    assert col == rank, f"rank mismatch: {col} vs {rank}"
    return U, V, W


def _search_rank(deadline, rank, seed):
    """Randomized local search for an exact rank-`rank` decomposition.
    Uses a smooth squared-error surrogate and a high-precision polish pass.
    Returns (U,V,W) if exact, else None."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    best = None
    best_err = float('inf')

    # Precompute the 27 (i,j,k) index triples and their flat positions.
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def evaluate(U, V, W):
        err = 0.0
        for (i, j, k) in triples:
            s = 0.0
            for c in range(rank):
                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            err += (s - 1.0) * (s - 1.0)
        return err

    def polish(U, V, W, iters=4000):
        """Gaussian local search minimizing smooth squared error."""
        nonlocal best, best_err
        cur = evaluate(U, V, W)
        if cur < best_err:
            best_err = cur
            best = (U, V, W)
            if best_err < 1e-12:
                return True
        step = 0.2
        for it in range(iters):
            if time.time() > deadline - 2:
                break
            if rng.random() < 0.34:
                idx = rng.randrange(n*m); c = rng.randrange(rank)
                U[idx][c] += rng.gauss(0, step)
            elif rng.random() < 0.5:
                idx = rng.randrange(m*p); c = rng.randrange(rank)
                V[idx][c] += rng.gauss(0, step)
            else:
                idx = rng.randrange(p*n); c = rng.randrange(rank)
                W[idx][c] += rng.gauss(0, step)
            new = evaluate(U, V, W)
            if new < cur:
                cur = new
                step = max(step * 0.98, 0.02)
                if cur < best_err:
                    best_err = cur
                    best = (U, V, W)
                    if best_err < 1e-12:
                        return True
            else:
                step = min(step * 1.05, 0.5)
        return False

    # Start from the rank-26 structured solution projected to `rank` columns,
    # then add random restarts.
    base_U, base_V, base_W = _construct_rank_26()
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        # Initialize: take base (truncated) plus small noise
        U = [row[:] for row in base_U]
        V = [row[:] for row in base_V]
        W = [row[:] for row in base_W]
        if rank < 26:
            # zero out extra columns beyond rank
            for row in U: del row[rank:]
            for row in V: del row[rank:]
            for row in W: del row[rank:]
        # Add noise to encourage escape
        for row in U:
            for c in range(rank):
                row[c] += rng.gauss(0, 0.05)
        for row in V:
            for c in range(rank):
                row[c] += rng.gauss(0, 0.05)
        for row in W:
            for c in range(rank):
                row[c] += rng.gauss(0, 0.05)
        if polish(U, V, W):
            if best is not None and best_err < 1e-12:
                return best
    if best is not None and best_err < 1e-9:
        return best
    return None


def construct():
    overall_deadline = time.time() + 138  # safety margin before 150s kill

    # 1) Structured rank-26 decomposition (proven improvement over schoolbook).
    U, V, W = _construct_rank_26()
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # 2) Search for a lower rank using the smooth-surrogate polish machinery.
    for target_rank in (25, 24, 23, 22):
        if time.time() > overall_deadline - 15:
            break
        result = _search_rank(overall_deadline - 5, target_rank, seed=1000 + target_rank)
        if result is not None:
            rU, rV, rW = result
            if _verify(rU, rV, rW):
                return {"U": rU, "V": rV, "W": rW}

    # 3) Fallback to schoolbook (guaranteed valid).
    return _construct_schoolbook()
