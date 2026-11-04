"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
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
    return U, V, W


def _search_rank(r, n, m, p, time_budget, seed):
    """Local search for an exact rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(seed)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Sparse random init
    for row in range(n * m):
        for col in range(r):
            if rng.random() < 0.35:
                U[row][col] = rng.choice([1, -1])
    for row in range(m * p):
        for col in range(r):
            if rng.random() < 0.35:
                V[row][col] = rng.choice([1, -1])
    for row in range(p * n):
        for col in range(r):
            if rng.random() < 0.35:
                W[row][col] = rng.choice([1, -1])

    def error():
        e = 0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = 0
                    for c in range(r):
                        s += u_row[c] * v_row[c] * w_row[c]
                    e += (s - 1) * (s - 1)
        return e

    cur = error()
    best = cur
    best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
    no_improve = 0
    steps = 0
    max_steps = 6000

    factors = [(U, n * m), (V, m * p), (W, p * n)]
    while no_improve < 250 and steps < max_steps:
        if time.time() - start > time_budget:
            break
        steps += 1
        M, nrows = factors[rng.randrange(3)]
        row = rng.randrange(nrows)
        col = rng.randrange(r)
        old = M[row][col]
        new = rng.choice([0, 1, -1])
        if new == old:
            continue
        M[row][col] = new
        ne = error()
        if ne < cur:
            cur = ne
            no_improve = 0
            if ne < best:
                best = ne
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            if rng.random() > 0.08:
                M[row][col] = old
            no_improve += 1

    Ub, Vb, Wb = best_state
    if _verify(Ub, Vb, Wb, n, m, p):
        return Ub, Vb, Wb
    return None


def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    # Guaranteed valid baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # Try progressively lower ranks; stop once time runs low.
    for target_rank in range(22, 17, -1):
        if time.time() - start_time > timeout - 6:
            break
        for seed in range(4):
            if time.time() - start_time > timeout - 6:
                break
            res = _search_rank(target_rank, n, m, p, time_budget=9.0, seed=seed)
            if res is not None:
                U, V, W = res
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # found valid at this rank, try next (lower) rank
        # If we found a valid decomposition, we can stop searching lower
        # (lower is better, but we want to keep the best found; continue to try lower)

    # Final safety: ensure we return something valid
    if time.time() - start_time > timeout - 1:
        return best

    return best
# EVOLVE-BLOCK-END
