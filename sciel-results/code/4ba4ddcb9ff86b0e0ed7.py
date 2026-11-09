# EVOLVE-BLOCK-START
import time
import numpy as np

N = 10
DEMANDS = [
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
]
D = len(DEMANDS)


def _edge_masks():
    """A[d, e] = 1 if demand d uses edge e when routed clockwise."""
    A = np.zeros((D, N), dtype=np.int8)
    for d, (s, t, w) in enumerate(DEMANDS):
        if s == t:
            continue
        e = s
        while e != t:
            A[d, e] = 1
            e = (e + 1) % N
    return A


def _ccw_masks(A):
    """B[d, e] = 1 if demand d uses edge e when routed counter-clockwise."""
    return (1 - A).astype(np.int8)


def _loads(x, A, B):
    """Edge loads given a 0/1 routing vector x (0=clockwise, 1=ccw)."""
    cw = (x < 0.5).astype(np.int8)
    ccw = 1 - cw
    return np.array(
        [np.dot(cw * A[:, e], np.array([w for (_, _, w) in DEMANDS])) +
         np.dot(ccw * B[:, e], np.array([w for (_, _, w) in DEMANDS]))
         for e in range(N)],
        dtype=np.int64,
    )


def _max_load(x, A, B):
    return int(_loads(x, A, B).max())


def _greedy_init(A, B):
    """Greedy: for each demand pick the direction that minimizes the current max edge load."""
    x = np.zeros(D, dtype=np.int8)
    w = np.array([w for (_, _, w) in DEMANDS], dtype=np.int64)
    for d in range(D):
        best_dir = 0
        best_ml = None
        for dir_ in (0, 1):
            x[d] = dir_
            ml = int(_loads(x, A, B).max())
            if best_ml is None or ml < best_ml:
                best_ml = ml
                best_dir = dir_
        x[d] = best_dir
    return x


def _local_search(x, A, B, deadline):
    """First-improvement single-flip local search, time-bounded."""
    x = x.astype(np.int8).copy()
    w = np.array([w for (_, _, w) in DEMANDS], dtype=np.int64)
    best = int(_loads(x, A, B).max())
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for d in range(D):
            if time.time() > deadline:
                break
            old = x[d]
            new = 1 - old
            x[d] = new
            ml = int(_loads(x, A, B).max())
            if ml < best:
                best = ml
                improved = True
            else:
                x[d] = old
    return x, best


def _try_ilp(A, B, deadline):
    """Exact ILP via scipy HiGHS if available. Returns (x, best) or (None, None)."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        return None, None

    w = np.array([w for (_, _, w) in DEMANDS], dtype=np.float64)
    # Variables: x[0..14] in {0,1}, x[15] = max load (continuous >= 0)
    nvars = D + 1
    c = np.zeros(nvars)
    c[-1] = 1.0  # minimize max load

    # Constraints: for each edge e, sum_d (x_d * A[d,e] * w_d + (1-x_d) * B[d,e] * w_d) <= x_max
    # = sum_d (B[d,e]*w_d) + sum_d ((A[d,e]-B[d,e])*w_d) * x_d <= x_max
    # => sum_d ((A[d,e]-B[d,e])*w_d) * x_d - x_max <= -sum_d (B[d,e]*w_d)
    A_mat = np.zeros((N, nvars))
    b_ub = np.zeros(N)
    for e in range(N):
        for d in range(D):
            A_mat[e, d] = (A[d, e] - B[d, e]) * w[d]
        A_mat[e, -1] = -1.0
        b_ub[e] = -np.dot(B[:, e], w)

    constraints = LinearConstraint(A_mat, -np.inf, b_ub)
    integrality = np.concatenate([np.ones(D, dtype=int), np.zeros(1, dtype=int)])
    bounds = Bounds(np.zeros(nvars), np.concatenate([np.ones(D), [np.inf]]))

    t0 = time.time()
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        return None, None

    if res.status == 0 and res.x is not None:
        x = (res.x[:D] >= 0.5).astype(np.int8)
        return x, int(_loads(x, A, B).max())
    return None, None


def ring_routing() -> np.ndarray:
    A = _edge_masks()
    B = _ccw_masks(A)

    deadline = time.time() + 590.0  # safety margin before the 600s limit

    # Exact ILP attempt (fast for this size); fall back to heuristic if unavailable/slow.
    x_ilp, _ = _try_ilp(A, B, deadline)

    # Greedy + local search (also used as fallback / cross-check).
    x_g = _greedy_init(A, B)
    x_ls, _ = _local_search(x_g, A, B, deadline)

    candidates = [c for c in (x_ilp, x_ls, x_g) if c is not None]
    best = None
    best_ml = None
    for x in candidates:
        ml = _max_load(x, A, B)
        if best_ml is None or ml < best_ml:
            best_ml = ml
            best = x

    if best is None:
        best = np.zeros(D)
    return best.astype(np.float64)


# EVOLVE-BLOCK-END
