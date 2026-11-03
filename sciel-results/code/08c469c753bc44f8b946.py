import numpy as np
import time

N = 11
S3 = np.sqrt(3.0)

# 6 edge normals of a unit hexagon at rotation 0
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

UNIT_VERT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def rot2(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def hex_vertices(center, ang_deg):
    R = rot2(ang_deg)
    return center + (UNIT_VERT @ R.T).T


def sep_axes(i_ang, j_ang):
    """All candidate separating axes for two hexagons at given rotations."""
    a1 = NORM @ rot2(i_ang).T
    a2 = NORM @ rot2(j_ang).T
    return np.vstack([a1, a2])


def min_separation(Ci, Ai, Cj, Aj):
    """Max over axes of |d · axis| — the separating distance."""
    d = Ci - Cj
    axes = sep_axes(Ai, Aj)
    return float(np.abs(d @ axes.T).max())


def outer_side_for(C, A, outer_rot):
    """Smallest outer side (centered at origin, rotation outer_rot) enclosing all."""
    R = rot2(outer_rot)
    n_outer = NORM @ R.T
    m = 0.0
    for i in range(N):
        verts = hex_vertices(C[i], A[i])
        proj = verts @ n_outer.T  # (6,6)
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def outer_side(C, A):
    """Minimize over outer rotation in 0..60 (symmetry)."""
    # sample outer rotations
    best = np.inf
    for orot in np.linspace(0, 60, 13):
        s = outer_side_for(C, A, orot)
        if s < best:
            best = s
    # refine around best
    orots = np.linspace(0, 60, 13)
    idx = int(np.argmin([outer_side_for(C, A, o) for o in orots]))
    lo = orots[max(0, idx - 1)]
    hi = orots[min(len(orots) - 1, idx + 1)]
    for _ in range(4):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if outer_side_for(C, A, m1) < outer_side_for(C, A, m2):
            hi = m2
        else:
            lo = m1
    return min(outer_side_for(C, A, (lo + hi) / 2), best)


def nonoverlap(C, A, eps=1e-9):
    for i in range(N):
        for j in range(i + 1, N):
            if min_separation(C[i], A[i], C[j], A[j]) < 2.0 - eps:
                return False
    return True


def inside_outer(C, A, S, outer_rot, eps=1e-9):
    R = rot2(outer_rot)
    n_outer = NORM @ R.T
    apothem = S * S3 / 2.0
    for i in range(N):
        verts = hex_vertices(C[i], A[i])
        proj = verts @ n_outer.T
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, A, S, outer_rot):
    return nonoverlap(C, A) and inside_outer(C, A, S, outer_rot)


def separation_violation(C, A):
    v = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            s = min_separation(C[i], A[i], C[j], A[j])
            if s < 2.0:
                v += (2.0 - s) ** 2
    return v


def hex_close_pack_seed():
    """Dense hexagonal packing of 11 unit hexagons (center + ring + partial)."""
    C = np.array([[0.0, 0.0]])
    # inner ring of 6 at distance S3
    for k in range(6):
        a = k * np.pi / 3
        C = np.vstack([C, [S3 * np.cos(a), S3 * np.sin(a)]])
    # outer partial: 4 more
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C = np.vstack([C, [2 * S3 * np.cos(a), 2 * S3 * np.sin(a)]])
    C = C[:N]
    A = np.zeros(N)
    return C, A


def hex_dense_seed2():
    """Another dense arrangement: 3-4-3-1 rows offset."""
    C = []
    rows = [(3, 0.0), (4, 1.5), (3, 3.0), (1, 4.5)]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def hex_ring_seed():
    C = np.array([[0.0, 0.0]])
    for k in range(10):
        a = k * 2 * np.pi / 10
        C = np.vstack([C, [2.1 * S3 * np.cos(a), 2.1 * S3 * np.sin(a)]])
    C = C[:N]
    A = np.zeros(N)
    return C, A


def push_apart(C, A, rng):
    """Find most-overlapping pair and push them apart."""
    worst_i, worst_j = -1, -1
    worst_def = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            s = min_separation(C[i], A[i], C[j], A[j])
            def_ = 2.0 - s
            if def_ > worst_def:
                worst_def = def_
                worst_i, worst_j = i, j
    if worst_i < 0:
        return C, A
    d = C[worst_i] - C[worst_j]
    n = np.linalg.norm(d)
    if n < 1e-9:
        d = np.array([1.0, 0.0])
    else:
        d = d / n
    push = min(0.3, worst_def * 0.6 + 0.05)
    C = C.copy()
    C[worst_i] += d * push
    C[worst_j] -= d * push
    return C, A


def local_push_shrink(C, A, deadline, rng, max_iter=20000):
    """Local search: repeatedly push overlaps apart, then shrink via small random moves,
    accepting only if outer side doesn't grow and non-overlap maintained."""
    C = C.copy()
    A = A.copy()
    S = outer_side(C, A)
    best_S = S if nonoverlap(C, A) else np.inf
    best_C, best_A = C.copy(), A.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Phase 1: fix overlaps
        C, A = push_apart(C, A, rng)
        # Phase 2: random small move on a random hexagon
        idx = rng.randint(0, N - 1)
        new_C = C.copy()
        new_A = A.copy()
        if rng.random() < 0.7:
            new_C[idx] += rng.normal(0, 0.15, 2)
        else:
            new_A[idx] += rng.normal(0, 4.0)
        if not nonoverlap(new_C, new_A):
            continue
        new_S = outer_side(new_C, new_A)
        if new_S < S + 1e-9:
            C, A, S = new_C, new_A, new_S
            if S < best_S:
                best_S = S
                best_C, best_A = C.copy(), A.copy()
    return best_C, best_A, best_S


def anneal(C, A, deadline, seed):
    """Simulated annealing minimizing outer side with non-overlap constraint."""
    rng = np.random.RandomState(seed)
    C = C.copy()
    A = A.copy()
    S = outer_side(C, A)
    valid_now = nonoverlap(C, A)
    cost = S if valid_now else S + 50.0 * separation_violation(C, A)
    best_S = S if valid_now else np.inf
    best_C, best_A = C.copy(), A.copy()

    T = 0.4
    T_min = 1e-4
    cool = 0.9997
    while time.time() < deadline and T > T_min:
        r = rng.random()
        new_C, new_A = C.copy(), A.copy()
        idx = rng.randint(0, N - 1)
        if r < 0.55:
            new_C[idx] += rng.normal(0, 0.2, 2)
        elif r < 0.8:
            new_A[idx] += rng.normal(0, 5.0)
        else:
            new_C, new_A = push_apart(new_C, new_A, rng)

        new_S = outer_side(new_C, new_A)
        nv = nonoverlap(new_C, new_A)
        new_cost = new_S if nv else new_S + 50.0 * separation_violation(new_C, new_A)
        dc = new_cost - cost
        if dc < 0 or rng.random() < np.exp(-dc / T):
            C, A, S = new_C, new_A, new_S
            cost = new_cost
            if nv and S < best_S:
                best_S = S
                best_C, best_A = C.copy(), A.copy()
        T *= cool
    return best_C, best_A, best_S


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s

    inits = [hex_close_pack_seed(), hex_dense_seed2(), hex_ring_seed()]
    # add a few random perturbations of the best init
    rng0 = np.random.RandomState(42)
    base_C, base_A = inits[0]
    for s in range(3):
        C = base_C + rng0.normal(0, 0.1, (N, 2))
        A = base_A + rng0.normal(0, 3.0, N)
        inits.append((C, A))

    best_C, best_A, best_S = None, None, np.inf
    for C, A in inits:
        if nonoverlap(C, A):
            S = outer_side(C, A)
            if S < best_S:
                best_S, best_C, best_A = S, C.copy(), A.copy()

    # Allocate remaining time to multiple anneal + local-push rounds.
    # Each round: short anneal then local_push_shrink polish.
    remaining = deadline - time.time()
    n_rounds = 6
    per_round = remaining / n_rounds
    seeds = [101, 202, 303, 404, 505, 606]
    for r in range(n_rounds):
        if time.time() > deadline - 2:
            break
        # pick a starting point: alternate between best and a fresh init
        if r % 2 == 0 and best_C is not None:
            C0, A0 = best_C + np.random.RandomState(seeds[r]).normal(0, 0.05, (N, 2)), best_A.copy()
        else:
            C0, A0 = inits[r % len(inits)]
        # anneal for half the round
        t_half = time.time() + per_round * 0.5
        Ca, Aa, Sa = anneal(C0, A0, t_half, seed=seeds[r])
        if Sa < best_S:
            best_S, best_C, best_A = Sa, Ca.copy(), Aa.copy()
        # local push-shrink for the rest
        t_end = time.time() + per_round * 0.5
        Cb, Ab, Sb = local_push_shrink(Ca, Aa, t_end, np.random.RandomState(seeds[r] + 7))
        if Sb < best_S:
            best_S, best_C, best_A = Sb, Cb.copy(), Ab.copy()

    # Final polish with more iterations if time allows
    if best_C is not None and time.time() < deadline - 5:
        Cb, Ab, Sb = local_push_shrink(best_C, best_A, deadline - 2,
                                       np.random.RandomState(999), max_iter=30000)
        if Sb < best_S:
            best_S, best_C, best_A = Sb, Cb.copy(), Ab.copy()

    if best_C is None:
        # fallback: spread them out far apart (valid but large)
        best_C = np.array([[6.0 * i, 0.0] for i in range(N)])
        best_A = np.zeros(N)
        best_S = outer_side(best_C, best_A)

    inner = [[float(best_C[i, 0]), float(best_C[i, 1]), float(best_A[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), 0.0
