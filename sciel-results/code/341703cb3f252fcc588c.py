"""Pack 200 equal circles in a unit square via hex-grid greedy + bottleneck elimination + SA."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
HALF = 0.5


def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))


def _hex_points(s, ox, oy):
    """Hexagonal lattice points with spacing s, offset (ox, oy)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 3
    rows = int(np.ceil(1.0 / dy)) + 3
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x, y + yoff))
    arr = np.array(pts)
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]


def _greedy_select(cands, k, r_min):
    """Greedy farthest-point selection of k points maximizing min distance."""
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    selected = []
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    for _ in range(k - 1):
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        j = int(np.argmax(min_dist))
        if min_dist[j] < r_min - 1e-9:
            break
        selected.append(j)
    return cands[selected]


def _bottleneck_elimination(c, time_limit=5.0, step=0.002):
    """
    Repeatedly find the binding constraint (closest pair or wall violation)
    and push the offending points apart. This operates on the exact objective.
    """
    c = c.copy()
    start = time.time()
    best = _score(c)
    
    while time.time() < start + time_limit:
        # Compute current radius
        wall_dists = HALF - np.max(np.abs(c), axis=1)
        min_wall = np.min(wall_dists)
        
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        
        r = min(min_pair / 2.0, min_wall)
        
        if r > best + 1e-15:
            best = r
        
        # Determine bottleneck
        if min_wall < min_pair / 2.0:
            # Wall is bottleneck
            idx = int(np.argmin(wall_dists))
            # Find which coordinate is closest to wall
            abs_c = np.abs(c[idx])
            axis = int(np.argmax(abs_c))
            sign = np.sign(c[idx, axis])
            # Push inward
            c[idx, axis] -= sign * step
        else:
            # Pair is bottleneck
            flat_idx = int(np.argmin(d))
            i = flat_idx // N
            j = flat_idx % N
            if i == j:
                continue
            # Push i and j apart
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Random small push
                c[i] += np.random.randn(2) * step
                c[j] -= np.random.randn(2) * step
            else:
                unit = diff / dist
                c[i] += unit * step * 0.5
                c[j] -= unit * step * 0.5
        
        # Clamp to square
        c = np.clip(c, -HALF, HALF)
        
        # Decay step size
        step *= 0.999
    
    return c, best


def _simulated_annealing(c, time_limit=30.0, T0=0.01, T_end=1e-5, cooling=0.999):
    """
    Simulated annealing on the exact radius objective.
    Propose random perturbations, accept if better or with probability exp(-delta/T).
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _score(c)
    T = T0
    start = time.time()
    rng = np.random.default_rng(42)
    
    while time.time() < start + time_limit and T > T_end:
        # Propose a perturbation: pick 1-3 points and nudge them
        n_perturb = rng.integers(1, 4)
        indices = rng.choice(N, size=n_perturb, replace=False)
        c_new = c.copy()
        for idx in indices:
            c_new[idx] += rng.randn(2) * 0.005
        
        # Clamp
        c_new = np.clip(c_new, -HALF, HALF)
        
        r_new = _score(c_new)
        delta = r_new - best_r
        
        if delta > 0:
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                c = c_new
        
        T *= cooling
    
    return best_c, best_r


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s

    rng = np.random.default_rng(2024)

    best_c = None
    best_v = 0.0

    # Target spacing estimate for hex packing
    # For N circles of radius r in unit square, area ~ N*pi*r^2 <= 1
    # Hex packing density ~ pi/(2*sqrt(3)) ~ 0.9069
    # So N*pi*r^2 <= 0.9069 => r <= sqrt(0.9069/(N*pi))
    r_est = np.sqrt(0.9069 / (N * np.pi))
    s0 = 2 * r_est  # spacing = 2r

    # Multiple hex-grid trials with random offsets and varied spacing
    n_trials = 30
    for trial in range(n_trials):
        if time.time() > DEADLINE - 40:
            break
        s = s0 * (0.98 + 0.04 * rng.random())
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        cands = _hex_points(s, ox, oy)
        if cands.shape[0] < N:
            continue
        r_min = 0.5 * s * 0.9
        sel = _greedy_select(cands, N, r_min)
        if sel.shape[0] < N:
            continue
        v = _score(sel)
        if v > best_v:
            best_v = v
            best_c = sel.copy()

    if best_c is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_v = _score(best_c)

    # Phase 1: Bottleneck elimination (quick local improvement)
    if time.time() < DEADLINE - 20:
        polished, pv = _bottleneck_elimination(best_c, time_limit=8.0, step=0.003)
        if pv > best_v:
            best_c = polished
            best_v = pv

    # Phase 2: Simulated annealing (escape local minima)
    if time.time() < DEADLINE - 5:
        sa_c, sa_v = _simulated_annealing(best_c, time_limit=30.0, T0=0.008, T_end=1e-5)
        if sa_v > best_v:
            best_c = sa_c
            best_v = sa_v

    # Phase 3: Final bottleneck elimination polish
    if time.time() < DEADLINE - 2:
        final_c, final_v = _bottleneck_elimination(best_c, time_limit=3.0, step=0.001)
        if final_v > best_v:
            best_c = final_c
            best_v = final_v

    # Final validity clamp
    best_c = np.clip(best_c, -HALF, HALF)
    
    return best_c
