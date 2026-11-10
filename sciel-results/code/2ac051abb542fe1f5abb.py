"""Combined program: ring-based representation + smooth surrogate optimization + local polish."""
import numpy as np
import scipy.optimize as opt
import time

N = 300

_DEADLINE = 135.0  # safety margin before 150 s evaluator kill
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _min_radius(pts):
    """Return the achievable common radius for a set of centres (double precision)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    d_center = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - d_center.max()
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * d.min()
    return min(r_cont, r_pair)


def _score(pts):
    return _min_radius(pts)


def _hex_init(n=N):
    """Hexagonal lattice packed into unit disk, take n nearest to origin."""
    for spacing in [0.06, 0.058, 0.056, 0.055, 0.054, 0.053, 0.052, 0.051, 0.050, 0.048, 0.046, 0.044, 0.042, 0.040]:
        pts = []
        R = 1.0
        dy = spacing * np.sqrt(3.0) / 2.0
        j = 0
        while True:
            y = j * dy
            if y > R:
                break
            for i in range(-20, 21):
                x = (i + 0.5 * (j % 2)) * spacing
                if x * x + y * y <= (R - spacing / 2.0) ** 2:
                    pts.append((x, y))
            j += 1
        arr = np.array(pts)
        if len(arr) >= n:
            r = np.hypot(arr[:, 0], arr[:, 1])
            order = np.argsort(r, kind="stable")
            return arr[order[:n]].copy()
    # fallback random
    return _random_init()


def _spiral_init(n=N):
    """Sunflower / Vogel spiral filling the disk."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    ang = np.arange(n) * golden * 2.0 * np.pi
    r = np.sqrt(np.arange(n) / float(n)) * 0.999
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def _rings_init(n=N):
    """Concentric rings with even angular spacing."""
    pts = [(0.0, 0.0)]
    k = 1
    while len(pts) < n:
        m = int(np.ceil(2.0 * np.pi * k * 0.5))
        m = max(m, 6)
        ang = np.linspace(0, 2.0 * np.pi, m, endpoint=False)
        rr = 0.999 * k / 5.0
        for a in ang:
            if len(pts) < n:
                pts.append((rr * np.cos(a), rr * np.sin(a)))
        k += 1
    return np.array(pts[:n])


def _random_init(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(N, 2))
    keep = np.hypot(pts[:, 0], pts[:, 1]) <= 0.999
    while keep.sum() < N:
        extra = rng.uniform(-1, 1, size=(N - keep.sum(), 2))
        pts = np.vstack([pts[keep], extra])
        keep = np.hypot(pts[:, 0], pts[:, 1]) <= 0.999
    return pts[:N]


def _soft_min(arr, beta=50.0):
    """Smooth approximation to min using log-sum-exp."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    m = arr.min()
    if m < -1e10:
        return m
    return m - np.log(np.sum(np.exp(beta * (arr - m)))) / beta


def _objective_ring(params, n_rings, ring_counts, beta=50.0):
    """
    Objective for ring-based optimization.
    params: flat array of [ring_radii, angular_offsets]
    Returns negative of the soft-min radius (to minimize).
    """
    n_rings = len(ring_counts)
    r_radii = params[:n_rings].copy()
    r_offsets = params[n_rings:].copy()
    
    # Ensure radii are in [0, 1]
    r_radii = np.clip(r_radii, 0.0, 1.0)
    
    # Build points
    pts = np.zeros((N, 2))
    idx = 0
    for k in range(n_rings):
        m = ring_counts[k]
        if m == 0:
            continue
        r_k = r_radii[k]
        offset = r_offsets[k]
        angles = offset + 2.0 * np.pi * np.arange(m) / m
        pts[idx:idx + m, 0] = r_k * np.cos(angles)
        pts[idx:idx + m, 1] = r_k * np.sin(angles)
        idx += m
    
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Compute container distances
    rad = np.hypot(pts[:, 0], pts[:, 1])
    d_cont = 1.0 - rad
    
    # We want to maximize min(0.5*min_pair_dist, min_cont_dist)
    # So minimize -min(0.5*min_pair_dist, min_cont_dist)
    
    # Use soft min for smoothness
    # Combine all constraints: pair distances (halved) and container distances
    pair_vals = 0.5 * d[d < np.inf]
    cont_vals = d_cont
    
    # We want the minimum of all these values
    # Use soft min over a combined set
    all_vals = np.concatenate([pair_vals, cont_vals])
    soft_min_val = _soft_min(all_vals, beta=beta)
    
    return -soft_min_val


def _optimize_ring(initial_radii, initial_offsets, n_rings, ring_counts, max_iter=2000):
    """
    Optimize ring parameters using L-BFGS-B.
    """
    n_params = 2 * n_rings
    x0 = np.concatenate([initial_radii, initial_offsets])
    
    def objective(x):
        return _objective_ring(x, n_rings, ring_counts, beta=50.0)
    
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < 0:  # valid
            best_radius = -res.fun
            r_radii = np.clip(res.x[:n_rings], 0.0, 1.0)
            r_offsets = res.x[n_rings:]
            
            # Build final points
            pts = np.zeros((N, 2))
            idx = 0
            for k in range(n_rings):
                m = ring_counts[k]
                if m == 0:
                    continue
                r_k = r_radii[k]
                offset = r_offsets[k]
                angles = offset + 2.0 * np.pi * np.arange(m) / m
                pts[idx:idx + m, 0] = r_k * np.cos(angles)
                pts[idx:idx + m, 1] = r_k * np.sin(angles)
                idx += m
            
            return pts, best_radius
    except Exception:
        pass
    
    return None, 0.0


def _local_optimize(pts, seed, max_iter=300):
    """Greedy coordinate descent on the minimum constraint."""
    pts = np.array(pts, dtype=float, copy=True)
    best = _score(pts)
    best_pts = pts.copy()
    rng = np.random.default_rng(seed)

    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        r_pair = 0.5 * d[i, j]
        rad = np.hypot(pts[:, 0], pts[:, 1])
        k = int(np.argmax(rad))
        r_cont = 1.0 - rad[k]
        if r_pair < r_cont:
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            L = np.hypot(dx, dy)
            if L < 1e-12:
                pts[i, 0] += 1e-4
                pts[j, 0] -= 1e-4
            else:
                ux, uy = dx / L, dy / L
                eps = 1e-4
                pts[i, 0] += ux * eps
                pts[i, 1] += uy * eps
                pts[j, 0] -= ux * eps
                pts[j, 1] -= uy * eps
        else:
            ux, uy = pts[k, 0], pts[k, 1]
            L = np.hypot(ux, uy)
            if L < 1e-12:
                continue
            eps = 1e-4
            pts[k, 0] -= ux / L * eps
            pts[k, 1] -= uy / L * eps

        s = _score(pts)
        if s > best:
            best = s
            best_pts = pts.copy()
    return best_pts, best


def _relax(pts, seed, max_iter, dt_base=0.02, temp_base=0.5):
    """Repulsion + confinement relaxation. Returns (best_pts, best_score)."""
    rng = np.random.default_rng(seed)
    pts = np.array(pts, dtype=float, copy=True)
    best = _score(pts)
    best_pts = pts.copy()

    r_target = 0.052
    n = N

    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)

        thresh = 2.0 * r_target
        mask = d < thresh
        strength = np.where(mask, (1.0 - d / thresh) ** 2, 0.0)
        d_safe = np.where(d > 1e-12, d, 1.0)
        dirx = diff[:, :, 0] / d_safe
        diry = diff[:, :, 1] / d_safe
        fx = (strength * dirx).sum(axis=1)
        fy = (strength * diry).sum(axis=1)

        rad = np.hypot(pts[:, 0], pts[:, 1])
        Rlim = 1.0 - r_target
        conf = np.where(rad > Rlim, (rad - Rlim) / r_target, 0.0)
        rad_safe = np.where(rad > 1e-12, rad, 1.0)
        fx -= conf * pts[:, 0] / rad_safe
        fy -= conf * pts[:, 1] / rad_safe

        jitter = rng.normal(0, temp_base * dt_base, size=(n, 2))
        fx += jitter[:, 0]
        fy += jitter[:, 1]

        fmag = np.sqrt(fx * fx + fy * fy)
        fmag_safe = np.where(fmag > 1e-12, fmag, 1.0)
        step = dt_base
        fx = fx / fmag_safe * np.minimum(fmag, 1.0) * step
        fy = fy / fmag_safe * np.minimum(fmag, 1.0) * step

        pts[:, 0] += fx
        pts[:, 1] += fy

        rad = np.hypot(pts[:, 0], pts[:, 1])
        over = rad > Rlim
        if over.any():
            scale = Rlim / rad[over]
            pts[over, 0] *= scale
            pts[over, 1] *= scale

        s = _score(pts)
        if s > best:
            best = s
            best_pts = pts.copy()

        dt_base *= 0.995
        temp_base *= 0.998

    return best_pts, best


def _build_ring_counts(n=N, n_rings=15):
    """
    Distribute N points into n_rings concentric rings.
    Ring k (0-indexed) has proportional points based on circumference.
    """
    # Approximate: ring radius ~ k/(n_rings-1), circumference ~ k
    # Points per ring proportional to circumference
    weights = np.arange(1, n_rings + 1, dtype=float)
    weights = weights / weights.sum()
    counts = np.round(weights * n).astype(int)
    # Adjust to sum to n
    diff = n - counts.sum()
    if diff > 0:
        counts[np.argsort(weights)[::-1][:diff]] += 1
    elif diff < 0:
        counts[np.argsort(weights)[:abs(diff)]] -= 1
    counts = np.maximum(counts, 1)
    # Re-adjust
    diff = n - counts.sum()
    if diff != 0:
        idx = np.argmax(counts) if diff > 0 else np.argmin(counts)
        counts[idx] += diff
    return counts


def construct():
    global _T0
    _T0 = time.time()

    best_pts = None
    best_score = -1.0

    # Phase 1: Ring-based optimization
    n_rings = 15
    ring_counts = _build_ring_counts(N, n_rings)
    
    # Initial ring radii: evenly spaced from 0 to 0.95
    init_radii = np.linspace(0.0, 0.95, n_rings)
    init_offsets = np.zeros(n_rings)
    
    pts_ring, score_ring = _optimize_ring(init_radii, init_offsets, n_rings, ring_counts, max_iter=1500)
    if pts_ring is not None and score_ring > best_score:
        best_score = score_ring
        best_pts = pts_ring.copy()
    
    # Local polish on ring result
    if best_pts is not None and _remaining() > 20.0:
        pts_polished, score_polished = _local_optimize(best_pts, seed=1, max_iter=300)
        if score_polished > best_score:
            best_score = score_polished
            best_pts = pts_polished.copy()

    # Phase 2: Multi-start with hex, spiral, random
    inits = []
    inits.append(("hex", _hex_init()))
    inits.append(("spiral", _spiral_init()))
    inits.append(("rings", _rings_init()))
    for s in (1, 2, 3):
        inits.append((f"rand{s}", _random_init(s)))

    for idx, (name, init) in enumerate(inits):
        if _remaining() < 15.0:
            break
        max_iter = 600 if idx < 3 else 400
        pts, s = _relax(init, seed=100 + idx, max_iter=max_iter)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 3: Final local optimization
    if _remaining() > 15.0 and best_pts is not None:
        pts, s = _local_optimize(best_pts, seed=7, max_iter=400)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Final safety
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init()

    # Clamp to be strictly inside
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    over = rad > 0.999999
    if over.any():
        scale = 0.999999 / rad[over]
        best_pts[over, 0] *= scale
        best_pts[over, 1] *= scale

    return best_pts[:N]
