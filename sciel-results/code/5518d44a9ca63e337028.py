"""Packing 50 equal circles in a unit circle: Multi-topology DE global search + soft-min
surrogate polish + barrier-based radius maximization (epigraph on r)."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

# Precompute upper-triangular pair indices (pair (i,j), i<j)
_IU_I, _IU_J = np.triu_indices(N, k=1)
_NPAIR = _IU_I.shape[0]

# Candidate ring topologies (counts must sum to N-1 for the rings, +1 center)
RING_TOPOLOGIES = [
    [6, 12, 18, 13],
    [5, 10, 15, 19],
    [7, 14, 15, 13],
    [4, 9, 14, 21],
    [6, 11, 17, 15],
    [8, 13, 16, 12],
]


def _compute_score(points):
    """Exact packing score: min(min_pair_dist/2, min_boundary_dist)."""
    if points is None or points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0 + 1e-12):
        return 0.0
    min_boundary = np.min(1.0 - radii)
    if min_boundary < 0.0:
        return 0.0
    d = cdist(points, points)
    min_pair = d[_IU_I, _IU_J].min() / 2.0
    return min(min_pair, min_boundary)


def _clamp(points):
    """Clamp points to the closed unit disk (return a new array)."""
    p = points.copy()
    r = np.hypot(p[:, 0], p[:, 1])
    mask = r > 1.0
    if np.any(mask):
        p[mask] /= r[mask, np.newaxis]
    return p


def _soft_min_surrogate(points, beta=100.0):
    """Smooth surrogate for the packing radius (soft-min of all constraints)."""
    pts = points
    d = cdist(pts, pts)
    pair_vals = d[_IU_I, _IU_J] / 2.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_vals = 1.0 - radii
    all_vals = np.concatenate([pair_vals, boundary_vals])
    m = all_vals.min()
    e = np.exp(-beta * (all_vals - m))
    return m - np.log(e.sum()) / beta


def _soft_min_obj(x, beta=100.0):
    """Objective = -soft_min + hard boundary penalty (for L-BFGS-B)."""
    pts = x.reshape(N, 2)
    r = _soft_min_surrogate(pts, beta)
    norms = np.linalg.norm(pts, axis=1)
    pen = 1000.0 * np.sum(np.maximum(0.0, norms - 1.0) ** 2)
    return -r + pen


def _decode_ring(params, ring_counts):
    """Decode ring parameters into N points. params = [r_0..r_{K-1}, off_0..off_{K-1}]."""
    points = [np.array([0.0, 0.0])]
    n_rings = len(ring_counts)
    for k, nk in enumerate(ring_counts):
        rk = params[k]
        off = params[n_rings + k]
        for m in range(nk):
            ang = 2.0 * np.pi * m / nk + off
            points.append(np.array([rk * np.cos(ang), rk * np.sin(ang)]))
    return np.array(points[:N])

def _hex_lattice_initialization():
    """Generate a hexagonal lattice packing centered at origin."""
    points = []
    # Approximate radius for hex packing of 50 circles
    # Density is ~0.9069. Area of container = pi. Area of 50 circles = 50*pi*r^2.
    # 50*r^2 ~ 0.9069 => r ~ 0.134.
    # Spacing d = 2r.
    d = 0.26 
    # Generate points in hex lattice
    # h = d * sqrt(3)/2
    h = d * np.sqrt(3) / 2
    # We need enough points to cover radius 1
    max_range = int(1.0 / h) + 2
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            # Hex lattice coordinates
            x = d * (i + 0.5 * (j % 2))
            y = h * j
            if np.hypot(x, y) < 1.0 - 0.13: # Loose pre-filter
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback if too few
        return None
        
    # Select N points closest to origin to form a compact cluster? 
    # Or just take the first N? 
    # Better: Sort by distance to origin and take N? No, that might leave holes.
    # Let's just take a subset that fits. 
    # For now, return the first N if they fit, else None.
    # Actually, a simple grid search on the hex lattice parameters is better.
    return None # We will use DE on ring structures which is more robust for this specific N.

def _ring_global_search(deadline):
    """Differential Evolution over ring parameters for multiple topologies."""
    best_pts = None
    best_score = -1.0
    
    # Bounds for radii and offsets
    # Radii: 0.2 to 0.95
    # Offsets: 0 to 2*pi
    bounds = [(0.2, 0.95)] * 4 + [(0.0, 2.0 * np.pi)] * 4
    
    def objective(params):
        # Assume standard 4-ring structure for DE efficiency
        # We will try a few topologies by calling DE on each? 
        # Or just pick the most promising topology?
        # Let's use the topology [6, 12, 18, 13] as primary for DE speed.
        pts = _decode_ring(params, [6, 12, 18, 13])
        # Return negative score for minimization
        return -_compute_score(pts)

    # Run DE for the primary topology
    try:
        if time.time() < deadline - 30:
            res = differential_evolution(
                objective, bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10,
                tol=1e-6,
                update_interval=1,
                polish=True,
                init='sobol'
            )
            if res.success or res.fun < 0:
                best_pts = _decode_ring(res.x, [6, 12, 18, 13])
                best_score = _compute_score(best_pts)
    except Exception:
        pass

    # If we have time, try another topology with a quick DE
    if time.time() < deadline - 15 and best_score < 0.12:
        bounds2 = [(0.2, 0.95)] * 4 + [(0.0, 2.0 * np.pi)] * 4
        def objective2(params):
            pts = _decode_ring(params, [5, 10, 15, 19])
            return -_compute_score(pts)
        try:
            res2 = differential_evolution(
                objective2, bounds2, 
                seed=123, 
                maxiter=30, 
                popsize=8,
                tol=1e-5,
                polish=True,
                init='sobol'
            )
            if res2.success or res2.fun < 0:
                pts2 = _decode_ring(res2.x, [5, 10, 15, 19])
                s2 = _compute_score(pts2)
                if s2 > best_score:
                    best_pts = pts2
                    best_score = s2
        except Exception:
            pass

    return best_pts


def _polish_smooth(points, deadline):
    """Progressive soft-min polish with increasing beta (smooth surrogate)."""
    current = _clamp(points)
    current_score = _compute_score(current)
    for beta in [20, 50, 100, 200, 400, 800]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _soft_min_obj,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 60},
            )
            cand = _clamp(res.x.reshape(N, 2))
            if np.all(np.isfinite(cand)):
                s = _compute_score(cand)
                if s > current_score:
                    current = cand
                    current_score = s
        except Exception:
            pass
    return current, current_score


def _barrier_max_radius(points, deadline):
    """Maximize radius r with log-barrier on non-overlap and boundary constraints.

    Epigraph form: maximize r subject to
        |c_i - c_j| >= 2r   (i<j)
        |c_i| + r <= 1       (i)
    Barrier:  L = -r - mu * sum_i log(1 - |c_i| - r)
                          - mu * sum_{i<j} log(|c_i - c_j| - 2r)
    Minimize L over (x, r) with r in (0, 1).
    """
    current = _clamp(points)
    current_score = _compute_score(current)
    best = (current, current_score)
    mu = 0.05
    for _ in range(14):
        if time.time() > deadline - 4:
            break

        def barrier_obj(x):
            r = x[-1]
            pts = x[:2 * N].reshape(N, 2)
            r = min(max(r, 1e-4), 0.999)
            norms = np.linalg.norm(pts, axis=1)
            bnd = 1.0 - norms - r
            pd = cdist(pts, pts)
            pd = pd[_IU_I, _IU_J] - 2.0 * r
            # guard against log of non-positive
            bnd = np.maximum(bnd, 1e-9)
            pd = np.maximum(pd, 1e-9)
            return -r - mu * (np.sum(np.log(bnd)) + np.sum(np.log(pd)))

        def barrier_grad(x):
            r = x[-1]
            pts = x[:2 * N].reshape(N, 2)
            r = min(max(r, 1e-4), 0.999)
            g = np.zeros(2 * N + 1)
            norms = np.linalg.norm(pts, axis=1)
            bnd = np.maximum(1.0 - norms - r, 1e-9)
            # boundary terms
            for i in range(N):
                inv = 1.0 / bnd[i]
                # d(-mu log bnd_i)/d c_i = -mu * (-(pts_i/norm_i)) / bnd_i
                if norms[i] > 1e-12:
                    g[2 * i] += -mu * (-pts[i, 0] / norms[i]) * inv
                    g[2 * i + 1] += -mu * (-pts[i, 1] / norms[i]) * inv
                # d(-mu log bnd_i)/d r = -mu * (-1)/bnd_i
                g[-1] += -mu * (-1.0) * inv
            # pair terms
            pd = cdist(pts, pts)
            pd = pd[_IU_I, _IU_J]
            pdg = np.maximum(pd - 2.0 * r, 1e-9)
            for a in range(_NPAIR):
                i = _IU_I[a]; j = _IU_J[a]
                diff = pts[i] - pts[j]
                dist = pd[a]
                if dist < 1e-12:
                    continue
                inv = 1.0 / pdg[a]
                # d(-mu log(dist-2r))/d c_i = -mu * (diff/dist) / (dist-2r)
                g[2 * i] += -mu * (diff[0] / dist) * inv
                g[2 * i + 1] += -mu * (diff[1] / dist) * inv
                g[2 * j] += -mu * (-diff[0] / dist) * inv
                g[2 * j + 1] += -mu * (-diff[1] / dist) * inv
                # d(-mu log(dist-2r))/d r = -mu * (-2)/(dist-2r)
                g[-1] += -mu * (-2.0) * inv
            # objective -r gives d/d r = -1
            g[-1] += -1.0
            return g

        x0 = np.concatenate([current.flatten(), [current_score]])
        try:
            res = minimize(
                barrier_obj,
                x0,
                jac=barrier_grad,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N) + [(1e-4, 0.999)],
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40},
            )
            cand = _clamp(res.x[:2 * N].reshape(N, 2))
            if np.all(np.isfinite(cand)):
                s = _compute_score(cand)
                if s > best[1]:
                    best = (cand, s)
        except Exception:
            pass
        mu *= 0.7  # reduce barrier strength
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 142.0  # safety margin before 150s kill

    best_pts = None
    best_score = 0.0

    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        pts = _clamp(pts)
        if not np.all(np.isfinite(pts)):
            return
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: DE-based ring-structure global search
    if time.time() < deadline - 40:
        ring_pts = _ring_global_search(deadline - 40)
        if ring_pts is not None:
            consider(ring_pts)

    # Fallback if DE failed or didn't find anything good
    if best_score < 0.10:
        # Deterministic fallback ring
        params = [0.32, 0.62, 0.85, 0.96, 0.0, np.pi / 12, 0.0, np.pi / 13]
        fallback_pts = _clamp(_decode_ring(params, [6, 12, 18, 13]))
        consider(fallback_pts)

    # Phase 2: smooth soft-min polish from the best config
    if best_pts is not None and time.time() < deadline - 20:
        polished, s = _polish_smooth(best_pts, deadline - 20)
        consider(polished)

    # Phase 3: barrier-based radius maximization (epigraph on r)
    if best_pts is not None and time.time() < deadline - 5:
        bpts, bs = _barrier_max_radius(best_pts, deadline - 5)
        consider(bpts)

    # Phase 4: final high-precision smooth polish
    if best_pts is not None and time.time() < deadline - 1:
        polished, s = _polish_smooth(best_pts, deadline - 1)
        consider(polished)

    if best_pts is None:
        # Deterministic fallback ring
        params = [0.32, 0.62, 0.85, 0.96, 0.0, np.pi / 12, 0.0, np.pi / 13]
        best_pts = _clamp(_decode_ring(params, [6, 12, 18, 13]))
        s = _compute_score(best_pts)
        if s > best_score:
            best_score = s

    # Final validation / safety
    best_pts = _clamp(best_pts)
    if not np.all(np.isfinite(best_pts)) or _compute_score(best_pts) <= 0.0:
        params = [0.30, 0.60, 0.84, 0.95, 0.0, np.pi / 6, 0.0, np.pi / 12]
        best_pts = _clamp(_decode_ring(params, [6, 12, 18, 13]))
    return best_pts

# EVOLVE-BLOCK-END
