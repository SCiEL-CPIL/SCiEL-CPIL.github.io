"""Packing 50 equal circles in a unit circle using ring-structured parametric optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _score(c):
    """Compute the maximum common radius for given centers."""
    c = c.reshape(-1, 2)
    n = len(c)
    # Boundary constraint
    radii = 1.0 - np.linalg.norm(c, axis=1)
    r = np.min(radii)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _softmin_logsumexp(values, beta):
    """Numerically stable soft min via log-sum-exp."""
    shifted = -beta * values
    max_shift = np.max(shifted)
    if max_shift > 500:
        return np.min(values)
    log_sum = np.log(np.sum(np.exp(shifted - max_shift))) + max_shift
    return log_sum / beta


def _smooth_objective(x, beta):
    """
    Smooth objective: -softmin of all constraints (pairwise distances/2 and boundary distances).
    We want to maximize the minimum, so we minimize the negative soft min.
    """
    c = x.reshape(N, 2)
    # Boundary distances
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    all_vals = np.concatenate([b, pair_dists])
    return -_softmin_logsumexp(all_vals, beta)


def _ring_initialization(n_rings=5, ring_counts=None, ring_radii=None, ring_phases=None):
    """
    Initialize with ring structure.
    Ring 0: center
    Ring 1: 6 circles
    Ring 2: 12 circles
    Ring 3: 18 circles
    Ring 4: remaining
    """
    if ring_counts is None:
        ring_counts = [1, 6, 12, 18, 13]
    if ring_radii is None:
        ring_radii = [0.0, 0.25, 0.50, 0.75, 0.92]
    if ring_phases is None:
        ring_phases = [0.0, 0.0, np.pi / 12, 0.0, np.pi / 26]
    
    points = []
    for ring_idx, (count, radius, phase) in enumerate(zip(ring_counts, ring_radii, ring_phases)):
        if count == 0:
            continue
        if count == 1 and radius < 0.01:
            points.append([0.0, 0.0])
        else:
            for k in range(count):
                angle = 2 * np.pi * k / count + phase
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(points[:N])


def _optimize_from_init(init_points, time_limit=15.0, beta_schedule=None):
    """
    Optimize from initial points using L-BFGS-B with increasing beta.
    """
    if beta_schedule is None:
        beta_schedule = [100.0, 200.0, 400.0, 800.0]
    
    best_c = init_points.copy()
    best_r = _score(best_c)
    t_start = time.time()
    
    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break
        
        def objective(x):
            return _smooth_objective(x, beta)
        
        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                objective,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16},
                callback=callback
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _binary_search_polish(c, r, time_limit=5.0):
    """
    Binary search on radius to find the maximum feasible radius.
    For a given r, check if we can place all circles with radius r.
    """
    t_start = time.time()
    lo, hi = r * 0.999, r * 1.001
    best_c = c.copy()
    best_r = r
    
    # Try to increase r slightly
    for _ in range(20):
        if time.time() - t_start >= time_limit:
            break
        mid = (lo + hi) / 2
        # Try to optimize with target radius mid
        # Use a penalty-based approach
        def penalized_obj(x, target_r=mid):
            c_local = x.reshape(N, 2)
            # Boundary penalty
            b = 1.0 - np.linalg.norm(c_local, axis=1)
            b_pen = np.sum(np.maximum(0, target_r - b) ** 2)
            # Pairwise penalty
            diff = c_local[:, np.newaxis, :] - c_local[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            iu = np.triu_indices(N, 1)
            pair_dists = dists[iu] / 2.0
            p_pen = np.sum(np.maximum(0, target_r - pair_dists) ** 2)
            return b_pen + p_pen
        
        try:
            res = minimize(
                penalized_obj,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r_cand = _score(cand)
                if r_cand >= mid - 1e-10:
                    lo = mid
                    best_c = cand.copy()
                    best_r = r_cand
                else:
                    hi = mid
        except Exception:
            break
    
    return best_c, best_r


def _de_ring_optimize(c, r, time_limit=10.0):
    """
    Differential-evolution-like optimization on ring parameters.
    """
    t_start = time.time()
    best_c = c.copy()
    best_r = r
    
    # Extract ring structure from current best
    # Sort by radius from center
    radii = np.linalg.norm(c, axis=1)
    angles = np.arctan2(c[:, 1], c[:, 0])
    
    # Try small perturbations
    for _ in range(50):
        if time.time() - t_start >= time_limit:
            break
        
        # Perturb angles slightly
        new_c = c.copy()
        for i in range(N):
            if radii[i] > 0.01:
                angle_noise = np.random.randn() * 0.02
                new_c[i, 0] = radii[i] * np.cos(angles[i] + angle_noise)
                new_c[i, 1] = radii[i] * np.sin(angles[i] + angle_noise)
        
        # Optimize from perturbed position
        cand, r_cand = _optimize_from_init(new_c, time_limit=min(3.0, time_limit - (time.time() - t_start)), beta_schedule=[200.0, 400.0])
        if r_cand > best_r:
            best_r = r_cand
            best_c = cand.copy()
    
    return best_c, best_r


def construct():
    t_start = time.time()
    
    # Multiple initializations
    inits = []
    
    # Ring initialization with various parameters
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.92], [0.0, 0.0, np.pi/12, 0.0, np.pi/26]),
        ([1, 6, 12, 18, 13], [0.0, 0.24, 0.48, 0.72, 0.90], [0.0, 0.0, np.pi/12, 0.0, np.pi/26]),
        ([1, 6, 12, 18, 13], [0.0, 0.26, 0.52, 0.78, 0.94], [0.0, 0.0, np.pi/12, 0.0, np.pi/26]),
        ([1, 6, 12, 17, 14], [0.0, 0.25, 0.50, 0.74, 0.92], [0.0, 0.0, np.pi/12, 0.0, np.pi/28]),
        ([1, 6, 13, 17, 13], [0.0, 0.25, 0.51, 0.75, 0.93], [0.0, 0.0, np.pi/13, 0.0, np.pi/26]),
    ]
    
    for counts, radii, phases in ring_configs:
        inits.append(_ring_initialization(ring_counts=counts, ring_radii=radii, ring_phases=phases))
    
    # Hexagonal lattice initialization
    best_hex = None
    best_hex_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 30):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            cand = points[idx]
            r = _score(cand)
            if r > best_hex_r:
                best_hex_r = r
                best_hex = cand
    
    if best_hex is not None:
        inits.append(best_hex)
    
    # Optimize each initialization
    best_c = None
    best_r = 0.0
    
    for init in inits:
        if time.time() - t_start >= DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - t_start)
        c, r = _optimize_from_init(init, time_limit=min(15.0, time_left * 0.5))
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        # Fallback
        best_c = _ring_initialization()
        best_r = _score(best_c)
    
    # Binary search polish
    if time.time() - t_start < DEADLINE - 5:
        time_left = DEADLINE - 5 - (time.time() - t_start)
        best_c, best_r = _binary_search_polish(best_c, best_r, time_limit=min(5.0, time_left))
    
    # DE ring optimization for final improvement
    if time.time() - t_start < DEADLINE - 2:
        time_left = DEADLINE - 2 - (time.time() - t_start)
        best_c, best_r = _de_ring_optimize(best_c, best_r, time_limit=min(10.0, time_left))
    
    # Final binary search
    if time.time() - t_start < DEADLINE - 1:
        time_left = DEADLINE - 1 - (time.time() - t_start)
        best_c, best_r = _binary_search_polish(best_c, best_r, time_limit=min(3.0, time_left))
    
    return best_c
