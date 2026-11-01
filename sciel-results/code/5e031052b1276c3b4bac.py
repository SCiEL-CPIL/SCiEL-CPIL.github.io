"""Packing 50 equal circles in a unit disk - multi-strategy optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Boundary constraint
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary = np.min(1.0 - radii)
    if boundary < 0:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    min_pair = np.min(d) / 2.0
    return min(boundary, min_pair)


def _objective(x, N=N):
    """Objective to minimize: negative of the minimum radius."""
    pts = x.reshape(-1, 2)
    # Boundary penalty
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - radii
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(pts)
    min_pair = np.min(d) / 2.0
    return -min(boundary, min_pair)


def _hex_init():
    """Hexagonal lattice initialization."""
    spacing = 0.24
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Pick the N closest to center
            idx = np.argsort(radius)
            return points[idx[:N]].copy()
        spacing *= 0.97


def _ring_init():
    """Ring-based initialization: concentric rings."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1: 6 points
    r1 = 0.22
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    # Ring 2: 12 points
    r2 = 0.42
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    # Ring 3: 18 points
    r3 = 0.60
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    # Ring 4: remaining
    r4 = 0.78
    remaining = N - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining + np.pi / remaining
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    return np.array(points[:N])


def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(N):
        while True:
            r = rng.uniform(0, 0.95)
            theta = rng.uniform(0, 2 * np.pi)
            points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def _nelder_mead_optimize(points, max_iter=500, timeout=30.0):
    """Optimize using Nelder-Mead."""
    x0 = points.flatten()
    t0 = time.time()
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    def time_check():
        return (time.time() - t0) > timeout
    
    result = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True, 'maxfev': max_iter * N})
    
    if result.fun < best_obj:
        best_obj = result.fun
        best_x = result.x.copy()
    
    return best_x.reshape(-1, 2)


def _perturb_and_optimize(points, n_perturbs=5, timeout=20.0):
    """Apply random perturbations and re-optimize to escape local minima."""
    best = points.copy()
    best_score = _score(best)
    t0 = time.time()
    
    for i in range(n_perturbs):
        if time.time() - t0 > timeout:
            break
        perturbed = best + np.random.randn(*best.shape) * 0.02
        # Clamp to disk
        radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
        mask = radii > 0.95
        perturbed[mask] = perturbed[mask] * (0.95 / radii[mask])
        opt = _nelder_mead_optimize(perturbed, max_iter=200, timeout=max(1.0, timeout - (time.time() - t0)))
        s = _score(opt)
        if s > best_score:
            best = opt
            best_score = s
    return best


def _progressive_polish(points, betas=(50, 100, 200, 400, 800), timeout=30.0):
    """Progressive beta polishing: increase constraint tightness."""
    t0 = time.time()
    best = points.copy()
    best_score = _score(best)
    
    for beta in betas:
        if time.time() - t0 > timeout:
            break
        # Optimize with tighter tolerance
        x0 = best.flatten()
        result = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True, 'maxfev': 500})
        opt = result.x.reshape(-1, 2)
        s = _score(opt)
        if s > best_score:
            best = opt
            best_score = s
    return best


def construct():
    global _start_time
    _start_time = time.time()
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Hexagonal lattice
    try:
        hex_pts = _hex_init()
        inits.append(hex_pts)
    except Exception:
        pass
    
    # 2. Ring-based
    try:
        ring_pts = _ring_init()
        inits.append(ring_pts)
    except Exception:
        pass
    
    # 3. Random seeds
    for seed in [42, 123, 7]:
        try:
            rand_pts = _random_init(seed)
            inits.append(rand_pts)
        except Exception:
            pass
    
    if not inits:
        # Fallback: simple grid
        spacing = 0.24
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 0.9:
                    points.append([x, y])
        if len(points) < N:
            spacing = 0.20
            points = []
            for i in range(-6, 7):
                for j in range(-6, 7):
                    x = i * spacing
                    y = j * spacing * np.sqrt(3) / 2
                    if np.hypot(x, y) <= 0.9:
                        points.append([x, y])
        return np.array(points[:N])
    
    # Optimize each initialization and keep the best
    best_points = None
    best_score = 0.0
    
    for init in inits:
        if _remaining() < 15.0:
            break
        
        # Initial Nelder-Mead optimization
        try:
            opt_pts = _nelder_mead_optimize(init, max_iter=300, timeout=25.0)
            s = _score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        except Exception:
            continue
    
    if best_points is None:
        return inits[0]
    
    # Perturbation-based local search to escape local minima
    if _remaining() > 20.0:
        try:
            perturbed = _perturb_and_optimize(best_points, n_perturbs=8, timeout=18.0)
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed
        except Exception:
            pass
    
    # Progressive polishing
    if _remaining() > 10.0:
        try:
            polished = _progressive_polish(best_points, betas=(50, 100, 200, 400, 800), timeout=8.0)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_points = polished
        except Exception:
            pass
    
    # Final validation and clamping
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii > 0.999
    if np.any(mask):
        best_points[mask] = best_points[mask] * (0.999 / radii[mask])
    
    return best_points
