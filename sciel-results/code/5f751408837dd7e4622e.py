"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _radius(pts):
    """Compute the common radius for a set of centres."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Boundary distances
    margins = 0.5 - np.abs(pts)
    min_bound = np.min(margins)
    return 0.5 * min(min_pair, 2.0 * min_bound)

def _radius_fast(pts):
    """Faster radius computation for large arrays."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    # Use broadcasting in chunks to save memory
    min_pair = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        if d.size > 0:
            m = np.min(d)
            if m < min_pair:
                min_pair = m
    margins = 0.5 - np.abs(pts)
    min_bound = np.min(margins)
    return 0.5 * min(min_pair, 2.0 * min_bound)

def _score(pts):
    """Negative radius for minimization."""
    return -_radius_fast(pts)

def _hex_initial():
    """Create a hexagonal-like initial arrangement."""
    # Start with a grid, then perturb to hexagonal pattern
    m = 8  # 8x8 grid gives 64 positions, we take 50
    spacing = 1.0 / m
    centers = []
    for row in range(m):
        y = -0.5 + spacing * (row + 0.5)
        offset = 0.0 if row % 2 == 0 else spacing / 2.0
        for col in range(m):
            x = -0.5 + spacing * (col + 0.5) + offset
            if x < 0.5 - spacing/2 and x > -0.5 + spacing/2:
                centers.append([x, y])
    centers = np.array(centers[:N])
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    # Clamp to square
    centers = np.clip(centers, -0.45, 0.45)
    return centers

def _simulated_annealing(pts, time_limit):
    """Simulated annealing to maximize radius."""
    np.random.seed(123)
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _radius_fast(pts)
    
    t_start = time.time()
    T = 0.01
    T_min = 0.0001
    cooling = 0.999
    step_scale = 0.02
    
    n = len(pts)
    while time.time() - t_start < time_limit:
        # Pick a random point to move
        idx = np.random.randint(n)
        # Random direction and step
        angle = np.random.uniform(0, 2 * np.pi)
        step = np.random.exponential(step_scale) * T / 0.01
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_pts = pts.copy()
        new_pts[idx, 0] += dx
        new_pts[idx, 1] += dy
        
        # Clamp to square
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _radius_fast(new_pts)
        delta = new_r - best_r
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_r

def _lbfgs_polish(pts, time_limit, ftol=1e-14, gtol=1e-12):
    """L-BFGS polish with tight tolerances."""
    t_start = time.time()
    x0 = pts.flatten().copy()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        if time.time() - t_start > time_limit:
            return np.inf
        pts = x.reshape(N, 2)
        return _score(pts)
    
    def gradient(x):
        if time.time() - t_start > time_limit:
            return np.zeros_like(x)
        pts = x.reshape(N, 2)
        # Numerical gradient
        eps = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 5000, 'ftol': ftol, 'gtol': gtol,
                               'maxfun': 500000, 'disp': False})
    
    if result.success or result.fun < _score(x0):
        return result.x.reshape(N, 2), -result.fun
    else:
        return pts, _radius_fast(pts)

def _multi_start_optimize(time_budget):
    """Try multiple initializations and local optimizations."""
    t_start = time.time()
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Hexagonal initial
    pts1 = _hex_initial()
    r1 = _radius_fast(pts1)
    
    # Strategy 2: Grid with perturbation
    m = 7
    spacing = 1.0 / m
    pts2 = np.array([[ -0.5 + spacing*(i+0.5), -0.5 + spacing*(j+0.5) ]
                     for j in range(m) for i in range(m)])
    pts2 = pts2[:N]
    np.random.seed(99)
    pts2 += np.random.uniform(-0.01, 0.01, size=pts2.shape)
    pts2 = np.clip(pts2, -0.49, 0.49)
    r2 = _radius_fast(pts2)
    
    # Strategy 3: Another hexagonal variant
    pts3 = _hex_initial().copy()
    np.random.seed(77)
    pts3 += np.random.uniform(-0.01, 0.01, size=pts3.shape)
    pts3 = np.clip(pts3, -0.49, 0.49)
    r3 = _radius_fast(pts3)
    
    candidates = [(pts1, r1), (pts2, r2), (pts3, r3)]
    candidates.sort(key=lambda x: -x[1])
    
    # Run SA on top candidates
    for pts_init, r_init in candidates[:2]:
        elapsed = time.time() - t_start
        if elapsed > time_budget * 0.4:
            break
        sa_time = min(time_budget * 0.3, time_budget - elapsed - 10)
        if sa_time < 5:
            break
        pts_sa, r_sa = _simulated_annealing(pts_init, sa_time)
        if r_sa > best_r:
            best_r = r_sa
            best_pts = pts_sa
    
    # Polish with L-BFGS
    elapsed = time.time() - t_start
    if best_pts is not None and time_budget - elapsed > 5:
        polish_time = min(time_budget - elapsed - 2, 30)
        pts_pol, r_pol = _lbfgs_polish(best_pts, polish_time)
        if r_pol > best_r:
            best_r = r_pol
            best_pts = pts_pol
    
    # Final ultra-tight polish
    elapsed = time.time() - t_start
    if best_pts is not None and time_budget - elapsed > 3:
        polish_time = min(time_budget - elapsed - 1, 10)
        pts_pol2, r_pol2 = _lbfgs_polish(best_pts, polish_time, ftol=1e-16, gtol=1e-14)
        if r_pol2 > best_r:
            best_r = r_pol2
            best_pts = pts_pol2
    
    if best_pts is None:
        best_pts = candidates[0][0]
        best_r = candidates[0][1]
    
    return best_pts, best_r

def construct():
    """Return centres of 50 equal circles in [-1/2, 1/2]^2 maximizing radius."""
    t_start = time.time()
    time_budget = DEADLINE - (time.time() - t_start) - 5.0  # 5s safety margin
    
    if time_budget < 10:
        # Very little time, just return a good grid
        return _hex_initial()
    
    pts, r = _multi_start_optimize(time_budget)
    
    # Final validation: ensure all points are inside and finite
    pts = np.asarray(pts, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return _hex_initial()
    pts = np.clip(pts, -0.5 + 1e-10, 0.5 - 1e-10)
    
    # Verify shape
    if pts.shape != (N, 2):
        return _hex_initial()
    
    return pts
