"""Stronger algorithm for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _score(pts):
    """Compute the packing radius for a set of points."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Check all points are inside the unit circle
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0 + 1e-9):
        return 0.0
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(pts)
    min_pair = np.min(d) if len(d) > 0 else 1.0
    min_boundary = 1.0 - np.max(dists)
    r = min(min_pair / 2.0, min_boundary)
    return max(0.0, r)

def _hex_lattice_init():
    """Generate a hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by distance from center and take first N
            idx = np.argsort(radius[mask])[:N]
            return points[idx]
        spacing *= 0.97

def _random_init():
    """Generate a random initialization inside the unit circle."""
    pts = []
    for _ in range(N):
        r = np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        pts.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(pts)

def _ring_init():
    """Generate points on concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        theta = 2 * np.pi * i / 6
        pts.append([0.3 * np.cos(theta), 0.3 * np.sin(theta)])
    # Ring 2: 12 points
    for i in range(12):
        theta = 2 * np.pi * i / 12
        pts.append([0.6 * np.cos(theta), 0.6 * np.sin(theta)])
    # Ring 3: remaining points
    remaining = N - len(pts)
    for i in range(remaining):
        theta = 2 * np.pi * i / remaining
        pts.append([0.85 * np.cos(theta), 0.85 * np.sin(theta)])
    return np.array(pts[:N])

def _force_directed_precondition(pts, iterations=200):
    """Apply force-directed relaxation to spread out points."""
    pts = pts.copy()
    for it in range(iterations):
        forces = np.zeros_like(pts)
        # Repulsion between all pairs
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    # Random nudge
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                else:
                    # Repulsive force ~ 1/d^2
                    f = diff / (dist ** 3)
                    forces[i] += f
                    forces[j] -= f
        # Attraction to stay within bounds (soft)
        for i in range(N):
            dist = np.linalg.norm(pts[i])
            if dist > 0.8:
                forces[i] -= pts[i] / dist * (dist - 0.8) * 0.5
        # Update positions with small step
        step = 0.001
        pts += step * forces
        # Clamp to unit circle
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        pts[mask] /= dists[mask, None]
    return pts

def _soft_min_objective(x, beta):
    """
    Soft-min surrogate for the packing radius.
    x: flat array of 2*N coordinates
    Returns negative soft-min of (pairwise distances/2, boundary distances)
    """
    pts = x.reshape(N, 2)
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(pts)
    pair_terms = d / 2.0
    # Boundary distances
    dists = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - dists
    # Combine all terms
    all_terms = np.concatenate([pair_terms, boundary_terms])
    # Soft min via log-sum-exp (negative log-sum-exp for min)
    # soft_min(x) = -logsumexp(-x) / beta
    m = np.min(all_terms)
    log_sum = m + np.log(np.sum(np.exp(-beta * (all_terms - m))))
    soft_min_val = -log_sum / beta
    # We want to maximize, so minimize negative
    return -soft_min_val

def _optimize_from_start(pts0, time_budget=30.0):
    """
    Optimize starting from pts0 using gradient-based methods with annealing.
    """
    start_time = time.time()
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    
    # Multi-stage beta annealing
    betas = [50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]
    
    x0 = pts0.flatten()
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
        
        def objective(x):
            return _soft_min_objective(x, beta)
        
        def gradient(x):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
        
        # Use L-BFGS-B with bounds
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          jac=gradient, bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < objective(x0):
                x0 = res.x
                pts_new = x0.reshape(N, 2)
                score_new = _score(pts_new)
                if score_new > best_score:
                    best_score = score_new
                    best_pts = pts_new.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Generate multiple initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(3):
        inits.append(_random_init())
    
    best_pts = None
    best_score = 0.0
    
    # Precondition and optimize each initialization
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:  # Safety margin
            break
        
        # Force-directed preconditioning
        precond = _force_directed_precondition(init, iterations=100)
        
        # Optimize
        pts, score = _optimize_from_start(precond, time_budget=20.0)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
    
    # Final high-precision polish pass
    if best_pts is not None and time.time() - start_time < deadline - 5:
        # Do a final optimization with higher beta
        x0 = best_pts.flatten()
        beta_final = 100000
        
        def objective(x):
            return _soft_min_objective(x, beta_final)
        
        def gradient(x):
            eps = 1e-8
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          jac=gradient, bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < objective(x0):
                pts_final = res.x.reshape(N, 2)
                score_final = _score(pts_final)
                if score_final > best_score:
                    best_score = score_final
                    best_pts = pts_final.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _hex_lattice_init()
    
    # Final validation
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 2):
        best_pts = _hex_lattice_init()
    
    # Clamp to unit circle
    dists = np.linalg.norm(best_pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] /= dists[mask, None]
    
    return best_pts
# EVOLVE-BLOCK-END
