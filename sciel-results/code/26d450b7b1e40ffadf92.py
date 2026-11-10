"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import cos, sin, pi, log, sqrt

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Diameter
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    # Scale to diameter 2
    scale = 2.0 / diam
    dists_scaled = dists * scale
    # Product over ordered pairs (i != j)
    # Set diagonal to 1 to avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score(points):
    """Negative log of the score for minimization."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    # score = exp(log_prod) / n^n
    # log(score) = log_prod - n*log(n)
    # neg_log_score = -(log_prod - n*log(n)) = n*log(n) - log_prod
    return n * log(n) - log_prod

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with regular octagon
    pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Try multiple initializations and optimization passes
    n = 8
    rng = np.random.RandomState(42)
    
    # Phase 1: Try various structured initializations
    initializations = []
    
    # Regular octagon (already have)
    initializations.append(pts.copy())
    
    # Squares with different configurations
    # Two squares, one rotated
    for angle_offset in [0, pi/4, pi/8, pi/16]:
        sq1 = np.array([[cos(2*pi*k/4 + angle_offset), sin(2*pi*k/4 + angle_offset)] for k in range(4)])
        sq2 = np.array([[1.5*cos(2*pi*k/4), 1.5*sin(2*pi*k/4)] for k in range(4)])
        init = np.vstack([sq1, sq2])
        initializations.append(init)
    
    # Random initializations
    for _ in range(20):
        init = rng.randn(n, 2) * 2.0
        initializations.append(init)
    
    # Phase 2: Optimize each initialization
    from scipy.optimize import minimize
    
    def objective(x):
        pts_x = x.reshape(n, 2)
        return _neg_log_score(pts_x)
    
    def constraint_diameter(x):
        pts_x = x.reshape(n, 2)
        diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return 2.0 - np.max(dists)
    
    constraints = [{'type': 'ineq', 'fun': constraint_diameter}]
    
    for i, init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(objective, init.flatten(), method='SLSQP', 
                            constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            if result.success or result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Polish with smaller perturbations around best solution
    if time.time() < deadline - 10:
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            # Small random perturbation
            noise = rng.randn(n, 2) * 0.01
            perturbed = best_pts + noise
            try:
                result = minimize(objective, perturbed.flatten(), method='SLSQP',
                                constraints=constraints, options={'maxiter': 100, 'ftol': 1e-14})
                if result.success or result.fun < 1e6:
                    opt_pts = result.x.reshape(n, 2)
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Phase 4: Final polish with very small steps
    if time.time() < deadline - 2:
        for step_size in [0.005, 0.001, 0.0005]:
            if time.time() > deadline - 1:
                break
            for trial in range(30):
                if time.time() > deadline - 1:
                    break
                noise = rng.randn(n, 2) * step_size
                perturbed = best_pts + noise
                try:
                    result = minimize(objective, perturbed.flatten(), method='SLSQP',
                                    constraints=constraints, options={'maxiter': 50, 'ftol': 1e-15})
                    if result.success or result.fun < 1e6:
                        opt_pts = result.x.reshape(n, 2)
                        sc = _score(opt_pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = opt_pts.copy()
                except Exception:
                    pass
    
    return best_pts.tolist()
# EVOLVE-BLOCK-END
