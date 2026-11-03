"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 5.0  # seconds before the 150s kill to stop


def _score(pts):
    """Compute the common radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 0.0
    # min distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(dist_boundary)
    # min pairwise distance / 2
    # Use broadcasting for pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _sa_optimize(pts, n_iter=50000, t0=0.5, t1=0.01, seed=42):
    """Simulated annealing to maximize the common radius."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    
    for i in range(n_iter):
        t = t0 * (t1 / t0) ** (i / max(n_iter - 1, 1))
        
        # Pick a random point to move
        idx = rng.integers(0, n)
        old_pos = current_pts[idx].copy()
        
        # Random perturbation
        step = t * 0.1
        new_pos = old_pos + rng.normal(0, step, size=2)
        
        # Keep inside unit circle (soft constraint)
        norm = np.linalg.norm(new_pos)
        if norm > 1.0:
            new_pos = new_pos / norm * 0.999
        
        current_pts[idx] = new_pos
        new_score = _score(current_pts)
        
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / t):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            current_pts[idx] = old_pos
            
    return best_pts, best_score


def _local_optimize(pts, n_iter=10000, seed=42):
    """Local optimization using coordinate descent with adaptive steps."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    
    step_size = 0.01
    
    for i in range(n_iter):
        # Pick a random point
        idx = rng.integers(0, n)
        old_pos = current_pts[idx].copy()
        
        # Try small moves in random directions
        improved = False
        for _ in range(5):
            direction = rng.normal(0, 1, size=2)
            direction /= (np.linalg.norm(direction) + 1e-10)
            
            # Try multiple step sizes
            for s in [step_size, step_size * 0.5, step_size * 2]:
                new_pos = old_pos + direction * s
                norm = np.linalg.norm(new_pos)
                if norm > 1.0:
                    new_pos = new_pos / norm * 0.999
                
                current_pts[idx] = new_pos
                new_score = _score(current_pts)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_pts[idx] = old_pos
            
            if improved:
                break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
            # Adapt step size
            step_size = max(0.001, step_size * 0.999)
        else:
            step_size = min(0.05, step_size * 1.01)
            
    return best_pts, best_score


def _init_hex(r_est):
    """Generate points on a hexagonal lattice scaled to approximate radius r_est."""
    spacing = 2 * r_est
    points = []
    k = int(1.0 / spacing) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r_est:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Select N points closest to center
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii, kind='stable')[:N]
        return points[idx]
    else:
        # Fallback: random points
        rng = np.random.default_rng(0)
        pts = []
        while len(pts) < N:
            p = rng.uniform(-1, 1, size=2)
            if np.linalg.norm(p) <= 1.0 - r_est:
                pts.append(p)
        return np.array(pts)


def _init_random(r_est, seed=42):
    """Generate random points inside the unit circle."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < N:
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 1.0 - r_est)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pts.append([x, y])
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initial estimate of radius
    r_est = 0.12
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_init_hex(r_est))
    inits.append(_init_random(r_est, seed=42))
    inits.append(_init_random(r_est, seed=123))
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Simulated annealing
        pts, score = _sa_optimize(init, n_iter=30000, t0=0.3, t1=0.005, seed=42)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Local optimization
        if time.time() > deadline:
            break
        pts, score = _local_optimize(pts, n_iter=5000, seed=42)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Additional rounds with different seeds if time permits
    if best_pts is not None and time.time() < deadline:
        for seed in [100, 200, 300]:
            if time.time() > deadline:
                break
            pts, score = _sa_optimize(best_pts + np.random.default_rng(seed).normal(0, 0.01, size=best_pts.shape), 
                                       n_iter=20000, t0=0.2, t1=0.003, seed=seed)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            pts, score = _local_optimize(pts, n_iter=3000, seed=seed)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: return hex lattice
        best_pts = _init_hex(r_est)
    
    # Final validation
    score = _score(best_pts)
    if not np.isfinite(score) or score <= 0:
        best_pts = _init_hex(r_est)
    
    return best_pts
# EVOLVE-BLOCK-END
