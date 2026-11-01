import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp with better numerical stability
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # For soft max: log-sum-exp(tau * (sq_dists - m_max)) / tau + m_max
    shifted_max = tau * (sq_dists - m_max)
    max_logsumexp = np.log(np.sum(np.exp(shifted_max)))
    soft_max = m_max + max_logsumexp / tau
    
    # For soft min: -log-sum-exp(-tau * (sq_dists - m_min)) / tau + m_min
    shifted_min = -tau * (sq_dists - m_min)
    min_logsumexp = np.log(np.sum(np.exp(shifted_min)))
    soft_min = m_min - min_logsumexp / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def objective_exact(x):
    """
    Exact (nonsmooth) log-ratio objective for final polishing.
    Uses subgradient-style handling via direct max/min.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def fibonacci_sphere(n_points):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def exact_polish(points, max_iters=200, tol=1e-12):
    """
    Coordinate descent polish: adjust points to reduce max distance 
    and increase min distance directly.
    """
    pts = points.copy()
    best_score = raw_score(pts)
    
    for it in range(max_iters):
        sq_dists = get_pairwise_sq_dists(pts)
        max_idx = np.argmax(sq_dists)
        min_idx = np.argmin(sq_dists)
        
        # Get the pairs
        iu = np.triu_indices(N, k=1)
        max_pair = (iu[0][max_idx], iu[1][max_idx])
        min_pair = (iu[0][min_idx], iu[1][min_idx])
        
        # Try to reduce max distance by moving those points closer
        # and increase min distance by moving those points apart
        step = 0.01
        improved = False
        
        # Move max-distance pair closer
        p1, p2 = max_pair
        mid = (pts[p1] + pts[p2]) / 2
        diff = pts[p1] - pts[p2]
        norm = np.linalg.norm(diff)
        if norm > 1e-15:
            direction = diff / norm
            # Try small moves
            for frac in [step, step*0.5, step*0.1]:
                trial = pts.copy()
                trial[p1] -= direction * frac * 0.5
                trial[p2] += direction * frac * 0.5
                s = raw_score(trial)
                if s < best_score * (1 - 1e-6):
                    best_score = s
                    pts = trial
                    improved = True
                    break
        
        # Move min-distance pair apart
        p3, p4 = min_pair
        diff2 = pts[p3] - pts[p4]
        norm2 = np.linalg.norm(diff2)
        if norm2 > 1e-15:
            direction2 = diff2 / norm2
            for frac in [step, step*0.5, step*0.1]:
                trial = pts.copy()
                trial[p3] += direction2 * frac * 0.5
                trial[p4] -= direction2 * frac * 0.5
                s = raw_score(trial)
                if s < best_score * (1 - 1e-6):
                    best_score = s
                    pts = trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return pts, best_score

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate a focused set of high-quality initial candidates
    candidates = []
    
    # 1. Fibonacci sphere (well-distributed on sphere)
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Perturbed Fibonacci (multiple scales)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        perturbed = fib_pts + rng.normal(size=(N, D)) * scale
        candidates.append(perturbed.flatten())
    
    # 3. Random points on sphere (well-normalized)
    for seed in range(8):
        r = np.random.default_rng(seed + 1000)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
    
    # 4. Random points in ball (varied radii)
    for seed in range(5):
        r = np.random.default_rng(seed + 2000)
        pts = r.normal(size=(N, D)) * 1.5
        # Add some structure: cluster around center
        pts[:7] *= 0.7
        pts[7:] *= 1.3
        candidates.append(pts.flatten())
    
    # 5. Two-shell configuration: 7 inner, 7 outer
    for seed in range(3):
        r = np.random.default_rng(seed + 3000)
        inner = r.normal(size=(7, D)) * 0.5
        outer = r.normal(size=(7, D)) * 1.5
        pts = np.vstack([inner, outer])
        candidates.append(pts.flatten())
    
    # Optimization parameters
    # Progressive tau schedule
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
    maxiter_schedule = [200, 300, 400, 500, 600]
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:
            break
        
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 10:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Evaluate current smooth-optimized result
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # Final exact polish on the best candidate found
    if best_points is not None and time.time() - start_time < timeout - 5:
        polished_pts, polished_score = exact_polish(best_points, max_iters=100)
        if polished_score < best_score:
            best_score = polished_score
            best_points = polished_pts
    
    # If no good result, try one more round with SLSQP on best candidate
    if best_points is not None and time.time() - start_time < timeout - 3:
        try:
            obj_final = partial(objective_smooth, tau=5000.0)
            res_polish = minimize(
                obj_final,
                best_points.flatten(),
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-15, 'maxls': 40}
            )
            if res_polish.x is not None and np.isfinite(res_polish.fun):
                pts_polished = res_polish.x.reshape(N, D)
                s = raw_score(pts_polished)
                if s < best_score:
                    best_score = s
                    best_points = pts_polished
        except Exception:
            pass
    
    # Fallback: if still nothing, use best raw candidate
    if best_points is None:
        for init_x in candidates:
            pts = init_x.reshape(N, D)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts.copy()
        
        if best_points is None:
            best_points = fibonacci_sphere(N)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
