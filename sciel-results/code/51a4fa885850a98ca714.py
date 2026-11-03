import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _score(centers):
    """
    Calculate the score (radius) for a given set of centers.
    centers: (N, 3) array
    Returns: min(min_pair_dist/2, min_dist_to_wall)
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Calculate pairwise distances
    # Use broadcasting for efficiency, but N=100 is small enough for loops if needed.
    # Let's use scipy.spatial.distance or just numpy.
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    # Distance to wall for coordinate k is 0.5 - |c_k|
    dists_to_wall = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_wall)
    
    # Radius is min(min_pair_dist / 2, min_wall_dist)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    # Validity check: if r <= 0, it's likely overlapping or outside (though bounds checked above)
    if r <= 0:
        return 0.0
        
    return r

def _potential(centers_flat):
    """
    Potential energy to minimize.
    Minimizing this pushes spheres apart and away from walls.
    """
    centers = centers_flat.reshape(N, 3)
    # Small epsilon to prevent division by zero
    eps = 1e-12
    
    # Pairwise repulsion
    # We want to maximize min distance. Minimizing sum of 1/d^2 approximates this.
    # However, 1/d^2 can be very large. 1/d is smoother.
    # Let's use 1/d^2 for stronger repulsion.
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            total += 1.0 / (d * d + eps)
            
    # Wall repulsion
    for i in range(N):
        for k in range(3):
            d_wall = 0.5 - np.abs(centers[i, k])
            total += 1.0 / (d_wall * d_wall + eps)
            
    return total

def _potential_grad(centers_flat):
    """
    Gradient of the potential.
    """
    centers = centers_flat.reshape(N, 3)
    eps = 1e-12
    grad = np.zeros_like(centers)
    
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d2 = np.dot(diff, diff) + eps
            d = np.sqrt(d2)
            # d/dx (1/(d^2)) = -2 * (dx/dx) / d^4 = -2 * diff / d^4
            # Wait. f(d) = 1/d^2. df/dd = -2/d^3. dd/dx = diff/d.
            # df/dx = -2/d^3 * diff/d = -2 * diff / d^4
            factor = -2.0 / (d2 * d2)
            grad[i] += factor * diff
            grad[j] -= factor * diff
            
    # Walls
    for i in range(N):
        for k in range(3):
            c = centers[i, k]
            d_wall = 0.5 - np.abs(c)
            # f(d) = 1/d^2. df/dd = -2/d^3.
            # dd/dc = -sign(c)
            # df/dc = -2/d^3 * (-sign(c)) = 2*sign(c)/d^3
            # Wait, if c > 0, d = 0.5 - c, dd/dc = -1. df/dc = 2/d^3.
            # If c < 0, d = 0.5 - (-c) = 0.5 + c, dd/dc = 1. df/dc = -2/d^3.
            # So df/dc = 2 * sign(c) / d^3 ?
            # Let's check: c=0.4, d=0.1. c->0.41, d=0.09. f increases. Grad should be positive.
            # sign(c)=1. 2*1/0.001 = 2000. Positive. Correct.
            sign_c = 1.0 if c >= 0 else -1.0
            grad[i, k] += 2.0 * sign_c / (d_wall * d_wall * d_wall + eps)
            
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Add some random restarts
    rng = np.random.default_rng(42)
    
    # We will try a few initializations:
    # 1. Grid
    # 2. Grid + noise
    # 3. Random
    
    inits = [
        initial_centers,
        initial_centers + rng.normal(0, 0.01, (N, 3)),
        rng.uniform(-0.4, 0.4, (N, 3))
    ]
    
    # We can also add more random restarts if time permits
    num_restarts = 10
    for _ in range(num_restarts):
        inits.append(rng.uniform(-0.4, 0.4, (N, 3)))
        
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Clip to valid range to ensure L-BFGS-B starts in a valid-ish region
        # Actually, L-BFGS-B can handle bounds, so we don't strictly need to clip,
        # but it helps if we provide bounds.
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                _potential,
                init.flatten(),
                jac=_potential_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.status in [1, 2, 4, 5]: # 1: max iter, 2: grad, 4: ftol, 5: gtol
                candidate = res.x.reshape(N, 3)
                # Ensure candidate is strictly inside
                # L-BFGS-B respects bounds, but floating point might be on the edge
                candidate = np.clip(candidate, -0.499999, 0.499999)
                
                score = _score(candidate)
                if score > best_score:
                    best_score = score
                    best_centers = candidate
                    
        except Exception as e:
            pass
            
    # If we have time, do a final local refinement on the best center using a tighter tolerance
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(
                _potential,
                best_centers.flatten(),
                jac=_potential_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            candidate = res.x.reshape(N, 3)
            candidate = np.clip(candidate, -0.499999, 0.499999)
            score = _score(candidate)
            if score > best_score:
                best_score = score
                best_centers = candidate
        except Exception:
            pass

    if best_centers is None:
        # Fallback to grid if optimization failed
        return initial_centers
        
    return best_centers
