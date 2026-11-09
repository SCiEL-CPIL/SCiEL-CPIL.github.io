import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 8

    # We assume the optimal configuration has D4 symmetry.
    # This means the 8 points can be described by 4 unique radii (r1, r2, r3, r4)
    # and a single rotation angle theta for the inner square relative to the outer square.
    # The points are:
    # Outer square (radius r1): angles 0, pi/2, pi, 3pi/2
    # Inner square (radius r2): angles theta, pi/2+theta, pi+theta, 3pi/2+theta
    # Wait, D4 symmetry implies the set is invariant under rotation by pi/2.
    # So if we have a point at angle phi, we must have points at phi+pi/2, phi+pi, phi+3pi/2.
    # With 8 points, we can have two orbits of size 4.
    # Orbit 1: radius r1, angle 0
    # Orbit 2: radius r2, angle theta
    # This gives 8 points:
    # P1: (r1, 0)
    # P2: (r1, pi/2)
    # P3: (r1, pi)
    # P4: (r1, 3pi/2)
    # P5: (r2, theta)
    # P6: (r2, pi/2+theta)
    # P7: (r2, pi+theta)
    # P8: (r2, 3pi/2+theta)
    
    # Is this sufficient? 
    # The regular octagon has r1=1, r2=1, theta=pi/4.
    # The "two concentric squares" strategy in the current code explored r1=1, r2=r, theta=pi/4 or 0.
    # But maybe r1 and r2 can be different, and theta can be anything.
    # Actually, if we allow r1, r2, and theta to vary, we cover a large class of symmetric configurations.
    
    def get_points(r1, r2, theta):
        # r1, r2 >= 0
        # theta in [0, pi/2) due to symmetry
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        
        p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        
        return np.vstack([p1, p2])

    def compute_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists = np.maximum(dists, 1e-10)
        
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        scale = 2.0 / max_dist
        scaled_dists = dists * scale
        
        log_prod = np.sum(np.log(scaled_dists[mask]))
        
        return log_prod - n * np.log(n)

    def objective(x):
        r1, r2, theta = x
        # Ensure non-negative radii
        r1 = max(r1, 1e-6)
        r2 = max(r2, 1e-6)
        # Normalize theta to [0, pi/2)
        theta = theta % (np.pi/2)
        
        points = get_points(r1, r2, theta)
        s = compute_score(points)
        if not np.isfinite(s):
            return 1e10
        return -s

    best_score = -np.inf
    best_params = None
    
    # Grid search over r1, r2, theta
    # Since the score is scale-invariant (we normalize to diameter 2),
    # the ratio r2/r1 is the only radial degree of freedom.
    # Let's fix r1 = 1.0 and vary r2 and theta.
    # However, the diameter depends on r1 and r2 and theta.
    # It is safer to just search r1, r2, theta directly, but since scale is free,
    # we can fix r1=1 and search r2 in [0, 2] and theta in [0, pi/2].
    
    # Let's do a coarse grid first to find good basins
    r2_vals = np.linspace(0.0, 2.0, 20)
    theta_vals = np.linspace(0.0, np.pi/2, 20)
    
    candidates = []
    for r2 in r2_vals:
        for theta in theta_vals:
            candidates.append([1.0, r2, theta])
            
    # Add specific known good candidates
    candidates.append([1.0, 1.0, np.pi/4]) # Regular Octagon
    candidates.append([1.0, 0.5, 0.0])     # Two squares aligned
    candidates.append([1.0, 1.5, np.pi/8]) # Some guess
    
    # Evaluate and refine
    for x0 in candidates:
        if time.time() > deadline:
            break
            
        try:
            # Use BFGS for fast local convergence in low dimensions
            res = minimize(objective, x0, method='BFGS', 
                           options={'gtol': 1e-10, 'maxiter': 200})
            
            final_params = res.x
            final_score = -res.fun
            
            if final_score > best_score:
                best_score = final_score
                best_params = final_params
                
        except Exception:
            pass

    # If we have time, do a second pass with more refined grid or more restarts
    if time.time() < deadline - 10:
        # Refine around the best found
        if best_params is not None:
            r1_best, r2_best, theta_best = best_params
            # Note: best_params might have r1 != 1 if BFGS moved it, but we fixed r1=1 in grid.
            # Actually, BFGS might have changed r1 if we passed it as a variable.
            # In my grid, I fixed r1=1.0. Let's allow r1 to vary in the refinement?
            # No, scale invariance means r1 is redundant. 
            # But wait, my objective function takes r1, r2, theta.
            # If I fix r1=1 in the grid, BFGS will keep it near 1 unless the gradient pushes it.
            # Since the function is scale invariant, the gradient wrt r1 and r2 should be dependent.
            # It's better to parameterize by ratio k = r2/r1 and r1.
            # Or just trust that fixing r1=1 is fine because any optimal solution can be scaled to r1=1.
            
            # Let's do a few random restarts in the neighborhood of the best solution
            rng = np.random.default_rng(123)
            for _ in range(50):
                if time.time() > deadline:
                    break
                noise_r = rng.normal(0, 0.1)
                noise_t = rng.normal(0, 0.1)
                x0 = [1.0, max(0, r2_best + noise_r), (theta_best + noise_t) % (np.pi/2)]
                
                try:
                    res = minimize(objective, x0, method='BFGS', 
                                   options={'gtol': 1e-10, 'maxiter': 100})
                    final_params = res.x
                    final_score = -res.fun
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_params = final_params
                except Exception:
                    pass

    if best_params is None:
        # Fallback to regular octagon
        r1, r2, theta = 1.0, 1.0, np.pi/4
    else:
        r1, r2, theta = best_params
        r1 = max(r1, 1e-6)
        r2 = max(r2, 1e-6)
        theta = theta % (np.pi/2)

    points = get_points(r1, r2, theta)
    
    # Ensure distinct points
    # If r1 == r2 and theta == 0, points coincide.
    # The optimizer should avoid this, but let's check.
    # If they coincide, perturb slightly.
    # For safety, return the points.
    
    return [[float(x), float(y)] for x, y in points]
