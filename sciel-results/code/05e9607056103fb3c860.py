import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 8
    
    # Objective function: Maximize log(product of distances) - n*log(n)
    # The score is Delta / n^n where Delta is product of distances after scaling to diameter 2.
    # Let d_ij be distances.
    # Scaled distance d'_ij = d_ij * (2 / D), where D is max distance.
    # Product P' = prod(d'_ij) = prod(d_ij) * (2/D)^(n(n-1))
    # Score = P' / n^n
    # log(Score) = sum(log(d_ij)) + n(n-1)*(log(2) - log(D)) - n*log(n)
    # We maximize log(Score).
    
    def compute_log_score(points):
        # points: (8, 2) array
        # Compute pairwise distances
        # Use broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal (self-distances are 0)
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-10):
            return -np.inf
            
        D = np.max(valid_dists)
        
        if D < 1e-10:
            return -np.inf
            
        # log(Score) = sum(log(valid_dists)) + n(n-1)*(log(2) - log(D)) - n*log(n)
        # Note: The problem statement says "product over ordered pairs".
        # Usually "product of distances" in these geometric contexts implies unordered pairs (i<j).
        # Let's check the regular octagon score.
        # Regular octagon side length s = 2*sin(pi/8). Diameter D = 2*sin(4*pi/8)=2.
        # If ordered pairs: n(n-1) terms.
        # If unordered pairs: n(n-1)/2 terms.
        # The prompt says: "Delta is the product over ordered pairs of distances".
        # So Delta = prod_{i != j} d_ij.
        # This means there are n(n-1) terms.
        
        log_prod = np.sum(np.log(valid_dists))
        
        # Scaling factor
        # d'_ij = d_ij * 2 / D
        # prod(d'_ij) = prod(d_ij) * (2/D)^(n(n-1))
        log_scaled_prod = log_prod + n * (n - 1) * (np.log(2.0) - np.log(D))
        
        # Score = log_scaled_prod - n * log(n)
        return log_scaled_prod - n * np.log(n)

    def objective(x):
        # x is a flat array of 16 coordinates
        # We constrain points to be within the unit disk? 
        # Actually, scale invariance means we can fix the diameter or just let the optimizer handle it.
        # However, L-BFGS-B needs bounds. 
        # We can bound coordinates to [-1.5, 1.5] to cover diameter up to ~4.2, 
        # but the optimal diameter is 2. 
        # A better constraint: keep points in the unit circle? No, the optimal set has diameter 2, 
        # so it fits in a circle of radius 1. 
        # If we bound x to [-1, 1], we might cut off solutions that temporarily expand? 
        # But scale invariance allows us to normalize. 
        # Let's just bound to [-2, 2] to be safe, and rely on the objective to pull them in.
        
        pts = x.reshape((n, 2))
        
        # Optional: Check for distinct points
        # The objective handles it via log(0) -> -inf
        
        s = compute_log_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Initial guesses
    # 1. Regular Octagon (Score 1.0)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    x_oct = pts_oct.flatten()
    
    # 2. "Two Squares" configuration (D4 symmetry but potentially different radii)
    # This is a known candidate that might be better or worse.
    # Let's generate a few diverse starting points.
    
    starting_points = [x_oct]
    
    # Add perturbations to the regular octagon to break symmetry
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random perturbation in the unit disk
        # Start near octagon but jitter
        jitter = rng.normal(0, 0.2, size=(n, 2))
        pts_pert = pts_oct + jitter
        # Normalize to roughly unit diameter to help convergence
        # Find max distance
        diff = pts_pert[:, np.newaxis, :] - pts_pert[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            pts_pert = pts_pert * (2.0 / D)
        starting_points.append(pts_pert.flatten())
        
    # Add some random points in the unit disk
    for _ in range(20):
        # Random angles and radii
        r = rng.uniform(0.5, 1.0, size=n)
        a = rng.uniform(0, 2*np.pi, size=n)
        pts_rand = np.column_stack([r * np.cos(a), r * np.sin(a)])
        # Normalize
        diff = pts_rand[:, np.newaxis, :] - pts_rand[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            pts_rand = pts_rand * (2.0 / D)
        starting_points.append(pts_rand.flatten())

    best_score = -np.inf
    best_points = pts_oct
    
    # Evaluate initial points
    for x0 in starting_points:
        if time.time() > deadline:
            break
        s = compute_log_score(x0.reshape((n, 2)))
        if s > best_score:
            best_score = s
            best_points = x0.reshape((n, 2))

    # Optimize from best starting points
    # Sort starting points by initial score, optimize the top ones
    scored_starts = []
    for x0 in starting_points:
        s = compute_log_score(x0.reshape((n, 2)))
        scored_starts.append((s, x0))
    scored_starts.sort(reverse=True)
    
    # Try optimizing the top 10 starting points
    for s, x0 in scored_starts[:10]:
        if time.time() > deadline:
            break
            
        try:
            # L-BFGS-B with bounds
            # Bounds: [-2, 2] for each coordinate
            bounds = [(-2.0, 2.0)] * (2 * n)
            
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.status in [1, 2, 3, 4, 5, 6]:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_points = res.x.reshape((n, 2))
                    
        except Exception:
            pass

    # If we have time, do a second pass with more aggressive restarts or different method
    if time.time() < deadline - 5:
        # Try a few more random restarts near the best solution found
        for _ in range(50):
            if time.time() > deadline:
                break
            
            noise = rng.normal(0, 0.05, size=(n, 2))
            x0 = best_points + noise
            # Normalize
            diff = x0[:, np.newaxis, :] - x0[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            D = np.max(dists[~np.eye(n, dtype=bool)])
            if D > 0:
                x0 = x0 * (2.0 / D)
            
            try:
                bounds = [(-2.0, 2.0)] * (2 * n)
                res = minimize(
                    objective, 
                    x0.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if res.success or res.status in [1, 2, 3, 4, 5, 6]:
                    final_score = -res.fun
                    if final_score > best_score:
                        best_score = final_score
                        best_points = res.x.reshape((n, 2))
                        
            except Exception:
                pass

    # Final cleanup: Ensure points are distinct and within bounds
    # The problem requires |coordinate| < 10^6. Our points are ~1.
    # Ensure distinctness
    # If any two points are too close, perturb them slightly.
    # But the optimizer should have separated them.
    
    # Return in required format
    return [[float(x), float(y)] for x, y in best_points]
