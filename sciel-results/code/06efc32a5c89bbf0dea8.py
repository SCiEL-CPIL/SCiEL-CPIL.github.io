import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def compute_radius(C):
        """Compute the actual packing radius for configuration C."""
        if len(C) != N:
            return 0.0
        if not np.all(np.isfinite(C)):
            return 0.0
        rad = np.linalg.norm(C, axis=1)
        min_boundary = np.min(1.0 - rad)
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return min(min_boundary, min_pair)

    def soft_violation(C, r):
        # C: (N, 2)
        # Returns sum of squared violations for radius r
        # Pairwise distance violations: (2r - d_ij)^2 if d_ij < 2r
        # Boundary violations: (|c_i| - (1-r))^2 if |c_i| > 1-r
        
        # Vectorized pairwise distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        
        # Pairwise violations
        # We only care about pairs where dist < 2r
        # max(0, 2r - dist)
        pair_viol = np.maximum(0, 2.0 * r - dist)
        pair_cost = np.sum(pair_viol ** 2)
        
        # Boundary violations
        rad = np.linalg.norm(C, axis=1)
        bound_viol = np.maximum(0, rad - (1.0 - r))
        bound_cost = np.sum(bound_viol ** 2)
        
        return pair_cost + bound_cost

    def optimize_for_r(r, init_C):
        # Flatten C for optimizer
        x0 = init_C.flatten()
        
        def obj(x):
            C = x.reshape(N, 2)
            return soft_violation(C, r)
            
        # Calculate bounds for each coordinate
        # x, y must be within [- (1-r), 1-r]
        limit = 1.0 - r
        bounds = [(-limit, limit)] * (2 * N)
        
        try:
            # Use L-BFGS-B for bounded optimization
            res = minimize(
                obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={
                    'maxiter': 500, 
                    'ftol': 1e-10, 
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return init_C, np.inf

    def get_hex_init():
        # Generate a hex lattice
        # Adjust spacing to get roughly N points
        # Area of circle = pi. Area per point in hex = s^2 * sqrt(3)/2.
        # N * s^2 * sqrt(3)/2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        k = int(1.0 / s) + 2
        pts = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    pts.append((x, y))
        
        pts = np.array(pts)
        
        # If we have more than N points, select N
        if len(pts) > N:
            # Select points closest to the center to ensure a compact start
            d = np.linalg.norm(pts, axis=1)
            idx = np.argsort(d)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            # If not enough, pad with random points inside circle
            missing = N - len(pts)
            random_pts = np.random.uniform(-1.0, 1.0, (missing, 2))
            valid = np.linalg.norm(random_pts, axis=1) <= 1.0
            random_pts = random_pts[valid]
            if len(random_pts) < missing:
                angles = np.linspace(0, 2*np.pi, missing, endpoint=False)
                extra = np.c_[angles, np.zeros(missing)]
                pass 
            pts = np.vstack([pts, random_pts])
            
        return pts

    def constraint_push(C, r, iterations=50):
        """Apply constraint pushing to fix violations."""
        C = C.copy()
        for _ in range(iterations):
            # Fix boundary violations
            rad = np.linalg.norm(C, axis=1)
            max_rad = 1.0 - r
            for i in range(N):
                if rad[i] > max_rad:
                    scale = max_rad / rad[i] if rad[i] > 1e-10 else 1.0
                    C[i] *= scale
            
            # Fix pairwise violations
            changed = True
            while changed:
                changed = False
                diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                
                # Find pairs that are too close
                mask = dist < 2.0 * r
                for i in range(N):
                    for j in range(i + 1, N):
                        if mask[i, j]:
                            d = dist[i, j]
                            if d < 1e-10:
                                # Random perturbation
                                C[i] += 0.01 * np.random.randn(2)
                                C[j] -= 0.01 * np.random.randn(2)
                            else:
                                # Push apart
                                direction = (C[i] - C[j]) / d
                                push = (2.0 * r - d) / 2.0 + 1e-6
                                C[i] += push * direction
                                C[j] -= push * direction
                            changed = True
            
            # Re-apply boundary constraints
            rad = np.linalg.norm(C, axis=1)
            max_rad = 1.0 - r
            for i in range(N):
                if rad[i] > max_rad:
                    scale = max_rad / rad[i] if rad[i] > 1e-10 else 1.0
                    C[i] *= scale
        
        return C

    # Initial configuration
    C_best = get_hex_init()
    
    # Binary search for max radius
    r_low = 0.1
    r_high = 0.14
    best_r = r_low
    best_C = C_best.copy()
    
    # We will perform a number of binary search steps
    # Each step runs an optimization
    num_steps = 20
    
    for i in range(num_steps):
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        C_opt, val = optimize_for_r(r_mid, C_best)
        
        # Apply constraint pushing to fix any remaining violations
        C_opt = constraint_push(C_opt, r_mid, iterations=30)
        
        # Check if feasible
        # Recompute violation after pushing
        val_after = soft_violation(C_opt, r_mid)
        
        if val_after < 1e-6:
            best_r = r_mid
            C_best = C_opt
            best_C = C_opt.copy()
            r_low = r_mid
        else:
            r_high = r_mid
    
    # Final polish: try to improve the best configuration
    if time.time() < deadline - 5.0:
        # Try a few local optimizations from the best configuration
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            C_try = best_C + 0.001 * np.random.randn(N, 2)
            C_try = constraint_push(C_try, best_r, iterations=20)
            r_try = compute_radius(C_try)
            if r_try > best_r:
                best_r = r_try
                best_C = C_try
    
    # Final check: ensure the returned configuration is valid
    final_r = compute_radius(best_C)
    if final_r <= 0:
        # Fallback to hex init
        return get_hex_init()
    
    return best_C
