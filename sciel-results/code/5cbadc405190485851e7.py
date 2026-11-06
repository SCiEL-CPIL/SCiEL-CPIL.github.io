import numpy as np
from scipy.optimize import differential_evolution

N = 50

def construct():
    # Try several row patterns for hexagonal-like packing
    patterns = [
        [7, 7, 7, 7, 7, 7, 7, 1],  # 8 rows
        [7, 6, 7, 6, 7, 6, 7, 4],  # 8 rows
        [8, 7, 7, 7, 7, 7, 7],     # 7 rows
        [7, 7, 7, 7, 7, 7, 8],     # 7 rows
        [6, 7, 7, 7, 7, 7, 6, 3],  # 8 rows
        [7, 7, 7, 7, 7, 7, 7, 1],  # 8 rows
        [5, 6, 7, 7, 7, 7, 6, 5],  # 8 rows
        [6, 7, 7, 7, 7, 7, 7, 2],  # 8 rows
    ]
    
    best_radius = 0
    best_centers = None
    
    for pattern in patterns:
        if sum(pattern) != N:
            continue
        
        n_rows = len(pattern)
        
        def compute_centers(dx, dy_scale, x_shift, y_shift, *args):
            """Compute centers for given parameters."""
            dy = dx * dy_scale  # dy_scale = sqrt(3)/2 for hex
            r = dx / 2
            
            centers = []
            for i, n in enumerate(pattern):
                # Row i: n circles
                # Center the row at x = x_shift
                # Positions: x_shift + (j - (n-1)/2) * dx
                for j in range(n):
                    x = x_shift + (j - (n - 1) / 2.0) * dx
                    y = y_shift + i * dy
                    centers.append([x, y])
            
            centers = np.array(centers)
            
            # Check validity
            if not np.all(np.isfinite(centers)):
                return -1
            
            # Check inside square
            if np.any(centers[:, 0] < -0.5 + 1e-10) or np.any(centers[:, 0] > 0.5 - 1e-10):
                return -1
            if np.any(centers[:, 1] < -0.5 + 1e-10) or np.any(centers[:, 1] > 0.5 - 1e-10):
                return -1
            
            # Check non-overlap
            min_dist = 1e10
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist < 2 * r - 1e-10:
                return -1
            
            # Score: the radius
            boundary_dist = min(0.5 - np.max(np.abs(centers[:, 0])), 
                               0.5 - np.max(np.abs(centers[:, 1])))
            actual_r = min(min_dist / 2, boundary_dist)
            
            return actual_r
        
        # Bounds for parameters
        # dx: spacing, roughly 1/7 to 1/5 for these patterns
        # dy_scale: sqrt(3)/2 ≈ 0.866 for hex, allow some variation
        # x_shift, y_shift: small adjustments
        
        bounds = [
            (0.12, 0.2),      # dx
            (0.8, 0.95),      # dy_scale
            (-0.05, 0.05),    # x_shift
            (-0.05, 0.05),    # y_shift
        ]
        
        try:
            result = differential_evolution(
                lambda *args: -compute_centers(*args),
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-8,
                polish=False
            )
            
            if result.fun < 0:
                r_candidate = -result.fun
                if r_candidate > best_radius:
                    best_radius = r_candidate
                    best_centers = compute_centers(*result.x)
                    # Recompute to get actual centers
                    dx, dy_scale, x_shift, y_shift = result.x
                    dy = dx * dy_scale
                    centers = []
                    for i, n in enumerate(pattern):
                        for j in range(n):
                            x = x_shift + (j - (n - 1) / 2.0) * dx
                            y = y_shift + i * dy
                            centers.append([x, y])
                    best_centers = np.array(centers)
        except Exception:
            pass
    
    if best_centers is not None and best_radius > 0:
        return best_centers
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
