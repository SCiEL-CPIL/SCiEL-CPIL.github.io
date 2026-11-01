"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        # Distance to boundaries
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Set diagonal to inf
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Minimum distance to any other circle
        min_dist = np.min(dist_matrix, axis=1)
        
        # Radius is limited by boundary and half the min distance to other circles
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params):
        """Smooth objective using soft-min approximation."""
        x = params[0::2]
        y = params[1::2]
        
        # Distance to boundaries
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Set diagonal to a large value
        np.fill_diagonal(dist_matrix, 1e10)
        
        # Soft-min of distances to other circles
        # Use log-sum-exp for numerical stability
        alpha = 100.0  # sharpness parameter
        min_dist_soft = -np.log(np.exp(-alpha * dist_matrix).sum(axis=1) / (N - 1)) / alpha
        
        # Soft-min with boundary
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        min_boundary_soft = -np.log(np.exp(-alpha * boundary_terms).sum(axis=1) / 4.0) / alpha
        
        # Combined soft-min
        all_terms = np.concatenate([min_dist_soft / 2.0, min_boundary_soft], axis=0)
        # Actually, we need min of (dist_boundary, min_dist/2)
        # Let's just use the hard min for now but with a smooth transition
        
        # Simpler: use the hard min but ensure differentiability via subgradient
        # For now, stick with hard min but use better optimization
        
        radii = np.minimum(dist_boundary, min_dist_soft / 2.0)
        radii = np.maximum(radii, 0)
        
        return -np.sum(radii)

    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))  # 6 for N=32
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx] = (i + 0.5) / side
            y0[idx] = (j + 0.5) / side
            idx += 1
    if idx < N:
        for k in range(idx, N):
            x0[k] = 0.5
            y0[k] = 0.5

    params0 = np.zeros(2 * N)
    params0[0::2] = x0
    params0[1::2] = y0

    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Phase 1: Powell optimization from grid start
    try:
        res = minimize(
            objective,
            params0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-12}
        )
        if res.fun < best_obj:
            best_params = res.x
            best_obj = res.fun
    except Exception:
        pass

    # Phase 2: Try multiple random restarts with better initialization
    np.random.seed(42)
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate random initial configuration with jittered grid
        x_rand = (np.arange(side) + 0.5 + np.random.randn(side) * 0.05) / side
        y_rand = (np.arange(side) + 0.5 + np.random.randn(side) * 0.05) / side
        
        params_rand = np.zeros(2 * N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                params_rand[2*idx] = np.clip(x_rand[i], 0.02, 0.98)
                params_rand[2*idx+1] = np.clip(y_rand[j], 0.02, 0.98)
                idx += 1
        if idx < N:
            for k in range(idx, N):
                params_rand[2*k] = 0.5
                params_rand[2*k+1] = 0.5
        
        try:
            res = minimize(
                objective,
                params_rand,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 3: Local refinement with small perturbations
    if time.time() < deadline - 15:
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        radii_best = compute_radii(x_best, y_best)
        sum_best = np.sum(radii_best)
        
        # Try moving each circle in small steps
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                # Try multiple step sizes
                for step_size in [0.005, 0.002, 0.001, 0.0005]:
                    for dx_try in [-step_size, 0, step_size]:
                        for dy_try in [-step_size, 0, step_size]:
                            if dx_try == 0 and dy_try == 0:
                                continue
                            x_try = x_best.copy()
                            y_try = y_best.copy()
                            x_try[i] += dx_try
                            y_try[i] += dy_try
                            
                            # Check bounds
                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue
                            
                            radii_try = compute_radii(x_try, y_try)
                            sum_try = np.sum(radii_try)
                            
                            if sum_try > sum_best + 1e-14:
                                x_best = x_try
                                y_best = y_try
                                radii_best = radii_try
                                sum_best = sum_try
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        
        # Update best_params
        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best
        best_obj = objective(best_params)

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective,
                best_params,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 10000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)
    
    # Shrink radii slightly to ensure strict disjointness (floating point safety)
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
