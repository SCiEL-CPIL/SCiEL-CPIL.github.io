import numpy as np
import time
from scipy.optimize import minimize

N = 50

def soft_min(values, beta):
    """
    Computes the soft-min of an array of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    """
    # Clip values to prevent overflow in exp
    # If values are large positive, exp(-beta * x) is small.
    # If values are large negative, exp(-beta * x) is large.
    # We want to minimize the sum of exps to maximize the min.
    # Actually, soft_min is an approximation of min.
    # min(x) ~ -1/beta * log(sum(exp(-beta * x)))
    
    # To be numerically stable:
    # Let m = min(values)
    # sum(exp(-beta * (x - m))) = exp(beta * m) * sum(exp(-beta * x))
    # log(sum(exp(-beta * x))) = -beta * m + log(sum(exp(-beta * (x - m))))
    # -1/beta * log(...) = m - 1/beta * log(sum(exp(-beta * (x - m))))
    
    m = np.min(values)
    shifted = values - m
    sum_exp = np.sum(np.exp(-beta * shifted))
    return m - (1.0 / beta) * np.log(sum_exp)

def get_radius(points):
    """
    Computes the actual minimum radius for a given configuration.
    """
    if len(points) < 2:
        return 1.0 - np.linalg.norm(points, axis=1).min()
        
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.min(radii)
    
    return min(min_pairwise, min_boundary)

def objective(points_flat, beta):
    """
    Objective function for minimization.
    Returns -soft_min(distances).
    """
    points = points_flat.reshape(N, 2)
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 1e10
        
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # We need the N(N-1)/2 unique distances
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - radii
    
    # Combine
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # If any distance is negative (overlap or out of bounds), the min is negative.
    # Soft min will handle it.
    
    sm = soft_min(all_dists, beta)
    return -sm

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_radius = 0.0
    
    # Initializations
    inits = []
    
    # 1. Hexagonal Lattice
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        # Sort by radius to take central ones
        points = points[np.argsort(radius, kind="stable")[:N]]
        inits.append(points)
    elif len(points) > 0:
        # Pad with random points if not enough
        missing = N - len(points)
        pad = np.random.uniform(-0.5, 0.5, (missing, 2))
        points = np.vstack([points, pad])
        inits.append(points)
        
    # 2. Random starts
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.uniform(-0.8, 0.8, (N, 2))
        inits.append(pts)
        
    # 3. Ring structure
    pts = np.zeros((1, 2))
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts = np.vstack([pts, [0.3 * np.cos(angle), 0.3 * np.sin(angle)]])
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts = np.vstack([pts, [0.6 * np.cos(angle), 0.6 * np.sin(angle)]])
    rem = N - len(pts)
    for i in range(rem):
        angle = 2 * np.pi * i / rem
        pts = np.vstack([pts, [0.85 * np.cos(angle), 0.85 * np.sin(angle)]])
    inits.append(pts)
    
    # Optimization
    for init in inits:
        if time.time() > deadline:
            break
            
        points = init.copy()
        
        # Ensure initial validity (project inside)
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 0.95
        if np.any(mask_out):
            points[mask_out] *= (0.95 / radii[mask_out])[:, np.newaxis]
            
        current_radius = get_radius(points)
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Run L-BFGS-B
        # Start with moderate beta
        beta = 50.0
        
        # Flatten points for scipy
        x0 = points.flatten()
        
        try:
            # Use L-BFGS-B
            # Bounds: x, y in [-1, 1]
            bounds = [(-1.0, 1.0)] * (2 * N)
            
            res = minimize(
                objective, 
                x0, 
                args=(beta,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective(x0, beta):
                new_points = res.x.reshape(N, 2)
                new_radius = get_radius(new_points)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
                    
                    # Polish with higher beta
                    beta_high = 100.0
                    x0_high = new_points.flatten()
                    res2 = minimize(
                        objective, 
                        x0_high, 
                        args=(beta_high,), 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    if res2.success or res2.fun < res.fun:
                        final_points = res2.x.reshape(N, 2)
                        final_radius = get_radius(final_points)
                        
                        if final_radius > best_radius:
                            best_radius = final_radius
                            best_points = final_points.copy()
                            
        except Exception:
            pass

    if best_points is None:
        # Fallback: Grid
        pts = []
        step = 0.2
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * step
                y = j * step
                if x*x + y*y <= 0.81:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        elif len(pts) < N:
            pad = np.random.uniform(-0.5, 0.5, (N - len(pts), 2))
            pts = np.vstack([pts, pad])
        return pts
        
    return best_points

