import numpy as np
from scipy.optimize import minimize

def _compute_score(points):
    points = np.array(points)
    n = len(points)
    if n < 2:
        return 0.0
    
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                return -np.inf
            dists.append(d)
    
    diameter = max(dists)
    if diameter < 1e-10:
        return -np.inf
    
    scale = 2.0 / diameter
    log_product = np.sum(np.log(dists)) + (n * (n - 1) / 2.0) * np.log(scale)
    return log_product

def _objective_and_gradient(points_flat, diam_idx_i, diam_idx_j):
    n = 8
    points = np.array(points_flat).reshape(n, 2)
    
    # Calculate distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                return 1e10, np.zeros_like(points_flat)
            dists[i, j] = d
            dists[j, i] = d
            
    # Diameter is assumed to be between diam_idx_i and diam_idx_j
    d_diam = dists[diam_idx_i, diam_idx_j]
    if d_diam < 1e-10:
        return 1e10, np.zeros_like(points_flat)
        
    # Check if this is actually the diameter (for validity of the smooth approximation)
    # If not, the gradient is wrong, but we might still converge to a local opt where this IS the diameter.
    # For now, we just compute the value and gradient based on this assumption.
    
    N_pairs = n * (n - 1) / 2.0
    
    # Objective: Maximize sum(log(d_ij)) - N_pairs * log(d_diam)
    # We minimize the negative.
    sum_log_dists = np.sum(np.log(dists[np.triu_indices(n, k=1)]))
    val = sum_log_dists - N_pairs * np.log(d_diam)
    
    # Gradient of val w.r.t points
    # grad_val[m] = sum_{j != m} (P_m - P_j) / d_mj^2 - N_pairs * grad_log_d_diam[m]
    grad_val = np.zeros((n, 2))
    
    for m in range(n):
        # Term 1: sum_{j != m} (P_m - P_j) / d_mj^2
        for j in range(n):
            if i == j: continue
            d = dists[m, j]
            grad_val[m] += (points[m] - points[j]) / (d * d)
            
        # Term 2: - N_pairs * d/dP_m (log d_diam)
        if m == diam_idx_i:
            # d/dP_i log ||P_i - P_j|| = (P_i - P_j) / ||P_i - P_j||^2
            grad_val[m] -= N_pairs * (points[diam_idx_i] - points[diam_idx_j]) / (d_diam * d_diam)
        elif m == diam_idx_j:
            # d/dP_j log ||P_i - P_j|| = (P_j - P_i) / ||P_i - P_j||^2
            grad_val[m] -= N_pairs * (points[diam_idx_j] - points[diam_idx_i]) / (d_diam * d_diam)
            
    # We want to maximize val, so we minimize -val
    # Gradient of -val is -grad_val
    return -val, -grad_val.flatten()

def construct():
    best_score = -np.inf
    best_points = None
    
    # Regular octagon
    angles = [2 * np.pi * k / 8 for k in range(8)]
    reg_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    reg_score = _compute_score(reg_points)
    best_score = reg_score
    best_points = reg_points.flatten()
    
    n = 8
    N_pairs = n * (n - 1) / 2.0
    
    # Try BFGS optimization assuming different pairs are the diameter
    # For regular octagon, diameters are (0,4), (1,5), (2,6), (3,7)
    diam_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    
    # Also try some "distorted" starts
    starts = [reg_points]
    # Perturb regular octagon
    for _ in range(5):
        perturbed = reg_points + np.random.normal(0, 0.05, (8, 2))
        starts.append(perturbed)
        
    # Try a configuration of two concentric squares (4-fold symmetry)
    # Points: (r1, 0), (-r1, 0), (0, r1), (0, -r1) and (r2/sqrt(2), r2/sqrt(2)) ... 
    # Actually, let's just use random starts with symmetry constraints?
    # For now, stick to BFGS on full space from these starts.
    
    for start in starts:
        x0 = start.flatten()
        for i, j in diam_pairs:
            try:
                # Use BFGS with analytical gradient
                res = minimize(
                    lambda x: _objective_and_gradient(x, i, j)[0],
                    x0,
                    jac=lambda x: _objective_and_gradient(x, i, j)[1],
                    method='BFGS',
                    options={'maxiter': 500, 'gtol': 1e-8}
                )
                
                if res.success:
                    pts = res.x.reshape(8, 2)
                    # Verify validity: distinct points, and check if the assumed diameter is indeed the diameter
                    # If the assumed diameter is not the actual diameter, the score calculation in _compute_score will handle scaling correctly,
                    # but the optimization might have been guided by the wrong gradient.
                    # We just evaluate the true score.
                    curr_score = _compute_score(pts)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_points = res.x
            except Exception:
                pass

    if best_points is not None:
        pts = best_points.reshape(8, 2)
        # Check validity
        for i in range(8):
            for j in range(i+1, 8):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-5:
                    return [[float(np.cos(2 * np.pi * k / 8)), float(np.sin(2 * np.pi * k / 8))] for k in range(8)]
        return [[float(p[0]), float(p[1])] for p in pts]
    
    return [[float(np.cos(2 * np.pi * k / 8)), float(np.sin(2 * np.pi * k / 8))] for k in range(8)]
