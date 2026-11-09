import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n = 8
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(pairs)
    
    def compute_diameter(points):
        """Compute the maximum pairwise distance."""
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d > max_d:
                    max_d = d
        return max_d
    
    def score_points(points):
        """Compute the score: product of all pairwise distances after scaling to diameter 2, divided by n^n."""
        d_max = compute_diameter(points)
        if d_max < 1e-12:
            return 0.0
        scale = 2.0 / d_max
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j]) * scale
                if d < 1e-12:
                    return 0.0
                log_prod += math.log(d)
        # log(score) = log_prod - n*log(n)
        # score = exp(log_prod) / n^n
        log_score = log_prod - n * math.log(n)
        if log_score > 700:  # overflow protection
            return float('inf')
        return math.exp(log_score)
    
    def neg_log_score(points_flat):
        """Negative log of the unnormalized product of distances (for optimization).
        We maximize sum of log(distances) subject to diameter <= 2.
        Since scale-invariant, we can fix diameter = 2 and maximize sum log(d_ij)."""
        P = points_flat.reshape(n, 2)
        log_sum = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(P[i] - P[j])
                if d < 1e-12:
                    return 1e10  # large penalty
                log_sum += math.log(d)
        return -log_sum
    
    # Create constraint functions for diameter <= 2
    def make_diameter_constraint(i, j):
        def constraint(points_flat):
            P = points_flat.reshape(n, 2)
            d = np.linalg.norm(P[i] - P[j])
            return 2.0 - d
        return constraint
    
    constraints = []
    for i, j in pairs:
        constraints.append({'type': 'ineq', 'fun': make_diameter_constraint(i, j)})
    
    bounds = [(-3.0, 3.0)] * (2 * n)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initial guesses
    guesses = []
    
    # 1. Regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    guesses.append(p_oct.flatten())
    
    # 2. Two squares (one rotated 45 degrees)
    angles_sq1 = 2 * np.pi * np.arange(4) / 4
    angles_sq2 = angles_sq1 + np.pi/4
    p_sq1 = np.column_stack([np.cos(angles_sq1), np.sin(angles_sq1)])
    p_sq2 = np.column_stack([np.cos(angles_sq2), np.sin(angles_sq2)])
    guesses.append(np.vstack([p_sq1, p_sq2]).flatten())
    
    # 3. Random points with some structure
    np.random.seed(123)
    for seed_val in [42, 7, 99, 2024]:
        np.random.seed(seed_val)
        r = np.random.uniform(0.3, 1.0, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        P = np.column_stack([x, y])
        # Scale to diameter ~2
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())
    
    # 4. Points with x-axis symmetry (4 pairs mirrored)
    for seed_val in [5, 11, 23]:
        np.random.seed(seed_val)
        # 4 unique points in upper half, mirror to lower
        r = np.random.uniform(0.3, 1.0, 4)
        theta = np.random.uniform(0, np.pi, 4)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        P_upper = np.column_stack([x, y])
        P_lower = np.column_stack([x, -y])
        P = np.vstack([P_upper, P_lower])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())
    
    # 5. Concentric circles: 4 on inner, 4 on outer
    for r_inner in [0.3, 0.5, 0.7]:
        r_outer = 1.0
        angles_inner = 2 * np.pi * np.arange(4) / 4
        angles_outer = 2 * np.pi * np.arange(4) / 4 + np.pi/4
        p_inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        p_outer = r_outer * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        P = np.vstack([p_inner, p_outer])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())
    
    # Optimize each guess
    for idx, p0 in enumerate(guesses):
        if time.time() > deadline - 5.0:
            break
        
        try:
            # First pass: SLSQP
            res = minimize(
                neg_log_score,
                p0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success and res.x is not None:
                P_opt = res.x.reshape(n, 2)
                # Check distinctness
                min_dist = min(
                    np.linalg.norm(P_opt[i] - P_opt[j])
                    for i in range(n) for j in range(i+1, n)
                )
                if min_dist > 1e-8:
                    sc = score_points(P_opt)
                    if sc > best_score:
                        best_score = sc
                        best_points = P_opt.copy()
                        
                        # Second pass: polish with tighter tolerance
                        if time.time() < deadline - 3.0:
                            res2 = minimize(
                                neg_log_score,
                                P_opt.flatten(),
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 2000, 'ftol': 1e-14}
                            )
                            if res2.success and res2.x is not None:
                                P_polish = res2.x.reshape(n, 2)
                                min_dist2 = min(
                                    np.linalg.norm(P_polish[i] - P_polish[j])
                                    for i in range(n) for j in range(i+1, n)
                                )
                                if min_dist2 > 1e-8:
                                    sc2 = score_points(P_polish)
                                    if sc2 > best_score:
                                        best_score = sc2
                                        best_points = P_polish.copy()
        except Exception:
            continue
    
    # Fallback to regular octagon
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure valid return
    return [[float(p[0]), float(p[1])] for p in best_points]
