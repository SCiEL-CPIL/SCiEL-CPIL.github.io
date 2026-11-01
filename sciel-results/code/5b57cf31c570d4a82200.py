"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s safety margin for 120s limit

    # 1. Improved Initial Hexagonal Lattice with multiple attempts
    def generate_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            offset = 0.5 * (j % 2)
            for i in range(-k, k + 1):
                x = (i + offset) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append((x, y))
        points = np.array(points)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points well inside to allow movement
        mask = radius <= 1.0 - spacing / 2.0
        return points[mask], radius[mask]

    # Try multiple spacings to get a good initial configuration
    x0 = None
    best_initial_score = -np.inf
    for trial in range(5):
        spacing = 2.0 / (np.sqrt(N) + 1.0 + trial * 0.05)
        points, radius = generate_hex_lattice(spacing)
        if len(points) >= N:
            # Select N points closest to center
            idx = np.argsort(radius, kind="stable")[:N]
            candidate = points[idx].copy()
            score = -raw_score(candidate)
            if score > best_initial_score:
                best_initial_score = score
                x0 = candidate
    
    if x0 is None:
        # Fallback: uniform grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        points, radius = generate_hex_lattice(spacing)
        if len(points) >= N:
            x0 = points[np.argsort(radius, kind="stable")[:N]].copy()
        else:
            # Last resort: random points
            x0 = np.random.randn(N, 2) * 0.8

    def get_radius_candidates(x):
        """
        Returns an array of all 'radius candidates':
        - 1 - |x_i| for each point (distance to boundary)
        - |x_i - x_j| / 2 for each pair (half distance between circles)
        The score is the minimum of this array.
        """
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x) / 2.0
        return np.concatenate([dist_boundary, d_pair])

    def soft_min_score(x, beta):
        """
        Approximates -min(d) using log-sum-exp.
        As beta -> inf, this approaches -min(d).
        We minimize this function.
        """
        x = x.reshape(-1, 2)
        d = get_radius_candidates(x)
        # Numerical stability
        A = -beta * d
        max_A = np.max(A)
        log_sum_exp = max_A + np.log(np.sum(np.exp(A - max_A)))
        return log_sum_exp / beta

    def raw_score(x):
        """
        The actual objective to maximize: min(radius_candidates).
        We return negative for minimization.
        """
        x = x.reshape(-1, 2)
        d = get_radius_candidates(x)
        return -np.min(d)

    best_x = x0.copy()
    best_score = -raw_score(best_x)

    bounds = [(-1.0, 1.0)] * (2 * N)

    # 2. Annealing Optimization with Soft-Min - more aggressive schedule
    # Start with smooth (low beta) to allow global rearrangement, then sharpen
    betas = [5, 20, 50, 100, 200, 400, 800]
    
    current_x = best_x.copy()
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        def neg_soft_min(x):
            return soft_min_score(x, beta)

        try:
            # Use L-BFGS-B. It's fast and handles bounds.
            res = minimize(neg_soft_min, current_x, method='L-BFGS-B', 
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < soft_min_score(current_x, beta):
                current_x = res.x.copy()
                
                # Update best raw score
                current_raw = -raw_score(current_x)
                if current_raw > best_score:
                    best_score = current_raw
                    best_x = current_x.copy()
                    
        except Exception:
            continue

    # 3. Final Polish with Raw Objective - multiple passes for high precision
    # The soft-min might leave points slightly suboptimal for the strict min.
    # Use a local search on the raw objective.
    polish_stages = [
        ('Nelder-Mead', {'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}),
        ('L-BFGS-B', {'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}),
        ('Nelder-Mead', {'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}),
    ]
    
    for method, opts in polish_stages:
        if time.time() > deadline - 3:
            break
        
        try:
            if method == 'L-BFGS-B':
                res = minimize(raw_score, best_x, method=method,
                               bounds=bounds, options=opts)
            else:
                res = minimize(raw_score, best_x, method=method, options=opts)
            
            if res.fun < -best_score:
                best_x = res.x.copy()
                best_score = -res.fun
        except Exception:
            pass

    # Final validation: ensure all points are inside the unit circle
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        # Project back inside
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
        
    # Ensure no NaNs or Infs
    if not np.all(np.isfinite(best_x)):
        return x0

    return best_x
# EVOLVE-BLOCK-END
