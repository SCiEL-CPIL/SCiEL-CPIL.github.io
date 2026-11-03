import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Best known packing for N=50 has radius ~0.1258254895304
    # Structure: 1 center, 6 ring 1, 12 ring 2, 12 ring 3, 19 ring 4
    # We optimize radial distances AND angular offsets for each ring
    
    ring_sizes = [6, 12, 12, 19]  # 4 rings plus center
    
    def get_centers(params):
        """
        params: [r1, r2, r3, r4, a1, a2, a3, a4]
        where r_i are radial distances and a_i are angular offsets
        """
        centers = [[0.0, 0.0]]
        for i in range(4):
            r = params[i]
            offset = params[i + 4]
            n = ring_sizes[i]
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + offset
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            centers.extend(zip(xs, ys))
        return np.array(centers)
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        # Distance to boundary
        dist_to_center = np.linalg.norm(centers, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 0.0
        
        # Minimum distance between any two centers
        n = len(centers)
        min_dist = 2.0  # large initial value
        for i in range(n):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            if i > 0:
                min_dist = min(min_dist, np.min(dists[:i]))
            else:
                if n > 1:
                    min_dist = min(min_dist, np.min(dists[1:]))
        
        r_circle = min(min_dist / 2.0, 1.0 - max_dist)
        return r_circle
    
    def soft_min(values, beta=50.0):
        """Smooth approximation of min using log-sum-exp."""
        values = np.asarray(values)
        if len(values) == 0:
            return 0.0
        m = np.min(values)
        if not np.isfinite(m):
            return 0.0
        shifted = values - m
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
        return m - log_sum
    
    def objective(params):
        """
        Negative of the radius using epigraph formulation with smooth min.
        params: [r1, r2, r3, r4, a1, a2, a3, a4, t]
        where t is the auxiliary variable (radius to maximize)
        """
        radii = np.abs(params[:4])
        radii = np.clip(radii, 0.05, 0.95)
        t = params[8]
        
        params_clipped = np.concatenate([radii, params[4:8]])
        
        centers = get_centers(params_clipped)
        
        # Check validity - penalty if outside
        dist_to_center = np.linalg.norm(centers, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 10.0 + max_dist
        
        # Compute all pair distances
        n = len(centers)
        # Use vectorized computation for speed
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Get all pair distances (upper triangle)
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        # Constraints: pair_dist >= 2*t and (1 - dist_to_center) >= t
        # Objective: maximize t, so minimize -t + penalties
        
        # Soft-min of constraints
        # For pairs: want min(pair_dist - 2*t) to be >= 0
        # For boundary: want min(1 - dist_to_center - t) to be >= 0
        
        pair_slack = pair_dists - 2.0 * t
        boundary_slack = 1.0 - dist_to_center - t
        
        # Use soft-min to create smooth penalty
        beta = 50.0
        min_pair_slack = soft_min(pair_slack, beta)
        min_boundary_slack = soft_min(boundary_slack, beta)
        
        # Penalty for negative slacks
        penalty = 0.0
        if min_pair_slack < 0:
            penalty += 100.0 * min_pair_slack**2
        if min_boundary_slack < 0:
            penalty += 100.0 * min_boundary_slack**2
        
        return -t + penalty
    
    def objective_simple(params):
        """Simpler objective: just maximize the computed radius."""
        radii = np.abs(params[:4])
        radii = np.clip(radii, 0.05, 0.95)
        params_clipped = np.concatenate([radii, params[4:8]])
        
        centers = get_centers(params_clipped)
        
        dist_to_center = np.linalg.norm(centers, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 1.0 + max_dist
        
        r = compute_radius(centers)
        if r <= 0:
            return 1.0
        return -r
    
    # Initial guess based on known good values
    # For r_circle ≈ 0.1258, the ring radii should be approximately:
    # Ring 1: 2*r ≈ 0.252
    # Ring 2: ~0.45
    # Ring 3: ~0.65
    # Ring 4: ~0.85
    initial_params = np.array([0.252, 0.45, 0.65, 0.85, 0.0, np.pi/12, np.pi/6, 0.0, 0.12])
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    best_params = initial_params.copy()
    best_score = objective_simple(initial_params[:8])
    best_centers = get_centers(best_params[:8])
    
    # Try multiple optimization methods and initial conditions
    # First, use the simple objective with Nelder-Mead for robustness
    methods_simple = ['Nelder-Mead', 'Powell']
    
    # Generate diverse initial conditions
    initial_conditions = [initial_params[:8]]
    
    # Perturb initial conditions
    np.random.seed(42)
    for _ in range(8):
        perturbed = initial_params[:8].copy()
        perturbed[:4] += np.random.randn(4) * 0.03
        perturbed[4:] += np.random.randn(4) * 0.1
        perturbed[:4] = np.clip(np.abs(perturbed[:4]), 0.05, 0.95)
        initial_conditions.append(perturbed)
    
    for method in methods_simple:
        for init in initial_conditions:
            if time.time() - start_time > time_limit - 30.0:
                break
            
            try:
                options = {}
                if method == 'Nelder-Mead':
                    options = {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                elif method == 'Powell':
                    options = {'maxiter': 1500, 'ftol': 1e-10, 'xtol': 1e-10}
                
                res = minimize(objective_simple, init, method=method, options=options)
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params[:8] = res.x.copy()
                    best_params[:4] = np.clip(np.abs(best_params[:4]), 0.05, 0.95)
                    best_centers = get_centers(best_params[:8])
                    
                    # Verify
                    dist_to_center = np.linalg.norm(best_centers, axis=1)
                    if np.max(dist_to_center) < 1.0:
                        current_r = compute_radius(best_centers)
                        if current_r > 0:
                            best_score = -current_r
                            
            except Exception:
                pass
        
        if time.time() - start_time > time_limit - 30.0:
            break
    
    # Now use the epigraph formulation with L-BFGS-B for refinement
    if time.time() - start_time < time_limit - 20.0:
        try:
            # Initialize t to current best radius
            current_r = -best_score
            t_init = max(current_r, 0.1)
            
            # Try multiple t initializations
            t_inits = [t_init, t_init * 1.01, t_init * 1.02, 0.125]
            
            for t0 in t_inits:
                if time.time() - start_time > time_limit - 15.0:
                    break
                
                init_epigraph = np.concatenate([best_params[:8], [t0]])
                
                res = minimize(objective, init_epigraph, method='L-BFGS-B',
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50})
                
                if res.fun < best_score:
                    new_params = res.x
                    new_t = new_params[8]
                    new_r = compute_radius(get_centers(new_params[:8]))
                    if new_r > 0 and new_r > current_r:
                        best_score = -new_r
                        best_params[:8] = new_params[:8]
                        best_params[8] = new_t
                        best_centers = get_centers(best_params[:8])
                        current_r = new_r
                        
        except Exception:
            pass
    
    # Final local refinement with tight tolerances
    if time.time() - start_time < time_limit - 10.0:
        try:
            res = minimize(objective_simple, best_params[:8], method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_params[:8] = res.x.copy()
                best_params[:4] = np.clip(np.abs(best_params[:4]), 0.05, 0.95)
                best_centers = get_centers(best_params[:8])
                
                dist_to_center = np.linalg.norm(best_centers, axis=1)
                if np.max(dist_to_center) < 1.0:
                    current_r = compute_radius(best_centers)
                    if current_r > 0:
                        best_score = -current_r
        except Exception:
            pass
    
    # Another L-BFGS-B pass with updated t
    if time.time() - start_time < time_limit - 5.0:
        try:
            current_r = -best_score
            t_init = max(current_r, 0.1)
            init_epigraph = np.concatenate([best_params[:8], [t_init]])
            
            res = minimize(objective, init_epigraph, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            
            if res.fun < best_score:
                new_params = res.x
                new_r = compute_radius(get_centers(new_params[:8]))
                if new_r > 0:
                    best_score = -new_r
                    best_params[:8] = new_params[:8]
                    best_params[8] = new_params[8]
                    best_centers = get_centers(best_params[:8])
        except Exception:
            pass
    
    # Final validation
    dist_to_center = np.linalg.norm(best_centers, axis=1)
    if np.max(dist_to_center) >= 1.0 or len(best_centers) != N:
        # Fallback
        return _fallback_hexagonal()
    
    # Final polish: tiny Nelder-Mead step
    if time.time() - start_time < time_limit - 2.0:
        try:
            res = minimize(objective_simple, best_params[:8], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_score:
                best_params[:8] = res.x.copy()
                best_params[:4] = np.clip(np.abs(best_params[:4]), 0.05, 0.95)
                best_centers = get_centers(best_params[:8])
                
                dist_to_center = np.linalg.norm(best_centers, axis=1)
                if np.max(dist_to_center) < 1.0:
                    current_r = compute_radius(best_centers)
                    if current_r > 0:
                        best_score = -current_r
        except Exception:
            pass
    
    # Final validation
    dist_to_center = np.linalg.norm(best_centers, axis=1)
    if np.max(dist_to_center) >= 1.0 or len(best_centers) != N:
        return _fallback_hexagonal()
    
    return best_centers

def _fallback_hexagonal():
    """Simple hexagonal packing as a safe fallback."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Pick N points closest to center for better packing
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

# EVOLVE-BLOCK-END
