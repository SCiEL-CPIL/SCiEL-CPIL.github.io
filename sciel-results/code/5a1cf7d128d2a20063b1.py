"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_exact_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Pairwise distances
        # Use broadcasting for vectorized distance calculation
        # diff[i, j] = pts[i] - pts[j]
        # We only need the upper triangle to avoid duplicates and self-distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle."""
        # Estimate spacing. Area per circle ~ pi * r^2. 
        # Total area pi. N circles. r ~ 1/sqrt(N).
        # Spacing ~ 2r ~ 2/sqrt(N).
        # For N=300, r ~ 0.057, spacing ~ 0.114.
        # Hex lattice row height = spacing * sqrt(3)/2.
        
        best_pts = None
        best_score = -1.0
        
        # Try a few spacings to ensure we get enough points
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            spacing = 2.0 / (np.sqrt(N) * scale)
            
            # Generate grid points
            # We need enough points to cover the unit circle.
            # Max radius 1. 
            # x range [-1, 1], y range [-1, 1]
            # Number of points in x: ~ 2/spacing + 1
            # Number of points in y: ~ 2/(spacing*sqrt(3)/2) + 1
            
            nx = int(2.0 / spacing) + 3
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
            
            # Center the grid
            x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
            
            # Create meshgrid
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Apply hex offset: shift even rows by spacing/2
            # Y index 0, 2, 4... are even rows? 
            # Let's iterate or use vectorized logic.
            # Row index is axis 0 of the meshgrid.
            rows = np.arange(len(y_coords))
            # Shift x for even rows (0, 2, ...)
            # Actually, standard hex packing: 
            # Row 0: x = 0, s, 2s...
            # Row 1: x = s/2, 3s/2...
            # Let's just generate all points and filter.
            
            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    # Check if inside unit circle with some margin for radius
                    # We don't know r yet, but it's roughly spacing/2.
                    # Let's keep points with norm < 1 - spacing/2
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])
            
            if len(pts_list) < N:
                continue
                
            pts = np.array(pts_list)
            
            # If we have more than N, select the N closest to the center?
            # Or just take the first N? 
            # Better: select N points that maximize the min distance.
            # For now, take the N points closest to the origin to ensure density.
            # Actually, for a good start, we want to cover the area.
            # A common heuristic is to take the N points with the smallest norm.
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]
            
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            # Fallback: random
            best_pts = np.random.rand(N, 2) * 2 - 1
            # Ensure inside
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)
            
        return best_pts, best_score

    def soft_min(x, y, beta):
        """
        Smooth approximation of min(x, y).
        x, y are scalars or arrays.
        """
        # log-sum-exp trick for stability
        # min(x, y) = -1/beta * log(exp(-beta*x) + exp(-beta*y))
        # To avoid overflow, we can shift, but x,y are small (~0.05), beta up to 1000.
        # exp(-50) is ~ 1e-22, safe.
        # If beta is very large, we might lose precision.
        # Let's use a stable formulation.
        
        # If beta is small, it's an average.
        # If beta is large, it's min.
        
        # For scalar x, y:
        if np.isscalar(x) and np.isscalar(y):
            # Avoid overflow in exp
            # log(exp(-b*x) + exp(-b*y)) = -b*min(x,y) + log(exp(-b*(x-min)) + exp(-b*(y-min)))
            m = min(x, y)
            return -1.0 / beta * (np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m))) - beta * m)
        
        # Vectorized case not needed for the final scalar objective, 
        # but useful if we wanted to vectorize the objective itself (we don't).
        
    def create_objective(pts_init, beta):
        """
        Create the objective function for L-BFGS-B.
        Returns negative of the soft-min radius.
        """
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            # Check validity
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            # Radial distances
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                # Penalty for being outside
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            # Pairwise distances
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            # We want to maximize r = min(min_pair_dist/2, min_radial_dist)
            # Let x = min_pair_dist / 2.0
            # Let y = min_radial_dist
            # Objective = -soft_min(x, y, beta)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            # If beta is very high, soft_min is close to min.
            # If beta is low, it's smoother.
            
            val = soft_min(x, y, beta)
            
            return -val
        
        return objective

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()
    
    # 2. Multi-stage optimization with increasing beta
    # Start with low beta for global movement, increase for local precision.
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_pts = pts
    current_score = best_score
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        objective = create_objective(current_pts, beta)
        
        # Flatten points for optimization
        x0 = current_pts.flatten()
        
        # Bounds: keep points inside the unit circle (with small margin)
        # Actually, L-BFGS-B with bounds is good.
        # But the objective has a penalty for going outside.
        # Let's use bounds [-1, 1] for x and y.
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < -current_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
                    
        except Exception as e:
            # If optimization fails, continue with current points
            pass

    # 3. Final Polish with very high beta
    if time.time() < deadline - 5:
        beta_final = 2000.0
        objective = create_objective(current_pts, beta_final)
        x0 = current_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < -current_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # Final check and return
    if current_score <= 0:
        # Fallback to valid random
        current_pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(current_pts, axis=1)
        current_pts = current_pts * (0.9 / norms)[:, None]
        
    return current_pts
# EVOLVE-BLOCK-END
