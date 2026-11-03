import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the exact radius efficiently."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def soft_min_obj_and_grad(centers, beta=50.0):
        """
        Compute a smooth approximation of the negative radius objective.
        We minimize E = -r. 
        Using a log-sum-exp or similar soft-min for the minimum distance.
        Here we use a sum of exponentials for the inverse of distance (repulsion) 
        which is equivalent to maximizing the minimum distance in a smooth way.
        
        Actually, a better smooth surrogate for max-min is:
        f(x) = -1/beta * log( sum_i exp(-beta * d_i) )
        where d_i are the constraints (distances).
        
        However, standard BFGS works well with:
        E = sum_{i<j} 1/d_{ij}^2 + sum_i 1/wall_i^2
        This drives all distances to be large. The minimum will be the bottleneck.
        
        Let's use a potential energy:
        U = sum_{i<j} (1/d_{ij})^2 + sum_i (1/wall_i)^2
        This is smooth and differentiable.
        """
        n = len(centers)
        U = 0.0
        grad = np.zeros_like(centers)
        
        # Pairwise repulsion
        # To avoid O(N^2) in Python, we can use vectorized operations or cKDTree.
        # But for gradient, we need specific pairs.
        # For N=200, N^2/2 = 20k pairs. Python loop is slow.
        # Let's use vectorized numpy for pairs.
        
        # Compute all pairwise differences
        # This is 200x200x2, which is 80k floats. Manageable.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Potential: 1/d^2 = 1/dist_sq
        # Gradient wrt x_i: sum_j 2 * (x_i - x_j) / d^4
        # Let's compute force matrix F where F[i,j] is force on i due to j
        # F[i,j] = 2 * (x_i - x_j) / (d_ij^2)^2 = 2 * diff[i,j] / dist_sq[i,j]^2
        
        # To avoid division by zero on diagonal, set dist_sq diag to 1
        dist_sq_safe = dist_sq.copy()
        np.fill_diagonal(dist_sq_safe, 1.0)
        
        # Force on i from j: 2 * (x_i - x_j) / dist_sq^2
        # We need sum over j != i
        force_factor = 2.0 / (dist_sq_safe ** 2)
        np.fill_diagonal(force_factor, 0.0)
        
        # grad[i] = sum_j force_factor[i,j] * diff[i,j]
        # diff is (N, N, 2). force_factor is (N, N).
        # Multiply force_factor[..., np.newaxis] with diff
        weighted_diff = diff * force_factor[..., np.newaxis]
        grad = np.sum(weighted_diff, axis=1)
        
        # Potential energy U_pairs = sum_{i<j} 1/d^2 = 0.5 * sum_{i!=j} 1/d^2
        U_pairs = 0.5 * np.sum(1.0 / dist_sq_safe) # Diagonal is 1/1=1, so subtract 0.5*N
        U_pairs -= 0.5 * n
        
        # Wall potential
        # wall_dist_i = 0.5 - |x_i|
        # U_wall = sum_i 1/wall_i^2
        # Let's handle signs carefully.
        # x in [-0.5, 0.5]
        # wall_dist = 0.5 - abs(x)
        # dU/dx = -2 * d(1/w^2)/dw * dw/dx = -2 * (-2/w^3) * sign(x) = 4*sign(x)/w^3
        # Wait, U = 1/w^2. dU/dw = -2/w^3. dw/dx = -sign(x).
        # dU/dx = (-2/w^3) * (-sign(x)) = 2*sign(x)/w^3.
        
        wall_dists = 0.5 - np.abs(centers)
        # Avoid division by zero if on wall
        wall_dists_safe = np.maximum(wall_dists, 1e-6)
        
        U_wall = np.sum(1.0 / (wall_dists_safe ** 2))
        
        # Gradient for wall
        signs = np.sign(centers)
        # If center is exactly 0, sign is 0, but derivative is undefined. 
        # Numerically, it's fine to use 0 or random. 
        # Actually, 0.5 - abs(x) is not differentiable at 0.
        # Smooth approximation: use x / sqrt(x^2 + eps) instead of sign(x)
        # Or just use sign, BFGS handles kinks somewhat or we can jitter.
        # Let's use a smooth sign: x / (abs(x) + 1e-8)
        smooth_sign = centers / (np.abs(centers) + 1e-8)
        
        grad_wall = 2.0 * smooth_sign / (wall_dists_safe ** 3)
        
        grad += grad_wall
        U = U_pairs + U_wall
        
        # We want to maximize r, so minimize -r.
        # But this potential doesn't directly equal -r.
        # Minimizing U pushes all distances up. The bottleneck will be the min distance.
        # This is a valid surrogate.
        
        return U, grad

    def hexagonal_initialization(n_rows, n_cols):
        """Create hexagonal packing layout."""
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        return pts

    def square_grid_init():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0

    # Try multiple hexagonal layouts
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            r = compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()

    # Also try square grid
    grid = square_grid_init()
    r_grid = compute_radius(grid)
    if r_grid > best_radius:
        best_radius = r_grid
        best_centers = grid.copy()

    # Run BFGS optimization on the best initial configuration
    if best_centers is not None and time.time() < deadline - 20:
        # Flatten centers for scipy
        x0 = best_centers.flatten()
        
        def obj_and_grad(x):
            centers = x.reshape(-1, 2)
            # Clip to ensure we don't go crazy, though BFGS should stay close
            centers = np.clip(centers, -0.4999, 0.4999)
            U, grad = soft_min_obj_and_grad(centers)
            return U, grad

        # Use BFGS
        try:
            res = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if time.time() < deadline - 5:
                opt_centers = res.x.reshape(-1, 2)
                opt_centers = np.clip(opt_centers, -0.4999, 0.4999)
                r_opt = compute_radius(opt_centers)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_centers.copy()
        except Exception as e:
            pass # Keep previous best

    # Second pass: Refine with smaller step or different beta? 
    # L-BFGS-B is robust. Let's do a second polish with a tighter tolerance if time permits.
    if best_centers is not None and time.time() < deadline - 5:
        x0 = best_centers.flatten()
        
        def obj_and_grad(x):
            centers = x.reshape(-1, 2)
            centers = np.clip(centers, -0.4999, 0.4999)
            U, grad = soft_min_obj_and_grad(centers, beta=100.0) # Higher beta for sharper min? 
            # Wait, my soft_min_obj doesn't use beta. It uses 1/d^2.
            # Let's stick to the same potential, just refine.
            return U, grad

        try:
            res = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 20}
            )
            
            if time.time() < deadline - 1:
                opt_centers = res.x.reshape(-1, 2)
                opt_centers = np.clip(opt_centers, -0.4999, 0.4999)
                r_opt = compute_radius(opt_centers)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_centers.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = square_grid_init()

    return best_centers

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    t1 = time.time()
    r = compute_radius(c)
    print(f"Time: {t1-t0:.2f}s, Radius: {r}")
