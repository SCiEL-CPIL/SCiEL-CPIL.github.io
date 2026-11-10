"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # 1. Initialization: Hexagonal lattice
    # This is a structurally superior starting point compared to a square grid
    # because it mimics the local packing structure of optimal circle packings.
    
    # Ideal spacing for 200 points in unit square
    # Area = 1, N = 200
    # Area per point = 1/200 = 0.005
    # Hex area per point = d^2 * sqrt(3)/2
    # d = sqrt(0.005 * 2 / sqrt(3))
    d = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate points in a hexagonal grid
    # We need to cover [-0.5, 0.5] x [-0.5, 0.5]
    # x = i * d
    # y = j * d * sqrt(3)/2 + (i % 2) * d * sqrt(3)/4
    
    max_idx = int(np.ceil(0.5 / d)) + 2
    xs = []
    ys = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            x = i * d
            y = j * d * np.sqrt(3) / 2 + (i % 2) * d * np.sqrt(3) / 4
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                xs.append(x)
                ys.append(y)
    
    points = np.array([xs, ys]).T
    
    # If we have more than N points, we need to select N
    # If we have fewer, we might have scaled too big or small.
    # For d calculated above, we should have approx N points.
    
    if len(points) > N:
        # Select points to maintain lattice structure as much as possible
        # A simple heuristic: Sort by distance from center? No, that creates a circle.
        # Better: Randomly sample? No.
        # Let's try to keep a regular subset.
        # Or simply: if we have too many, we can reduce d slightly and regenerate?
        # Or just take the first N? The generation order is row-major in a square patch.
        # Let's try to select points that are evenly distributed.
        # For now, let's just take a random subset to break symmetry, 
        # but ensure we have exactly N.
        # Actually, a better approach for hex lattice:
        # If we have M points, we can select every k-th point?
        # Let's just use the first N points generated, which form a contiguous patch.
        # This might be biased.
        # Let's try to center the patch.
        # The patch is roughly square.
        # Let's just take the first N.
        points = points[:N]
    elif len(points) < N:
        # Fallback: Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]

    # 2. Optimization: Physics-based L-BFGS-B
    # We define a potential energy function that is minimized when the circles are packed tightly.
    # E = sum_{i<j} V(d_ij) + sum_i V_boundary(d_i)
    # where V is a repulsive potential that goes to infinity as d -> 0.
    # A common choice is V(d) = 1/d^2 or similar, but we need to be careful with gradients.
    # Alternatively, we can maximize the minimum radius directly, but that is non-smooth.
    # Using a smooth approximation:
    # Let r_min be the current minimum radius.
    # We want to maximize r_min.
    # This is equivalent to minimizing -r_min.
    # Since r_min is non-differentiable, we can use a soft penalty:
    # E = sum_{i<j} max(0, r_target - d_ij/2)^2 + sum_i max(0, r_target - d_i_boundary)^2
    # But we don't know r_target.
    
    # Better approach: Use a repulsive potential that penalizes close distances.
    # E = sum_{i<j} 1/d_ij^2 + sum_i 1/d_i_boundary^2
    # Minimizing this pushes points apart.
    # However, this doesn't directly maximize the *minimum* distance, it minimizes the *sum* of inverse squares.
    # This is a good proxy because if the minimum distance is small, the sum will be large.
    # To bias towards the minimum, we can use a steeper potential, e.g., 1/d^4.
    
    # Let's define the objective function for L-BFGS-B.
    # The variables are the flattened array of points: [x0, y0, x1, y1, ...]
    
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        
        # Calculate pairwise distances
        # diff[i, j, :] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Avoid division by zero
        # We only care about i < j
        # Create a mask for i < j
        # Or just compute all and ignore diagonal and double count
        # Let's use a mask for efficiency and correctness
        # Actually, for N=200, 200x200 matrix is fine.
        
        # Set diagonal to inf to avoid division by zero
        np.fill_diagonal(dist_sq, np.inf)
        
        # Inverse square distance
        # We want to minimize sum(1/d^2)
        # But if d is very small, 1/d^2 is huge.
        # To avoid numerical issues, we can clip d^2 to a small value.
        min_dist_sq = 1e-12
        dist_sq_clipped = np.maximum(dist_sq, min_dist_sq)
        
        # Sum of inverse squares for all pairs (i,j) where i != j
        # This counts each pair twice (i,j) and (j,i)
        pair_energy = np.sum(1.0 / dist_sq_clipped)
        
        # Boundary distances
        # d_b_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        
        # Clip boundary distances
        d_b_clipped = np.maximum(d_b, min_dist_sq)
        
        # Boundary energy
        boundary_energy = np.sum(1.0 / d_b_clipped)
        
        # Total energy
        # We can weight the boundary energy if needed, but 1:1 is usually fine.
        total_energy = pair_energy + boundary_energy
        
        return total_energy

    # Initial guess
    x0 = points.flatten()
    
    # Bounds: points must be inside [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = -1.0
    best_points = points.copy()
    
    # Helper to calculate the actual radius score
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        min_boundary_dist = np.min(d_b)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    current_r = calc_radius(points)
    if current_r > best_r:
        best_r = current_r
        best_points = points.copy()

    # Run L-BFGS-B
    # We might run it multiple times with different initializations or restarts if time permits.
    # But L-BFGS-B is quite efficient.
    
    try:
        # maxiter=1000 might be enough, but let's let it run until time is up or convergence.
        # We can't easily interrupt L-BFGS-B mid-run without a callback.
        # Let's use a callback to check time.
        
        def callback(flat_x):
            nonlocal best_r, best_points
            if time.time() > deadline:
                return True  # Stop optimization
            
            # Calculate current radius
            pts = flat_x.reshape(N, 2)
            # Clip to ensure validity during optimization (though bounds should handle it)
            pts_clipped = np.clip(pts, -0.5, 0.5)
            r = calc_radius(pts_clipped)
            if r > best_r:
                best_r = r
                best_points = pts_clipped.copy()
            return False

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 5000, 'callback': callback, 'maxls': 50})
        
        # Check result
        if result.success:
            final_points = result.x.reshape(N, 2)
            final_points = np.clip(final_points, -0.5, 0.5)
            final_r = calc_radius(final_points)
            if final_r > best_r:
                best_r = final_r
                best_points = final_points.copy()
                
    except Exception as e:
        # If optimization fails, return the best found so far
        pass

    # If we have time, we can try a local search or restart from a perturbed best solution.
    # But L-BFGS-B is usually good.
    
    # One more refinement: If we have time, do a simple gradient ascent on the minimum distance?
    # Or just return the best found.
    
    return best_points

# EVOLVE-BLOCK-END
