"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Hexagonal lattice initialization ---
    def hex_init():
        # For N=200, a hexagonal arrangement with slightly varied spacing
        # Start with a base hexagonal lattice
        # Approximate number of columns/rows
        # Hex packing: each row has alternating offsets
        # Try to fit ~14x15 grid with hex offset
        # Actually, let's generate a dense hex lattice and pick the best N
        # Or better: generate points in a hex pattern within the square
        
        # Generate hex lattice points
        # Hex lattice: x = i * dx + (j % 2) * dx/2, y = j * dy
        # where dx = 2r, dy = r * sqrt(3)
        # We want to maximize r, so let's start with a reasonable r
        # For N=200 in unit square, r ~ 0.0366
        # dx ~ 0.0732, dy ~ 0.0627
        
        # Generate a grid of hex points
        r_est = 0.035
        dx = 2 * r_est
        dy = r_est * np.sqrt(3)
        
        # Generate points in hex pattern
        points = []
        # x ranges from -0.5 to 0.5
        # For each row j, x starts at -0.5 + (j%2)*dx/2
        j = 0
        while True:
            y = -0.5 + dy/2 + j * dy
            if y > 0.5:
                break
            offset = (j % 2) * dx / 2
            i = 0
            while True:
                x = -0.5 + dx/2 + offset + i * dx
                if x > 0.5:
                    break
                points.append((x, y))
                i += 1
            j += 1
        
        points = np.array(points)
        
        # If we have more than N points, select the best N
        # If we have fewer, add more
        if len(points) > N:
            # Select points that maximize minimum distance
            # Use a greedy approach: start with corners, then add farthest point
            # But for speed, just take a subset
            # Better: use a simple selection - take every k-th point or random
            # For now, just take the first N or a random subset
            # Actually, let's use a smarter approach: sort by distance from center and take
            # Or just take a random subset and rely on optimization
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
        elif len(points) < N:
            # Add random points
            extra = np.random.uniform(-0.45, 0.45, (N - len(points), 2))
            points = np.vstack([points, extra])
        
        # Clip to valid range
        points = np.clip(points, -0.49, 0.49)
        return points

    # --- Objective function: compute the radius (minimum of pairwise distances/2 and boundary distances) ---
    def compute_radius(pts):
        # Pairwise distances
        tree = cKDTree(pts)
        # Find nearest neighbors
        # Actually, we need the minimum distance
        # Use query_pairs for efficiency
        pairs = tree.query_pairs(2 * 0.04)  # Slightly larger than expected radius
        if len(pairs) == 0:
            min_dist = 1.0
        else:
            min_dist = np.inf
            for i, j in pairs:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
            # Also check all pairs if the threshold was too small
            # For safety, compute min over all pairs for small N
            # But N=200, so 19900 pairs, which is fine
            # Let's just compute all pairwise distances for correctness
            from scipy.spatial.distance import pdist
            min_dist = pdist(pts).min()
        
        # Boundary distances
        # Distance to each wall
        dist_left = 0.5 + pts[:, 0]
        dist_right = 0.5 - pts[:, 0]
        dist_bottom = 0.5 + pts[:, 1]
        dist_top = 0.5 - pts[:, 1]
        min_boundary = np.min([dist_left.min(), dist_right.min(), dist_bottom.min(), dist_top.min()])
        
        radius = min(min_dist / 2, min_boundary)
        return radius

    # --- Optimization using stochastic subgradient / perturbation ---
    def optimize(pts, time_budget):
        best_pts = pts.copy()
        best_r = compute_radius(best_pts)
        
        # Gradient-free optimization with perturbations
        # Use a simple simulated annealing or random restart approach
        
        # First, try local optimization with small perturbations
        step_size = 0.01
        n_iter = 0
        max_iter = 50000
        
        while n_iter < max_iter:
            if time.time() > time_budget:
                break
            
            # Random perturbation
            idx = np.random.randint(0, N)
            delta = np.random.normal(0, step_size, 2)
            new_pts = best_pts.copy()
            new_pts[idx] += delta
            
            # Clip to valid range
            new_pts = np.clip(new_pts, -0.49, 0.49)
            
            new_r = compute_radius(new_pts)
            
            if new_r >= best_r:
                best_pts = new_pts
                best_r = new_r
                # Reduce step size occasionally
                if n_iter % 1000 == 0:
                    step_size *= 0.95
            else:
                # Occasionally accept worse solutions (simulated annealing)
                if np.random.random() < np.exp(-10 * (best_r - new_r) / max(step_size, 1e-10)):
                    best_pts = new_pts
                    best_r = new_r
            
            n_iter += 1
        
        return best_pts, best_r

    # Generate initial points
    pts = hex_init()
    
    # Optimize
    pts, r = optimize(pts, deadline)
    
    # Final check: ensure validity
    # Clip to valid range
    pts = np.clip(pts, -0.49, 0.49)
    
    # Verify no coincident points
    from scipy.spatial.distance import pdist
    min_d = pdist(pts).min()
    if min_d < 1e-10:
        # Perturb coincident points
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    pts[j] += np.random.normal(0, 1e-5, 2)
                    pts[j] = np.clip(pts[j], -0.49, 0.49)
    
    return pts
# EVOLVE-BLOCK-END
