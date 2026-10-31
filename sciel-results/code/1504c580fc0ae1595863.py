"""
Voronoi-based circle packing in the unit square.

Strategy:
1. Represent the packing by N seed points (x, y).
2. Define the radius of each circle as the minimum distance from its seed to:
   - The boundaries of the unit square [0, 1]^2
   - The perpendicular bisectors between the seed and all other seeds (Voronoi edges)
   This guarantees that circles defined by these radii are disjoint (non-overlapping).
3. Maximize the sum of these radii by performing gradient ascent on the seed coordinates.
   The objective function is smooth almost everywhere (except at Voronoi vertex coincidences),
   allowing for efficient gradient-based optimization.
4. Use a multi-start strategy with random and structured initializations to escape local optima.
5. Use a deadline guard to return the best valid solution before the 150s limit.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radii(pts):
        """
        Compute the maximum non-overlapping radius for each seed point
        based on Voronoi distances.
        
        pts: (N, 2) array of seed coordinates.
        Returns: (N,) array of radii.
        """
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Distance to boundaries
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        
        # Distance to other seeds (Voronoi edge distance)
        # For each pair (i, j), the Voronoi edge is the perpendicular bisector.
        # The distance from seed i to the bisector is half the distance between i and j.
        # However, the Voronoi cell boundary for i is the locus of points equidistant to i and its nearest neighbor j.
        # The distance from i to this boundary is ||p_i - p_j|| / 2 ONLY IF j is the unique nearest neighbor.
        # Actually, the radius of the largest circle centered at p_i inside its Voronoi cell is the minimum
        # distance from p_i to any point in the boundary of the cell.
        # The boundary is formed by segments of perpendicular bisectors.
        # The distance from p_i to the bisector of (i, j) is ||p_i - p_j|| / 2.
        # The actual distance to the cell boundary is the minimum over all neighbors j of the distance to the bisector,
        # BUT we must also consider that the bisector segment might not extend to the closest point.
        # However, for a point inside its own Voronoi cell, the closest point on the cell boundary
        # is indeed on the bisector with one of the neighbors.
        # Specifically, dist(p_i, VoronoiBoundary_i) = min_j ( ||p_i - p_j|| / 2 )?
        # No. Consider 3 points forming an equilateral triangle.
        # The center of the Voronoi cell (the seed) is at distance h from the edges.
        # The distance to the neighbor is d. The bisector is at d/2.
        # Is the distance to the bisector d/2? Yes.
        # So, r_i = min( dist_to_boundary, min_j (dist(i,j)/2) )?
        # Let's verify. If I have two points, r1 = d/2, r2 = d/2. They touch. Correct.
        # If I have 3 points in a line: A---B---C.
        # Voronoi cell of B is bounded by bisector(A,B) and bisector(B,C).
        # Dist(B, bisector(A,B)) = d_AB/2.
        # Dist(B, bisector(B,C)) = d_BC/2.
        # So r_B = min(d_AB/2, d_BC/2, boundary_dist).
        # This logic holds: The largest circle centered at p_i that stays within the Voronoi cell
        # has radius equal to the minimum distance from p_i to the cell boundary.
        # Since the cell boundary is composed of bisectors, and the projection of p_i onto any bisector (i,j)
        # is the midpoint m_ij, the distance is ||p_i - m_ij|| = ||p_i - p_j||/2.
        # The closest point on the boundary is the minimum of these distances over all neighbors.
        
        # Calculate pairwise distances
        # To avoid O(N^2) memory if N was huge, but N=26 is small.
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Distance to bisector with j is dists[i,j] / 2
        # We want min over j != i of dists[i,j] / 2
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_dist_to_neighbor = np.min(dists, axis=1) / 2.0
        
        # The radius is the min of boundary distance and neighbor distance
        radii = np.minimum(r_bound, min_dist_to_neighbor)
        
        return radii

    def objective(pts):
        """
        Minimize negative sum of radii.
        """
        radii = compute_radii(pts)
        return -np.sum(radii)

    def gradient(pts):
        """
        Numerical gradient of the objective.
        Since the function is piecewise smooth, numerical gradient is robust.
        """
        eps = 1e-5
        grad = np.zeros_like(pts)
        base_val = objective(pts)
        
        for i in range(N):
            for k in range(2):
                pts_plus = pts.copy()
                pts_plus[i, k] += eps
                val_plus = objective(pts_plus)
                
                pts_minus = pts.copy()
                pts_minus[i, k] -= eps
                val_minus = objective(pts_minus)
                
                grad[i, k] = (val_plus - val_minus) / (2 * eps)
        
        return grad

    def objective_flat(pts_flat):
        """
        Objective function for scipy.optimize.minimize.
        Takes a flat array of 2*N coordinates, reshapes to (N,2),
        and returns negative sum of radii.
        """
        pts = pts_flat.reshape(N, 2)
        radii = compute_radii(pts)
        return -np.sum(radii)

    def optimize_from_start(start_pts, time_limit):
        """
        Perform optimization from a given start using scipy.optimize.minimize.
        Returns best pts and best val.
        """
        # Flatten points for minimize
        x0 = start_pts.flatten()
        
        # Define bounds: each coordinate in [0, 1]
        bounds = [(0.0, 1.0) for _ in range(2 * N)]
        
        best_pts = start_pts.copy()
        best_val = objective_flat(x0)
        
        try:
            # Use L-BFGS-B which is efficient for box-constrained problems
            res = minimize(
                objective_flat, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_pts = res.x.reshape(N, 2)
                
        except Exception:
            pass
            
        return best_pts, best_val

    # Generate initial candidates
    candidates = []
    
    # 1. Random starts
    for _ in range(30):
        pts = np.random.rand(N, 2)
        candidates.append(pts)
    
    # 2. Grid starts
    # 5x5 grid = 25 points, add 1 in center of a cell?
    # Or 6x5 grid = 30 points, take 26.
    # Let's try a few grid configurations.
    
    # Grid 5x5 (25 pts) + 1 random
    x_grid, y_grid = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    base_grid = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    for _ in range(5):
        extra_pt = np.random.rand(1, 2)
        pts = np.vstack([base_grid, extra_pt])
        # Shuffle to avoid bias in ordering? Not strictly necessary for Voronoi.
        candidates.append(pts)
        
    # Grid 6x5 (30 pts) -> take 26
    x_grid, y_grid = np.meshgrid(np.linspace(0.08, 0.92, 6), np.linspace(0.1, 0.9, 5))
    full_grid = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    # Randomly select 26
    for _ in range(5):
        idx = np.random.choice(30, 26, replace=False)
        candidates.append(full_grid[idx])
        
    # 3. Hexagonal-like start
    # Create a hexagonal lattice fragment
    # Spacing s
    s = 0.18
    hex_pts = []
    for i in range(6):
        for j in range(5):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * 0.866
            # Center and scale to fit in [0,1]
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Normalize to [0,1]
    hex_pts = (hex_pts - hex_pts.min(axis=0)) / (hex_pts.max(axis=0) - hex_pts.min(axis=0))
    # Add noise
    for _ in range(5):
        noisy_hex = hex_pts[:26] + np.random.normal(0, 0.02, (26, 2))
        noisy_hex = np.clip(noisy_hex, 0.0, 1.0)
        candidates.append(noisy_hex)

    best_val = np.inf
    best_result = None
    
    # Sort candidates by initial objective to try good ones first?
    # Or just iterate.
    # Time budget per candidate: 145s / len(candidates)
    # Let's give each candidate a fixed time slice, e.g., 2s, and iterate if time remains.
    # Or just run until deadline.
    
    num_candidates = len(candidates)
    # We have ~50 candidates. 145/50 ~ 2.9s each.
    # L-BFGS-B is fast, so we can afford more candidates or longer time.
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Time limit for this candidate: remaining time / remaining candidates
        # But we want to ensure we at least try all of them briefly.
        # Let's assign 2s per candidate initially.
        t_limit = min(time.time() + 2.0, deadline)
        
        try:
            pts_opt, val_opt = optimize_from_start(cand, t_limit)
            if val_opt < best_val:
                best_val = val_opt
                best_result = pts_opt
        except Exception:
            pass

    # If we have time left, do a final polish on the best result found
    if best_result is not None and time.time() < deadline - 5.0:
        # Run optimization again on the best result with more time
        t_limit = deadline
        try:
            pts_opt, val_opt = optimize_from_start(best_result, t_limit)
            if val_opt < best_val:
                best_val = val_opt
                best_result = pts_opt
        except Exception:
            pass

    if best_result is None:
        # Fallback: simple grid
        n_x, n_y = 6, 5
        r_val = 0.08
        x_coords = np.linspace(r_val, 1-r_val, n_x)
        y_coords = np.linspace(r_val, 1-r_val, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()
        y_f = y_f.flatten()
        best_result = np.column_stack((x_f[:N], y_f[:N]))

    # Calculate final radii for the best result
    radii = compute_radii(best_result)
    
    # Construct output: (x, y, radius)
    result = np.column_stack((best_result[:, 0], best_result[:, 1], radii))
    
    # Validation: Ensure no negative radii (should be handled by min with boundary dist, but just in case)
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    
    # Check validity: Are they really disjoint?
    # The Voronoi construction guarantees disjoint interiors IF the radii are computed correctly.
    # r_i = min(dist_to_boundary, min_j dist(i,j)/2)
    # dist(i, j) >= r_i + r_j ?
    # We know r_i <= dist(i,j)/2 and r_j <= dist(i,j)/2.
    # So r_i + r_j <= dist(i,j).
    # Thus, the circles are disjoint (or touching).
    # Boundary: r_i <= dist_to_boundary. So they are inside.
    
    return result

# EVOLVE-BLOCK-END
