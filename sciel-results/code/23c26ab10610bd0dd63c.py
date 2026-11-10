# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area 
    to the convex hull area.
    
    Strategy:
    1. Use a smooth surrogate objective (soft-min) to handle the non-differentiable 'min' operation.
    2. Employ Simulated Annealing (SA) for global optimization, which is robust to local minima 
       common in Heilbronn-type problems.
    3. Initialize with diverse structures, specifically including non-circular hulls (grids, 
       rectangles) as circular hulls are known to be suboptimal for maximizing this ratio.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # 30s margin from 630s limit
    
    # Pre-compute indices for all 286 triangles (i < j < k)
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def calculate_metrics(points):
        """
        Calculate A_min / A_hull and validity.
        Returns (score, A_min, A_hull, is_valid)
        """
        if not np.all(np.isfinite(points)):
            return 0.0, 0.0, 0.0, False
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0, 0.0, A_hull, False
        except Exception:
            return 0.0, 0.0, 0.0, False

        # Calculate min triangle area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        # Cross product magnitude in 2D
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0, A_min, A_hull, True
            
        score = A_min / A_hull
        return score, A_min, A_hull, True

    def soft_min(areas, T):
        """
        Smooth approximation of the minimum area.
        As T -> 0, this approaches min(areas).
        """
        # Use log-sum-exp trick for numerical stability
        # min(x) = -lim_{T->0} T * log(sum(exp(-x/T)))
        # To avoid overflow, shift by min
        min_a = np.min(areas)
        if min_a <= 0:
            return 0.0
        # exp(-(areas - min_a)/T) is <= 1, so no overflow
        val = -T * np.log(np.sum(np.exp(-(areas - min_a) / T)))
        return min_a - val

    def objective(points_flat, T):
        """
        Objective function to minimize (negative smooth score).
        """
        points = points_flat.reshape((n, 2))
        
        if not np.all(np.isfinite(points)):
            return 1e6
            
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 1e6
        except Exception:
            return 1e6

        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min_soft = soft_min(areas, T)
        
        if A_min_soft <= 1e-10:
            return 1e6

        score = A_min_soft / A_hull
        return -score

    def simulated_annealing(start_points, rng, time_budget):
        """
        Run Simulated Annealing from a specific starting point.
        """
        current_points = start_points.copy()
        current_score, _, _, _ = calculate_metrics(current_points)
        
        if current_score <= 0:
            # Try to fix invalid start by randomizing
            current_points = rng.uniform(-1, 1, (n, 2))
            current_score, _, _, _ = calculate_metrics(current_points)
            if current_score <= 0:
                return current_points, 0.0

        best_points = current_points.copy()
        best_score = current_score
        
        # Annealing parameters
        T_start = 0.5
        T_end = 0.001
        alpha = 0.999 # Cooling rate per step
        max_steps = 50000 # Max steps, but we will break on time
        
        step_size = 0.1
        
        for step in range(max_steps):
            if time.time() - start_time > time_budget:
                break
                
            T = T_start * (alpha ** step)
            if T < T_end:
                break
                
            # Generate neighbor
            # Randomly perturb a few points
            num_perturbed = rng.integers(1, 4)
            idxs = rng.choice(n, size=num_perturbed, replace=False)
            new_points = current_points.copy()
            new_points[idxs] += rng.normal(0, step_size, (num_perturbed, 2))
            
            # Evaluate
            new_score, _, _, is_valid = calculate_metrics(new_points)
            
            if not is_valid:
                continue
                
            if new_score > current_score:
                # Accept if better
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_points = current_points.copy()
                    best_score = current_score
            else:
                # Accept if worse with probability exp(-delta / T)
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
                else:
                    # Should not happen if new_score <= current_score, but just in case
                    current_points = new_points
                    current_score = new_score
            
            # Adjust step size slightly? Keep constant for simplicity, 
            # but maybe reduce it slowly
            step_size = max(0.01, 0.1 * (T / T_start))
            
        return best_points, best_score

    # --- Generate Initial Structures ---
    
    structures = []
    rng = np.random.default_rng(seed=123)
    
    # 1. Regular 13-gon (Circle) - known to be suboptimal but a good baseline
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_13 = np.column_stack([np.cos(angles), np.sin(angles)])
    structures.append(regular_13)
    
    # 2. Grid-like structure (Square/Rectangle hull) - likely better
    # 4x4 grid minus 3 points to get 13
    grid_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [1, 3]
    ], dtype=float)
    # Center and scale
    grid_pts = (grid_pts - np.mean(grid_pts, axis=0))
    structures.append(grid_pts)
    
    # 3. Hexagonal lattice fragment
    # Generate points in a hexagonal pattern
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Keep only 13 points closest to center
    dists = np.linalg.norm(hex_pts, axis=1)
    idxs = np.argsort(dists)[:13]
    hex_pts = hex_pts[idxs]
    hex_pts = (hex_pts - np.mean(hex_pts, axis=0))
    structures.append(hex_pts)
    
    # 4. Random perturbations of the grid
    for _ in range(5):
        perturbed_grid = grid_pts + rng.normal(0, 0.1, (n, 2))
        structures.append(perturbed_grid)
        
    # 5. Random points in a square
    for _ in range(5):
        pts = rng.uniform(-1, 1, (n, 2))
        structures.append(pts)

    # --- Optimization Loop ---
    
    best_score_global = 0.0
    best_points_global = None
    
    # Allocate time budget per structure
    # We have ~580s. 11 structures. ~50s each.
    time_per_struct = 50.0
    
    for i, start_pts in enumerate(structures):
        if time.time() - start_time > time_limit - 30: # Reserve 30s for final polish
            break
            
        # Run SA
        refined_pts, score = simulated_annealing(start_pts, rng, time.time() + time_per_struct)
        
        if refined_pts is not None and score > best_score_global:
            best_score_global = score
            best_points_global = refined_pts
            
    # --- Final Polish Phase ---
    # Use the best point found and run a shorter, tighter SA or local refinement
    if best_points_global is not None and time.time() - start_time < time_limit - 10:
        # Run SA again with smaller step sizes and lower T_start to polish
        # We can reuse the SA function but with tighter params
        # For simplicity, run SA again on the best point with a smaller time budget
        # and smaller initial step size
        # Let's just run SA again, it will converge faster if close to optimum
        polished_pts, polished_score = simulated_annealing(best_points_global, rng, time.time() + 20)
        
        if polished_pts is not None and polished_score > best_score_global:
            best_points_global = polished_pts
            best_score_global = polished_score

    # --- Normalization ---
    if best_points_global is not None:
        centroid = np.mean(best_points_global, axis=0)
        best_points_global = best_points_global - centroid
        max_dist = np.max(np.linalg.norm(best_points_global, axis=1))
        if max_dist > 0:
            best_points_global = best_points_global / max_dist
            
        # Final verification
        final_score, _, _, is_valid = calculate_metrics(best_points_global)
        if is_valid and final_score > 0:
            return best_points_global
    
    # Fallback: Return a regular 13-gon if everything failed
    return regular_13

# EVOLVE-BLOCK-END
