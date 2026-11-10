# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Strategy: Geometric Topology Search.
    Instead of optimizing 26 coordinates freely (which is highly non-convex), 
    we generate a set of candidate "topologies" (combinatorial structures) 
    known to be good for Heilbronn's problem (lattice clips, hexagonal rings).
    For each topology, we only optimize the global affine transformation 
    (specifically aspect ratio and rotation, or just scaling if we assume 
    optimal aspect ratio is 1 for symmetric shapes, but we allow aspect ratio 
    to deform the shape). 
    
    Actually, a more robust approach:
    1. Generate diverse initial structures based on geometric principles 
       (lattice, rings, grids).
    2. Use a "Simulated Annealing" or "Nelder-Mead" hybrid on the full 
       coordinate set, but initialized from these strong geometric priors.
    3. Crucially, we use a "Repulsive" force model: points repel each other 
       based on the smallest triangle area they form, effectively maximizing 
       the minimum area via a physics-based simulation. This is often more 
       robust than gradient-based methods for this specific non-smooth 
       objective.
    
    Let's implement a robust Simulated Annealing / Metropolis-Hastings 
    approach with geometric initialization. This handles the non-differentiability 
    of min() naturally.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin

    # Pre-compute triangle indices
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
        """
        if not np.all(np.isfinite(points)):
            return 0.0, 0.0, 0.0, False
        
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0, 0.0, A_hull, False
        except Exception:
            return 0.0, 0.0, 0.0, False

        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0, A_min, A_hull, True
            
        score = A_min / A_hull
        return score, A_min, A_hull, True

    def generate_lattice_clip(num_points, scale=1.0):
        """
        Generate points from a triangular lattice, clipped to a hexagon or circle.
        This is a very strong prior for Heilbronn's problem.
        """
        # Basis vectors for triangular lattice
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        
        # Generate a large grid of lattice points
        lattice_points = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                p = i * v1 + j * v2
                # Normalize to keep them in a reasonable range
                lattice_points.append(p * scale)
        
        lattice_points = np.array(lattice_points)
        
        # We want to select 'num_points' points that are "dense" in the center 
        # but spread out. A simple way: sort by distance to origin and take 
        # the closest 'num_points'? No, that creates a cluster.
        # Better: Sort by distance and take points that form a convex hull 
        # or just take the first N in a spiral?
        
        # Let's just take points within a radius R, and if there are more than N, 
        # randomly sample N from them. If less, expand R.
        
        # For N=13, a hexagonal cluster of radius ~2 (in lattice units) 
        # has 1 + 6 + 12 = 19 points. So radius 1 (7 points) is too small.
        # Radius 2 (19 points) is good. We will sample 13 from the 19.
        
        # Let's generate points in a hexagonal shape of "radius" k (number of steps)
        # Hexagon of radius k has 3k^2 + 3k + 1 points.
        # k=1: 7 points
        # k=2: 19 points
        # So we take the 19 points of the k=2 hexagon and randomly drop 6?
        # Or better, we can optimize the selection.
        
        # For now, let's generate the 19 points of the k=2 hexagon.
        # Indices (i,j) such that max(|i|, |j|, |i-j|) <= 2? 
        # Actually, the hexagonal norm is max(|x|, |y|, |x+y|) in dual basis?
        # Let's just use the standard axial coordinates.
        
        hex_points = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                # Check if inside hexagon of radius 2
                # Distance in hex grid: max(|q|, |r|, |q+r|) <= 2? 
                # Wait, the standard hex distance is (|q| + |r| + |q+r|)/2
                dist = (abs(q) + abs(r) + abs(q+r)) / 2
                if dist <= 2:
                    # Convert axial to cartesian
                    x = q + 0.5 * r
                    y = (np.sqrt(3)/2) * r
                    hex_points.append([x, y])
        
        hex_points = np.array(hex_points)
        
        # We have 19 points. We need 13.
        # Which 6 to remove? The ones on the boundary? 
        # Removing boundary points reduces the hull area, which might be good 
        # if it allows the remaining points to spread out more? 
        # Or removing interior points?
        
        # Let's try a few subsets:
        # 1. Remove 6 random boundary points
        # 2. Remove 6 specific symmetric points
        
        return hex_points

    def generate_ring_structure(radii):
        """
        Generate points in concentric rings.
        """
        points = []
        for r, count in radii:
            if count > 0:
                angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                # Add a small random rotation to break symmetry if needed?
                # For now, fixed.
                pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
                points.append(pts)
        
        if not points:
            return np.array([[0,0]])
            
        return np.vstack(points)

    def simulated_annealing(points_init, iterations=5000, start_temp=0.1, end_temp=0.001):
        """
        Optimize point positions using Simulated Annealing.
        Objective: Maximize A_min / A_hull.
        Move: Randomly perturb one point or all points slightly.
        """
        current_points = points_init.copy()
        current_score, _, _, is_valid = calculate_metrics(current_points)
        if not is_valid:
            # Try to fix by spreading out
            current_points = points_init + np.random.normal(0, 0.1, points_init.shape)
            current_score, _, _, is_valid = calculate_metrics(current_points)
            if not is_valid:
                return None, 0.0

        best_points = current_points.copy()
        best_score = current_score
        
        rng = np.random.default_rng(42)
        
        for i in range(iterations):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Cooling schedule
            temp = start_temp * (end_temp / start_temp) ** (i / iterations)
            
            # Generate a new candidate
            # Strategy: Perturb a random subset of points
            # Or perturb all points with small noise
            # Let's perturb 1 or 2 random points significantly, or all points slightly.
            
            candidate_points = current_points.copy()
            
            # Choose move type
            move_type = rng.random()
            if move_type < 0.5:
                # Perturb one random point
                idx = rng.integers(0, n)
                noise = rng.normal(0, 0.1, 2)
                candidate_points[idx] += noise
            elif move_type < 0.8:
                # Perturb all points slightly (global drift)
                noise = rng.normal(0, 0.05, (n, 2))
                candidate_points += noise
            else:
                # "Jitter" all points with larger step
                noise = rng.normal(0, 0.2, (n, 2))
                candidate_points += noise

            # Ensure points are finite
            if not np.all(np.isfinite(candidate_points)):
                continue

            candidate_score, _, _, is_valid = calculate_metrics(candidate_points)
            
            if not is_valid:
                candidate_score = 0.0
            
            # Metropolis criterion
            if candidate_score > current_score:
                current_points = candidate_points
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp((new - old) / temp)
                # Since we are maximizing, delta = new - old is negative.
                # If delta is very negative, prob is 0.
                delta = candidate_score - current_score
                if delta < 0:
                    # Avoid underflow
                    if delta / temp > -700:
                        prob = np.exp(delta / temp)
                    else:
                        prob = 0.0
                    if rng.random() < prob:
                        current_points = candidate_points
                        current_score = candidate_score

        return best_points, best_score

    # --- Generate Initial Structures ---
    
    structures = []
    
    # 1. Hexagonal Lattice Clip (19 points -> 13)
    hex_pts = generate_lattice_clip(13)
    # We have 19 points. Let's create a few variants by removing different points.
    # Center is (0,0). 
    # Let's just take the first 13? Or sort by distance and take the 13 "most central"?
    # Taking the 13 most central creates a small cluster. 
    # Taking a random subset of the 19 is better for diversity.
    
    # Variant 1: Random subset of 13 from the 19 lattice points
    rng_init = np.random.default_rng(101)
    for _ in range(5):
        indices = rng_init.choice(len(hex_pts), 13, replace=False)
        structures.append(hex_pts[indices])
        
    # Variant 2: Symmetric removal. Remove 3 pairs of opposite boundary points.
    # Identify boundary points (distance > 1.5 in lattice units?)
    # Let's just use the 7-point hexagon + 6 random interior? No, we have the 19-point set.
    
    # 2. Concentric Rings
    # 1 center, 6 inner ring, 6 outer ring
    structures.append(generate_ring_structure([(0, 1), (0.5, 6), (1.0, 6)]))
    # 1 center, 5 inner, 7 outer?
    structures.append(generate_ring_structure([(0, 1), (0.6, 5), (1.0, 7)]))
    # 1 center, 4 inner, 8 outer?
    structures.append(generate_ring_structure([(0, 1), (0.7, 4), (1.0, 8)]))
    
    # 3. Regular 13-gon (all on hull)
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    structures.append(np.column_stack([np.cos(angles_13), np.sin(angles_13)]))
    
    # 4. Grid-like
    # 4x4 grid minus corners? 
    # Let's try a 3x5 grid scaled down
    grid_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            grid_pts.append([i*0.5, j*0.5])
    grid_pts = np.array(grid_pts)
    # We have 25 points. Sample 13.
    for _ in range(3):
        indices = rng_init.choice(len(grid_pts), 13, replace=False)
        structures.append(grid_pts[indices])

    # --- Optimization Loop ---
    
    best_score = 0.0
    best_points = None
    
    # We have ~16 structures. 
    # SA is slow. Let's run SA on the top few diverse structures.
    # Or run a shorter SA on all, then polish the best.
    
    for i, start_pts in enumerate(structures):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Normalize initial points to unit hull area roughly
        # Center and scale
        centroid = np.mean(start_pts, axis=0)
        start_pts = start_pts - centroid
        max_dist = np.max(np.linalg.norm(start_pts, axis=1))
        if max_dist > 0:
            start_pts = start_pts / max_dist
            
        # Run SA
        # Iterations: 2000 is fast enough for a decent search in <1s
        refined_pts, score = simulated_annealing(start_pts, iterations=3000)
        
        if refined_pts is not None and score > best_score:
            best_score = score
            best_points = refined_pts
            
    # --- Polish Phase ---
    # If we found a good point, use L-BFGS-B with the smooth objective to polish.
    if best_points is not None and time.time() - start_time < time_limit - 20:
        
        def objective_smooth(points_flat, temp=0.05):
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
            
            if np.min(areas) < 1e-10:
                 return 1e6

            m = np.min(areas)
            z = -areas / temp
            max_z = np.max(z)
            if max_z > 700:
                 return 1e6
                 
            sum_exp = np.sum(np.exp(z - max_z))
            log_sum_exp = max_z + np.log(sum_exp)
            
            soft_min_area = -temp * log_sum_exp
            
            return -(soft_min_area / A_hull)

        try:
            res = minimize(
                objective_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                bounds=[(-10.0, 10.0)] * (n * 2),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            polished_pts = res.x.reshape((n, 2))
            polished_score, _, _, is_valid = calculate_metrics(polished_pts)
            
            if is_valid and polished_score > best_score:
                best_points = polished_pts
                best_score = polished_score
                
        except Exception:
            pass

    # --- Final Normalization and Return ---
    if best_points is not None:
        centroid = np.mean(best_points, axis=0)
        best_points = best_points - centroid
        max_dist = np.max(np.linalg.norm(best_points, axis=1))
        if max_dist > 0:
            best_points = best_points / max_dist
            
        final_score, _, _, is_valid = calculate_metrics(best_points)
        if is_valid and final_score > 0:
            return best_points
    
    # Fallback
    return generate_ring_structure([(0, 1), (0.5, 6), (1.0, 6)])

# EVOLVE-BLOCK-END
