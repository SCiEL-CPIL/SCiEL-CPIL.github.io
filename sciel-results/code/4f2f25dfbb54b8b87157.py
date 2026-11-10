# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the smallest triangle.
    
    Strategy:
    1. Fix the convex hull to a regular hexagon. It is known that for small N, 
       hexagonal or near-hexagonal hulls are often optimal or near-optimal for Heilbronn's problem.
       This reduces the degrees of freedom and stabilizes the hull area.
    2. Use a smooth soft-min (log-sum-exp) approximation for the minimum triangle area 
       in the objective function. This makes the landscape differentiable, allowing 
       L-BFGS-B to use gradient information effectively.
    3. Multi-start optimization from diverse initial configurations (grid, random, perturbed hex).
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin

    # Pre-compute indices for all 286 triangles
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

    def objective_smooth(points_flat, temp=0.05):
        """
        Smooth objective function: minimize -(soft_min_area / A_hull).
        We use a log-sum-exp approximation for the minimum to make it differentiable.
        soft_min(x) = -1/T * log(sum(exp(-x_i/T)))
        As T -> 0, this approaches min(x).
        """
        points = points_flat.reshape((n, 2))
        
        # Basic validity check: ensure points are finite and not all same
        if not np.all(np.isfinite(points)):
            return 1e6
            
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 1e6
        except Exception:
            return 1e6

        # Calculate Triangle Areas
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        # Cross product magnitude
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # If any area is 0, penalty
        if np.min(areas) < 1e-10:
             return 1e6

        # Log-sum-exp for soft minimum
        # To avoid overflow in exp, we shift by the max (or min, since we want min)
        # We want min(areas). 
        # Let f = -1/T * log(sum(exp(-areas/T)))
        # For numerical stability, let m = min(areas)
        # sum(exp(-(areas-m)/T)) = exp(m/T) * sum(exp(-(areas-m)/T))
        # log(sum(exp(-areas/T))) = m/T + log(sum(exp(-(areas-m)/T)))
        # f = -1/T * (m/T + log(...)) = -m/T^2 - ... 
        # Wait, the standard softmin is: -1/T * log(sum(exp(-x_i/T)))
        # Let's just use the stable version directly.
        
        m = np.min(areas)
        # If m is very small, the gradient is huge, which is good for pushing it up, 
        # but we need to be careful with overflow.
        
        # Stable computation of softmin
        # We want S = -1/T * log( sum( exp(-areas_i / T) ) )
        
        # Let z_i = -areas_i / T
        # max_z = max(z_i) = -min(areas_i)/T = -m/T
        # sum(exp(z_i)) = exp(max_z) * sum(exp(z_i - max_z))
        # log(sum(exp(z_i))) = max_z + log(sum(exp(z_i - max_z)))
        
        # Since T is small, m/T can be large.
        # However, areas are typically O(1) in our normalized space.
        # Let's just compute it directly with clipping to avoid inf.
        
        z = -areas / temp
        max_z = np.max(z)
        if max_z > 700: # Prevent overflow
             return 1e6
             
        sum_exp = np.sum(np.exp(z - max_z))
        log_sum_exp = max_z + np.log(sum_exp)
        
        soft_min_area = -temp * log_sum_exp
        
        # We want to maximize soft_min_area / A_hull
        # So we minimize - (soft_min_area / A_hull)
        
        # Add a small penalty if points go far outside the initial hull to prevent drift
        # (Optional, but helps stability)
        
        return -(soft_min_area / A_hull)

    def optimize_from_start(start_points, max_iter=300):
        """
        Run L-BFGS-B from a specific starting point using the smooth objective.
        """
        if time.time() - start_time > time_limit:
            return None, 0.0
            
        start_flat = start_points.flatten()
        
        # Bounds: Keep points within a reasonable box.
        # Since we normalize later, we can allow a wide range.
        bounds = [(-10.0, 10.0)] * (n * 2)
        
        try:
            res = minimize(
                objective_smooth, 
                start_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            best_points = res.x.reshape((n, 2))
            score, _, _, is_valid = calculate_metrics(best_points)
            if is_valid and score > 0:
                return best_points, score
        except Exception:
            pass
            
        return None, 0.0

    # --- Generate Initial Structures ---
    
    structures = []
    
    # 1. Regular Hexagon Hull + Interior Points
    # The hull is fixed to a regular hexagon of radius 1.
    # Vertices:
    angles_hex = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_verts = np.column_stack([np.cos(angles_hex), np.sin(angles_hex)])
    
    # We need 13 points total. 6 are on the hull. 7 are inside.
    # Let's place the 7 interior points in a pattern.
    
    # Structure A: 7 points in a smaller hexagon + 1 center? 
    # Or a grid clipped to hexagon?
    
    # Let's try: 6 hull points, and 7 interior points arranged in a "flower" or grid.
    # Simple interior: regular 7-gon scaled down?
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_7 = 0.5 * np.column_stack([np.cos(angles_7), np.sin(angles_7)])
    structures.append(np.vstack([hex_verts, inner_7]))
    
    # Structure B: 6 hull points, 7 interior points in a 3x3 grid (centered)
    # Grid points:
    grid_coords = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            grid_coords.append([i*0.4, j*0.4])
    grid_9 = np.array(grid_coords)
    # We only need 7. Take 7 of them? Or use a different pattern.
    # Let's just take the first 7 from a larger set or use a specific pattern.
    # Let's use 7 points: center + 6 surrounding?
    center = np.array([[0.0, 0.0]])
    angles_6_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring_6 = 0.5 * np.column_stack([np.cos(angles_6_inner), np.sin(angles_6_inner)])
    interior_7 = np.vstack([center, ring_6])
    structures.append(np.vstack([hex_verts, interior_7]))
    
    # Structure C: Perturbed Hexagon
    rng = np.random.default_rng(seed=123)
    for _ in range(5):
        # Perturb the interior points
        int_pts = np.vstack([center, ring_6]) + rng.normal(0, 0.1, (7, 2))
        structures.append(np.vstack([hex_verts, int_pts]))
        
    # Structure D: Grid-like initialization scaled to fit in hexagon
    # 4x4 grid minus corners?
    # Let's generate random points in the hexagon
    for _ in range(5):
        # Rejection sampling for points in hexagon
        pts_in = []
        while len(pts_in) < 7:
            x, y = rng.uniform(-1, 1, 2)
            # Check if inside hexagon (radius 1)
            # Hexagon inequality: |x| <= 1, |y| <= 1, |x+y| <= 1.5? 
            # Actually, for regular hexagon with vertices at (1,0), (0.5, .866)...
            # Distance to origin <= 1 is circle. Hexagon is larger.
            # Let's just use random points in circle for simplicity, they will be inside hexagon.
            if np.sqrt(x*x+y*y) <= 0.9:
                pts_in.append([x, y])
        int_pts = np.array(pts_in)
        structures.append(np.vstack([hex_verts, int_pts]))

    # Structure E: "Hexagonal Lattice" snippet
    # Points in a triangular lattice
    # Basis: v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
    # Scale to fit in hexagon
    # Generate lattice points and keep those inside hexagon
    pts_lattice = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = (np.sqrt(3)/2) * j
            if np.sqrt(x*x+y*y) <= 0.9:
                pts_lattice.append([x, y])
    # We have many, let's pick 7 that are spread out.
    # Or just take the first 7.
    if len(pts_lattice) >= 7:
        # Shuffle and take 7
        rng.shuffle(pts_lattice)
        int_pts = np.array(pts_lattice[:7])
        structures.append(np.vstack([hex_verts, int_pts]))

    # --- Optimization Loop ---
    
    best_score = 0.0
    best_points = None
    
    for i, start_pts in enumerate(structures):
        if time.time() - start_time > time_limit - 60:
            break
            
        # Refine this structure
        refined_pts, score = optimize_from_start(start_pts)
        
        if refined_pts is not None and score > best_score:
            best_score = score
            best_points = refined_pts
            
            # Optional: Polish with smaller temperature
            # The objective function uses a fixed temp. 
            # For polishing, we could run another optimization with a smaller temp.
            # But L-BFGS-B might be stuck. Let's just do one more pass with tighter tol.
            if time.time() - start_time < time_limit - 30:
                 polished_pts, polished_score = optimize_from_start(refined_pts, max_iter=100)
                 if polished_pts is not None and polished_score > best_score:
                     best_points = polished_pts
                     best_score = polished_score

    # --- Normalization ---
    if best_points is not None:
        centroid = np.mean(best_points, axis=0)
        best_points = best_points - centroid
        max_dist = np.max(np.linalg.norm(best_points, axis=1))
        if max_dist > 0:
            best_points = best_points / max_dist
            
        # Final verification
        final_score, _, _, is_valid = calculate_metrics(best_points)
        if is_valid and final_score > 0:
            return best_points
    
    # Fallback: Return a regular hexagon + interior points
    return np.vstack([hex_verts, interior_7])

# EVOLVE-BLOCK-END
