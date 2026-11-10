# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    # Precompute triangle indices for fast area calculation
    # There are C(13, 3) = 286 triangles
    tri_indices = list(itertools.combinations(range(n), 3))
    tri_indices = np.array(tri_indices)
    
    def compute_score(points):
        """
        Compute the ratio: min_triangle_area / convex_hull_area
        """
        # Check for degeneracy
        if np.any(~np.isfinite(points)):
            return 0.0
            
        # Calculate areas of all triangles
        # points[tri_indices] has shape (286, 3, 2)
        p = points[tri_indices]
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # Vectorized calculation:
        # x1 = p[:, 0, 0], y1 = p[:, 0, 1], etc.
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        
        areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        
        min_area = np.min(areas)
        
        if min_area < 1e-10:
            return 0.0
            
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume # For 2D, volume is area
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
            
        return min_area / hull_area

    def generate_initial_point():
        """
        Generate a structured initial configuration.
        Strategy: Place points on concentric rings to ensure good spread.
        Ring 0: 1 point at center? Or maybe distribute on 2-3 rings.
        For n=13:
        Option A: 1 center, 6 on ring 1, 6 on ring 2.
        Option B: 1 center, 5 on ring 1, 7 on ring 2.
        Let's try 1 center, 6 on inner ring (radius r1), 6 on outer ring (radius r2).
        This creates a hexagonal-like symmetry which is often good for packing.
        """
        pts = np.zeros((n, 2))
        
        # Center point
        pts[0] = [0.0, 0.0]
        
        # Inner ring: 6 points
        r1 = 0.5
        angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        pts[1:7] = np.c_[r1 * np.cos(angles1), r1 * np.sin(angles1)]
        
        # Outer ring: 6 points
        r2 = 1.0
        # Offset angles to avoid alignment with inner ring
        angles2 = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
        pts[7:13] = np.c_[r2 * np.cos(angles2), r2 * np.sin(angles2)]
        
        # Normalize to fit within unit circle roughly, or just use as is.
        # The metric is scale-invariant (ratio of areas), so scale doesn't matter, 
        # but numerical stability might. Let's keep coordinates around 1.
        return pts

    # Optimization Loop: Simulated Annealing / Hill Climbing
    current_points = generate_initial_point()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Temperature schedule
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.999
    
    # We will perform random moves: perturb one random point slightly
    # Or perturb all points slightly? Perturbing one point is faster and more local.
    
    max_steps = 50000 # Upper bound, but we are time-limited
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Generate a neighbor
        neighbor_points = current_points.copy()
        # Pick a random point to move
        idx = np.random.randint(n)
        # Move it by a small random amount
        # Step size can be proportional to T or fixed
        delta = np.random.normal(0, 0.1, size=2)
        neighbor_points[idx] += delta
        
        # Optional: Keep points within a bounding box to prevent drift?
        # The metric is scale invariant, so drift doesn't change score, 
        # but large coordinates might cause numerical issues in ConvexHull or area calc.
        # Let's re-center if drift is too large.
        if np.any(np.abs(neighbor_points) > 10):
            neighbor_points = neighbor_points - np.mean(neighbor_points, axis=0)
            
        neighbor_score = compute_score(neighbor_points)
        
        # Acceptance criterion
        if neighbor_score > current_score:
            current_points = neighbor_points
            current_score = neighbor_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new - old) / T)
            # Since we want to maximize, delta = new - old is negative.
            # prob = exp(delta / T)
            delta = neighbor_score - current_score
            if delta > 0:
                prob = 1.0
            else:
                prob = np.exp(delta / T)
            
            if np.random.random() < prob:
                current_points = neighbor_points
                current_score = neighbor_score
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If T is very low, we can increase step size or reset?
        # For simplicity, let's just let it run.
        
    return best_points

# EVOLVE-BLOCK-END
