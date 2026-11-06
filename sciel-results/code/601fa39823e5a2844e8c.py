import numpy as np
import time
from scipy.spatial import ConvexHull
import random

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s
    
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def repulsion_init(n, iterations=100, learning_rate=0.01):
        """
        Initialize points using a repulsion force to spread them evenly.
        """
        np.random.seed(42)
        points = np.random.randn(n, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        for _ in range(iterations):
            forces = np.zeros_like(points)
            # Calculate repulsive forces
            # dist[i, j] = ||points[i] - points[j]||
            # We want force on i from j to be proportional to (points[i] - points[j]) / dist^3
            # Using broadcasting
            diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            
            # Avoid division by zero
            dists[dist < 1e-10] = 1e-10
            # dists is NxN, we want to sum over j for each i
            # force[i] = sum_j (points[i] - points[j]) / dist[i,j]^3
            # This is O(N^2), which is fine for N=130
            inv_dist_cubed = 1.0 / (dists ** 3)
            inv_dist_cubed[np.diag_indices(n)] = 0.0
            
            # forces[i] = sum_j diffs[i,j] * inv_dist_cubed[i,j]
            forces = np.sum(diffs * inv_dist_cubed[:, :, np.newaxis], axis=1)
            
            # Update points
            points += learning_rate * forces
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # Decay learning rate
            learning_rate *= 0.99
            
        return points

    def calc_covering_radius(pts):
        """
        Calculate the covering radius by finding the maximum circumradius
        of the facets of the convex hull.
        Returns (radius, worst_simplex_indices)
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_simplex = None
            
            # Vectorized calculation of circumradii for all simplices
            # For each simplex (triangle), vertices v0, v1, v2
            # Circumradius R = (a*b*c) / (4*Area)
            # a, b, c are side lengths
            # Area = 0.5 * || (v1-v0) x (v2-v0) ||
            
            simplices = hull.simplices
            # Get vertices for all simplices
            # simplices shape (M, 3)
            v0 = pts[simplices[:, 0]]
            v1 = pts[simplices[:, 1]]
            v2 = pts[simplices[:, 2]]
            
            # Side lengths
            a = np.linalg.norm(v1 - v0, axis=1)
            b = np.linalg.norm(v2 - v1, axis=1)
            c = np.linalg.norm(v0 - v2, axis=1)
            
            # Cross product for area
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            # Avoid division by zero
            valid = areas > 1e-12
            radii = np.zeros(len(simplices))
            radii[valid] = (a[valid] * b[valid] * c[valid]) / (4.0 * areas[valid])
            
            max_rad = np.max(radii)
            worst_idx = np.argmax(radii)
            worst_simplex = simplices[worst_idx]
            
            return max_rad, worst_simplex
            
        except Exception:
            # Fallback: sample points if hull fails
            # This should rarely happen if points are in general position
            samples = np.random.randn(10000, 3)
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)
            dists = np.linalg.norm(samples[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            max_dist = np.max(min_dists)
            rad = 2.0 * np.arcsin(max_dist / 2.0)
            return rad, None

    # 1. Generate multiple initial candidates
    candidates = []
    
    # Candidate 1: Fibonacci Spiral
    p_fib = fibonacci_spiral(N)
    p_fib /= np.linalg.norm(p_fib, axis=1, keepdims=True)
    candidates.append(p_fib)
    
    # Candidate 2: Repulsion Init
    p_rep = repulsion_init(N, iterations=200, learning_rate=0.05)
    candidates.append(p_rep)
    
    # Candidate 3: Random Repulsion Init (different seed)
    np.random.seed(123)
    p_rep2 = repulsion_init(N, iterations=200, learning_rate=0.05)
    candidates.append(p_rep2)
    
    # Evaluate initial radii and pick the best
    best_points = None
    best_radius = float('inf')
    
    for p in candidates:
        r, _ = calc_covering_radius(p)
        if r < best_radius:
            best_radius = r
            best_points = p.copy()
            
    points = best_points.copy()
    current_radius = best_radius
    
    # 2. Local Search with Simulated Annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    step_size = 0.1
    stuck_counter = 0
    max_stuck = 100
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 2.0:
            break
            
        # Get the worst facet
        current_radius, worst_simplex = calc_covering_radius(points)
        
        if worst_simplex is None:
            # Degenerate case
            noise = np.random.randn(N, 3) * 0.01
            points = points + noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        # Calculate circumcenter of the worst facet
        tri = points[worst_simplex]
        v1, v2, v3 = tri
        c_vec = np.cross(v1 - v2, v2 - v3)
        c_norm = np.linalg.norm(c_vec)
        
        if c_norm < 1e-9:
            c_vec = (v1 + v2 + v3) / 3.0
            c_norm = np.linalg.norm(c_vec)
            
        if c_norm < 1e-9:
            break
            
        c_vec /= c_norm
        
        # Propose move: Move vertices towards circumcenter
        new_points = points.copy()
        for idx in worst_simplex:
            v = new_points[idx]
            new_v = v + step_size * (c_vec - v)
            new_v /= np.linalg.norm(new_v)
            new_points[idx] = new_v
            
        # Evaluate new radius
        new_radius, _ = calc_covering_radius(new_points)
        
        # Acceptance criterion (Simulated Annealing)
        delta = new_radius - current_radius
        if delta < 0 or random.random() < np.exp(-delta / T):
            points = new_points
            current_radius = new_radius
            stuck_counter = 0
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
            # Cool down
            T = max(T * cooling_rate, T_min)
        else:
            stuck_counter += 1
            
            # If stuck, try to escape local minimum
            if stuck_counter > max_stuck:
                # Increase step size or randomize
                if step_size < 0.5:
                    step_size *= 1.5
                else:
                    # Random perturbation
                    noise = np.random.randn(N, 3) * 0.05
                    points = points + noise
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    current_radius, _ = calc_covering_radius(points)
                    step_size = 0.1
                stuck_counter = 0
                T = 0.01 # Reheat
                
    return best_points
