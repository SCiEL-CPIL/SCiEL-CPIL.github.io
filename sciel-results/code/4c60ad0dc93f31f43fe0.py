# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    Uses Delaunay-based local repair combined with soft-min optimization.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def calculate_score(points):
        """Calculate A_min / A_hull for a set of points."""
        try:
            if not np.all(np.isfinite(points)):
                return -np.inf
            hull = ConvexHull(points)
            a_hull = hull.volume
            if a_hull <= 1e-10:
                return -np.inf
            
            min_area = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        p1, p2, p3 = points[i], points[j], points[k]
                        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                        if area < min_area:
                            min_area = area
            return min_area / a_hull
        except Exception:
            return -np.inf
    
    def get_min_triangle_indices(points):
        """Find indices of the minimum-area triangle."""
        min_area = np.inf
        min_idx = (0, 1, 2)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        return min_idx, min_area
    
    def delaunay_repair(points, max_iter=200, step_size=0.01):
        """
        Delaunay-based local repair: iteratively move vertices incident to 
        minimum-area triangles toward the centroid of their adjacent triangles.
        """
        points = points.copy()
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
                
            try:
                # Compute Delaunay triangulation
                tri = Delaunay(points)
                
                # Find all triangle areas from Delaunay
                tri_areas = []
                for simplex in tri.simplices:
                    i, j, k = simplex
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    tri_areas.append((area, i, j, k))
                
                if not tri_areas:
                    break
                    
                # Sort by area, get the smallest few
                tri_areas.sort()
                
                # Get the minimum area triangle
                min_area, mi, mj, mk = tri_areas[0]
                
                # Find adjacent triangles for each vertex of the min triangle
                # For each vertex, find triangles that share that vertex
                vertex_adjacent = {mi: [], mj: [], mk: []}
                for area, i, j, k in tri_areas[1:]:  # Skip the min triangle itself
                    for v in [mi, mj, mk]:
                        if v in [i, j, k]:
                            centroid = (points[i] + points[j] + points[k]) / 3.0
                            vertex_adjacent[v].append((centroid, [i, j, k]))
                
                # Move each vertex of the min triangle toward the average centroid of adjacent triangles
                moved = False
                for v in [mi, mj, mk]:
                    if vertex_adjacent[v]:
                        # Average of adjacent centroids
                        avg_centroid = np.mean([c for c, _ in vertex_adjacent[v]], axis=0)
                        # Move toward centroid by step_size
                        direction = avg_centroid - points[v]
                        norm = np.linalg.norm(direction)
                        if norm > 1e-10:
                            points[v] += step_size * direction / norm
                            moved = True
                
                if not moved:
                    # No adjacent triangles to move toward; try small random perturbation
                    rng = np.random.default_rng(iteration)
                    for v in [mi, mj, mk]:
                        points[v] += rng.normal(0, 0.005, 2)
                    
            except Exception:
                break
        
        return points
    
    def soft_min_objective(x, temperature=0.01):
        """
        Soft-min surrogate: -log(sum(exp(-area_i/temperature))) / (temperature * n_triangles)
        This provides a smooth approximation to the minimum triangle area.
        """
        points = x.reshape((n, 2))
        if not np.all(np.isfinite(points)):
            return 1e6
        
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
            if a_hull <= 1e-10:
                return 1e6
        except Exception:
            return 1e6
        
        # Compute all triangle areas
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    areas.append(area)
        
        areas = np.array(areas)
        if len(areas) == 0 or np.any(areas < 0):
            return 1e6
        
        # Soft min: -T * log(sum(exp(-area/T)))
        # For numerical stability, subtract max
        max_neg = -np.max(areas)
        log_sum = np.log(np.sum(np.exp((areas + max_neg) / temperature))) + max_neg / temperature
        soft_min = -temperature * log_sum
        
        return -soft_min / a_hull  # Negate because minimize
    
    def init_regular_polygon():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack((x, y))
    
    def init_triangular_lattice():
        """Generate a triangular lattice configuration with 13 points."""
        pts = []
        for i in range(5):
            for j in range(5 - i):
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                pts.append([x, y])
        pts = np.array(pts)
        
        # Remove 2 points to get 13
        com = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - com, axis=1)
        indices_to_remove = np.argsort(dists)[:2]
        mask = np.ones(15, dtype=bool)
        mask[indices_to_remove] = False
        return pts[mask]
    
    # Generate multiple initial configurations
    inits = []
    inits.append(init_regular_polygon())
    
    # Perturbed regular polygons
    rng = np.random.default_rng(123)
    for _ in range(3):
        base = init_regular_polygon()
        base += rng.normal(0, 0.15, (n, 2))
        inits.append(base)
    
    # Triangular lattice
    inits.append(init_triangular_lattice())
    
    # More diverse inits
    for seed in [42, 99, 7]:
        rng2 = np.random.default_rng(seed)
        # Random points in unit circle
        angles = rng2.uniform(0, 2*np.pi, n)
        radii = rng2.uniform(0.3, 1.0, n)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts)
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Delaunay repair on each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        repaired = delaunay_repair(init, max_iter=100, step_size=0.01)
        score = calculate_score(repaired)
        
        if score > best_score:
            best_score = score
            best_points = repaired
    
    # Phase 2: Soft-min optimization from best repaired point
    if time.time() < deadline - 60 and best_points is not None:
        try:
            res = minimize(soft_min_objective, best_points.flatten(), 
                          method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            opt_points = res.x.reshape(n, 2)
            opt_score = calculate_score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception:
            pass
    
    # Phase 3: Another round of Delaunay repair on the optimized point
    if time.time() < deadline - 20 and best_points is not None:
        repaired2 = delaunay_repair(best_points, max_iter=150, step_size=0.005)
        score2 = calculate_score(repaired2)
        
        if score2 > best_score:
            best_score = score2
            best_points = repaired2
    
    # Phase 4: Fine-tuning with small Nelder-Mead
    if time.time() < deadline - 10 and best_points is not None:
        def fine_objective(x):
            points = x.reshape((n, 2))
            score = calculate_score(points)
            if score == -np.inf:
                return 1e6
            return -score
        
        try:
            res = minimize(fine_objective, best_points.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            fine_points = res.x.reshape(n, 2)
            fine_score = calculate_score(fine_points)
            
            if fine_score > best_score:
                best_score = fine_score
                best_points = fine_points
        except Exception:
            pass
    
    # Final verification
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = init_regular_polygon()
    
    return best_points

# EVOLVE-BLOCK-END
