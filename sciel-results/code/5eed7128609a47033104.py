# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from itertools import combinations


def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2.0


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the Heilbronn ratio: A_min / A_hull."""
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _delaunay_triangles(points):
    """Get Delaunay triangulation and return list of triangles."""
    try:
        tri = Delaunay(points)
        return tri.simplices
    except Exception:
        return None


def _circumradius(p1, p2, p3):
    """Compute circumradius of triangle p1, p2, p3."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    
    d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-12:
        return 0.0
    
    ux = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) + (cx**2 + cy**2)*(ay - by)) / d
    uy = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) + (cx**2 + cy**2)*(bx - ax)) / d
    
    return np.sqrt((ux - ax)**2 + (uy - ay)**2)


def _min_circumradius(points):
    """Compute the minimum circumradius among all Delaunay triangles."""
    triangles = _delaunay_triangles(points)
    if triangles is None:
        return 0.0
    
    min_r = np.inf
    for simplex in triangles:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        r = _circumradius(p1, p2, p3)
        if r < min_r:
            min_r = r
    return min_r


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area.
    
    Uses a constrained Delaunay triangulation approach where internal vertices
    are moved via Laplacian smoothing with repulsive forces to maximize the
    minimum circumradius of adjacent triangles, enforcing the farthest-point
    Voronoi property inherent to optimal Heilbronn configurations.
    """
    n = 13
    
    # Start with a better initial configuration: 
    # 9 points on a circle (hull) + 4 internal points arranged in a smaller square
    # This gives us internal points to optimize
    # Outer ring: 9 points
    outer_angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
    outer_points = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    # Inner points: 4 points in a small square pattern
    inner_r = 0.3
    inner_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    inner_points = inner_r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    points = np.vstack([outer_points, inner_points])
    
    # Normalize to unit area hull
    hull_area = _hull_area(points)
    if hull_area > 0:
        points = points / np.sqrt(hull_area)
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Try multiple random initial configurations
    rng_init = np.random.default_rng(seed=123)
    for trial in range(20):
        # Random perturbation of initial config
        trial_points = points + 0.05 * rng_init.standard_normal((n, 2))
        trial_score = _score(trial_points)
        if trial_score > best_score:
            best_score = trial_score
            best_points = trial_points.copy()
    
    def _get_hull_indices(points):
        """Get indices of points on the convex hull."""
        try:
            hull = ConvexHull(points)
            return set(hull.vertices)
        except Exception:
            return set(range(len(points)))
    
    def _laplacian_smooth(points, i, adj, weight=0.3):
        """Compute Laplacian smoothing displacement for point i."""
        pi = points[i]
        neighbors = list(adj.get(i, []))
        if not neighbors:
            return np.zeros(2)
        
        # Move toward centroid of neighbors
        centroid = np.mean(points[neighbors], axis=0)
        displacement = centroid - pi
        return weight * displacement
    
    def _repulsion_force(points, i, strength=1.0):
        """Compute repulsive force on point i from all other points."""
        force = np.zeros(2)
        pi = points[i]
        for j in range(len(points)):
            if i == j:
                continue
            diff = pi - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                dist_sq = 1e-12
            dist = np.sqrt(dist_sq)
            # Inverse square law repulsion
            force += strength * diff / (dist_sq * dist)
        return force
    
    def _spring_force(points, i, adj, rest_length=0.5, k=1.0):
        """Compute spring forces from Delaunay neighbors."""
        force = np.zeros(2)
        pi = points[i]
        for j in adj.get(i, []):
            diff = points[j] - pi
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            # Hooke's law - pull toward rest length
            force += k * (dist - rest_length) * diff / dist
        return force
    
    def _identify_worst_triangles(points, num_worst=5):
        """Identify the triangles with smallest circumradii."""
        triangles = _delaunay_triangles(points)
        if triangles is None:
            return []
        
        tri_data = []
        for simplex in triangles:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            r = _circumradius(p1, p2, p3)
            tri_data.append((r, simplex))
        
        tri_data.sort(key=lambda x: x[0])
        return [t[1] for t in tri_data[:num_worst]]
    
    # Iterative optimization with physics-based relaxation
    rng_opt = np.random.default_rng(seed=42)
    
    current_points = points.copy()
    current_score = best_score
    
    # Simulated annealing parameters
    T = 0.1
    T_min = 0.0005
    cooling_rate = 0.997
    max_iterations = 8000
    
    # Track convergence
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iterations):
        # Get current Delaunay structure
        triangles = _delaunay_triangles(current_points)
        if triangles is None:
            break
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for simplex in triangles:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        adj[i].add(j)
        
        # Get hull indices
        hull_idx = _get_hull_indices(current_points)
        
        # Choose a point to move - prefer internal points, especially those in worst triangles
        worst_tris = _identify_worst_triangles(current_points, num_worst=3)
        
        # Collect candidate points from worst triangles (prefer internal)
        candidates = []
        for tri in worst_tris:
            for idx in tri:
                if idx not in hull_idx:
                    candidates.append(idx)
        
        # If no internal points in worst triangles, pick random internal point
        internal_points = [i for i in range(n) if i not in hull_idx]
        if candidates:
            i = rng_opt.choice(candidates)
        elif internal_points:
            i = rng_opt.choice(internal_points)
        else:
            i = rng_opt.integers(0, n)
        
        # Compute forces for this point
        # 1. Laplacian smoothing (toward neighbor centroid)
        lap_force = _laplacian_smooth(current_points, i, adj, weight=0.2)
        
        # 2. Repulsion from all points
        rep_force = _repulsion_force(current_points, i, strength=0.3)
        
        # 3. Spring force from Delaunay neighbors
        spring_force = _spring_force(current_points, i, adj, rest_length=0.4, k=0.2)
        
        # Combined force
        total_force = lap_force + rep_force + spring_force
        
        # Add random perturbation for exploration
        noise = 0.02 * T * rng_opt.standard_normal(2)
        
        # Proposed move
        step_size = 0.08 * T
        new_points = current_points.copy()
        new_points[i] = current_points[i] + step_size * (total_force / (np.linalg.norm(total_force) + 1e-10) + noise)
        
        # Ensure point stays within reasonable bounds
        if np.linalg.norm(new_points[i]) > 5.0:
            new_points[i] = current_points[i]
        
        # Evaluate
        new_score = _score(new_points)
        
        # Accept if better or with probability based on temperature
        if new_score >= current_score or rng_opt.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # Early stopping if converged
        if no_improve_count > max_no_improve and T <= T_min * 10:
            break
    
    # Final local optimization using scipy on the worst triangles
    def _neg_score(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = best_points.flatten()
    result = minimize(_neg_score, x0, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
    
    if result.fun < -best_score:
        best_points = result.x.reshape(-1, 2)
        best_score = -result.fun
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
