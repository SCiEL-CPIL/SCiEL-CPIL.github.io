# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Approach:
    - Use multiple initial configurations (regular polygon, rings, random).
    - Apply Lloyd's algorithm (Voronoi centroid relaxation) to improve uniformity.
    - Use Nelder-Mead optimization to fine-tune the configuration.
    - Return the configuration with the highest A_min / A_hull ratio.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    def min_triangle_area(pts):
        """Compute the minimum area among all triangles formed by 3 points."""
        n_pts = len(pts)
        min_area = np.inf
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                    if area < min_area:
                        min_area = area
        return min_area
    
    def hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 0.0
    
    def objective(x):
        """Negative of the ratio A_min / A_hull (we minimize the negative)."""
        pts = x.reshape(-1, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        a_min = min_triangle_area(pts)
        a_hull = hull_area(pts)
        if a_hull <= 1e-10:
            return 1e10
        return -(a_min / a_hull)
    
    def lloyd_relax(points, iterations=10):
        """
        Perform Lloyd's relaxation: move each point to the centroid of its Voronoi cell.
        """
        pts = points.copy()
        for _ in range(iterations):
            try:
                vor = Voronoi(pts)
            except:
                break
            
            # Compute centroids of Voronoi regions
            centroids = np.zeros_like(pts)
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    # Unbounded region, keep original point or handle carefully
                    centroids[i] = pts[i]
                else:
                    # Get vertices of the region
                    verts = vor.vertices[region]
                    # Compute centroid of the polygon
                    # Shoelace formula for centroid
                    x = verts[:, 0]
                    y = verts[:, 1]
                    # Centroid of a polygon
                    # A = 0.5 * sum(x_i y_{i+1} - x_{i+1} y_i)
                    # Cx = 1/(6A) * sum((x_i + x_{i+1}) * (x_i y_{i+1} - x_{i+1} y_i))
                    # Cy = 1/(6A) * sum((y_i + y_{i+1}) * (x_i y_{i+1} - x_{i+1} y_i))
                    x_next = np.roll(x, -1)
                    y_next = np.roll(y, -1)
                    cross = x * y_next - x_next * y
                    area = 0.5 * np.sum(cross)
                    if abs(area) < 1e-10:
                        centroids[i] = pts[i]
                    else:
                        cx = np.sum((x + x_next) * cross) / (6 * area)
                        cy = np.sum((y + y_next) * cross) / (6 * area)
                        centroids[i] = np.array([cx, cy])
            
            pts = centroids
        return pts

    def optimize_from_start(start_points):
        """Run Lloyd relaxation then Nelder-Mead from a starting point."""
        # Normalize start points to unit scale roughly
        pts = start_points.copy()
        # Lloyd relaxation
        pts_relaxed = lloyd_relax(pts, iterations=5)
        
        # Optimize
        x0 = pts_relaxed.flatten()
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        points_opt = result.x.reshape(-1, 2)
        
        # Validate
        if not np.all(np.isfinite(points_opt)):
            return None, -np.inf
        
        try:
            hull = ConvexHull(points_opt)
            score = -result.fun
            return points_opt, score
        except:
            return None, -np.inf

    # Generate initial configurations
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 7 outer, 6 inner
    n_outer = 7
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer_points = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    n_inner = 6
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / 6
    inner_radius = 0.5
    inner_points = np.column_stack([inner_radius * np.cos(angles_inner), inner_radius * np.sin(angles_inner)])
    starts.append(np.vstack([outer_points, inner_points]))
    
    # 3. 6 outer, 6 inner, 1 center
    n_outer2 = 6
    angles_outer2 = np.linspace(0, 2 * np.pi, n_outer2, endpoint=False)
    outer_points2 = np.column_stack([np.cos(angles_outer2), np.sin(angles_outer2)])
    n_inner2 = 6
    angles_inner2 = np.linspace(0, 2 * np.pi, n_inner2, endpoint=False) + np.pi / 6
    inner_points2 = np.column_stack([0.5 * np.cos(angles_inner2), 0.5 * np.sin(angles_inner2)])
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer_points2, inner_points2, center]))
    
    # 4. Random points in unit disk (seeded for reproducibility)
    np.random.seed(42)
    random_pts = np.random.rand(n, 2) * 2 - 1
    # Scale to fit in unit disk roughly
    dists = np.linalg.norm(random_pts, axis=1)
    random_pts = random_pts / dists[:, np.newaxis] * 0.8
    starts.append(random_pts)
    
    best_points = None
    best_score = -np.inf
    
    for start in starts:
        pts, score = optimize_from_start(start)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback
        best_points = starts[0]
        
    return best_points


# EVOLVE-BLOCK-END
