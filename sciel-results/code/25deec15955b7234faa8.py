# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Approach:
    1. Initialize with a heuristic layout: regular hexagon hull + hexagonal interior packing.
    2. Apply Lloyd's algorithm (Voronoi centroid iteration) to interior points to uniformize density.
    3. Optimize all points jointly using Nelder-Mead to maximize A_min / A_hull.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Initial configuration:
    # 6 hull points (regular hexagon)
    # 7 interior points (1 center + 6 ring)
    n_hull = 6
    n_int = 7
    
    # Hull: Regular hexagon
    angles_hull = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    hull_points = np.column_stack([np.cos(angles_hull), np.sin(angles_hull)])
    
    # Interior: 1 center + 6 in a ring
    int_points = np.zeros((n_int, 2))
    int_points[0] = [0.0, 0.0]
    angles_ring = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r_ring = 0.4
    int_points[1:] = np.column_stack([r_ring * np.cos(angles_ring), r_ring * np.sin(angles_ring)])
    
    points = np.vstack([hull_points, int_points])
    
    def lloyd_relax(pts, n_iter=50):
        """
        Perform Lloyd relaxation on interior points.
        Hull points are kept fixed to maintain the container shape.
        """
        pts = pts.copy()
        for _ in range(n_iter):
            try:
                hull = ConvexHull(pts)
                hull_indices = hull.vertices
                int_indices = np.setdiff1d(np.arange(len(pts)), hull_indices)
                
                if len(int_indices) == 0:
                    break
                    
                # Compute Voronoi diagram
                vor = Voronoi(pts)
                
                new_pts = pts.copy()
                for idx in int_indices:
                    # Get region for point idx
                    region_idx = vor.point_region[idx]
                    region = vor.regions[region_idx]
                    
                    if -1 in region or len(region) == 0:
                        continue
                        
                    # Get vertices of the Voronoi cell
                    cell_verts = vor.vertices[region]
                    
                    # Interior points have bounded Voronoi cells
                    centroid = np.mean(cell_verts, axis=0)
                    new_pts[idx] = centroid
                
                pts = new_pts
            except Exception:
                # If Voronoi fails or hull fails, break
                break
                
        return pts
    
    # Apply Lloyd relaxation
    points = lloyd_relax(points, n_iter=50)
    
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
        # Check for NaN or inf
        if not np.all(np.isfinite(pts)):
            return 1e10
        a_min = min_triangle_area(pts)
        a_hull = hull_area(pts)
        if a_hull <= 0:
            return 1e10
        return -(a_min / a_hull)

    # Optimize using Nelder-Mead
    x0 = points.flatten()
    
    # Run optimization
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
    
    points_opt = result.x.reshape(-1, 2)
    
    # Verify the result is valid
    if not np.all(np.isfinite(points_opt)):
        points_opt = points
    
    # Ensure the convex hull is well-defined
    try:
        hull = ConvexHull(points_opt)
    except:
        points_opt = points
        
    return points_opt


# EVOLVE-BLOCK-END
