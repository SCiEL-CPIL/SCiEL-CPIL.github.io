"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import linprog
import math

N = 50

# EVOLVE-BLOCK-START

def construct():
    # 1. Initialization: Hexagonal lattice
    # Create a dense grid and select N points
    pts = []
    spacing = 0.1
    for i in range(-15, 16):
        for j in range(-15, 16):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y <= 0.95:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) < N:
        # Fallback
        pts = np.random.rand(N, 2) * 2 - 1
        r = np.linalg.norm(pts, axis=1)
        mask = r > 0.9
        pts[mask] = pts[mask] / r[mask, None] * 0.9
    else:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
    
    # Approximate the unit circle with a polygon for LP constraints
    # 128 sides is plenty accurate
    M = 128
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Vertices of the approximating polygon
    poly_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Edges of the approximating polygon (as lines ax+by+c=0)
    # We need the inward normal for each edge
    # Edge from v_k to v_{k+1}
    # Normal n = (-(y_{k+1}-y_k), x_{k+1}-x_k) normalized?
    # Let's define constraints as n_i . x + c_i <= 0 for interior, or >= for exterior?
    # For Chebyshev center, we want distance to edge >= r.
    # If edge is defined by normal n (unit, pointing inward) and point p on edge:
    # n . (x - p) >= r  =>  n . x - n . p >= r => -n . x + n . p + r <= 0
    
    # Pre-compute circle polygon constraints
    # Vertices v_i
    # Edge i connects v_i and v_{i+1}
    # Midpoint m_i = (v_i + v_{i+1}) / 2
    # Normal n_i = (v_{i+1} - v_i) rotated 90 deg CCW? 
    # If vertices are CCW, the interior is to the left.
    # Vector e = v_{i+1} - v_i. Left normal is (-e_y, e_x).
    # This points inward for a CCW polygon enclosing the origin?
    # Let's check: v_0=(1,0), v_1=(cos, sin). e = (cos-1, sin).
    # Left normal: (-sin, cos-1). For small angle, cos-1 is negative.
    # This seems to point inward.
    
    circle_edges = []
    for i in range(M):
        v1 = poly_verts[i]
        v2 = poly_verts[(i + 1) % M]
        e = v2 - v1
        n = np.array([-e[1], e[0]])
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n = n / n_norm
            # Check if it points inward (towards origin)
            # n . v1 should be positive? Or n . origin (0) - n . v1 ?
            # Distance from origin to line: |n . v1|.
            # Since origin is inside, n . v1 should be positive if n points outward?
            # Let's just test with origin.
            # If n . v1 < 0, flip n.
            if np.dot(n, v1) < 0:
                n = -n
            circle_edges.append((n, np.dot(n, v1)))
    
    # 2. Iterative Optimization
    for step in range(50):
        # Compute Voronoi diagram
        try:
            vor = Voronoi(pts)
        except Exception:
            break
            
        new_pts = pts.copy()
        
        for i in range(N):
            # Get Voronoi region for point i
            # vor.regions is a list of lists of vertex indices.
            # Some regions might be -1 (infinite). We clip with circle, so all are bounded.
            region_indices = vor.regions[i]
            if len(region_indices) == 0 or -1 in region_indices:
                # Handle infinite regions by just clipping the infinite rays?
                # For simplicity, if infinite, we might skip or handle carefully.
                # With points inside a bounded area, and we will clip, 
                # Voronoi might still report infinite if points are on boundary.
                # We initialized inside, so should be ok.
                # If it fails, keep current point.
                continue
            
            # Get vertices of the Voronoi polygon
            # vor.vertices contains the coordinates.
            # The vertices in the region are not necessarily in order.
            # We need to order them to form a polygon.
            # A simple way: sort by angle around the centroid.
            
            verts = vor.vertices[region_indices]
            if len(verts) < 3:
                continue
            
            centroid = np.mean(verts, axis=0)
            angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            verts = verts[sorted_indices]
            
            # Now we have a convex polygon (Voronoi cell) defined by verts.
            # We need to intersect this polygon with the unit circle (approximated by M-gon).
            # Intersection of two convex polygons.
            # We can use Sutherland-Hodgman algorithm or similar.
            # Or, simpler: Generate the constraints from both polygons and solve LP.
            
            # Constraints for Voronoi polygon:
            # For each edge of the Voronoi polygon, create a half-plane constraint.
            vor_constraints = []
            for j in range(len(verts)):
                v1 = verts[j]
                v2 = verts[(j + 1) % len(verts)]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 0:
                    n = n / n_norm
                    # Ensure normal points inward (towards centroid)
                    if np.dot(n, centroid - v1) < 0:
                        n = -n
                    d = np.dot(n, v1)
                    # Constraint: n . x >= d  =>  -n . x + d <= 0
                    vor_constraints.append((n, d))
            
            # Combine Voronoi constraints and Circle constraints
            # Total constraints: A_ub x + b_ub <= 0
            # For Voronoi: -n . x + d <= 0  => A_row = -n, b = d
            # For Circle:  -n_c . x + d_c <= 0 => A_row = -n_c, b = d_c
            
            A_ub = []
            b_ub = []
            
            for n, d in vor_constraints:
                A_ub.append(-n)
                b_ub.append(d)
                
            for n_c, d_c in circle_edges:
                A_ub.append(-n_c)
                b_ub.append(d_c)
                
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Objective: Maximize r
            # Variables: x, y, r
            # Maximize r  =>  Minimize -r
            # c = [0, 0, -1]
            c = np.array([0.0, 0.0, -1.0])
            
            # Constraints:
            # 1. Polygon constraints:
            #    n . (x, y) >= d + r
            #    n_x x + n_y y - r >= d
            #    -n_x x - n_y y + r <= -d
            #    A_row = [-n_x, -n_y, 1], b = -d
            
            # 2. Circle constraints (approximated):
            #    Same form: -n_cx x - n_cy y + r <= -d_c
            
            # 3. r >= 0
            #    -r <= 0  => A_row = [0, 0, -1], b = 0
            
            A_full = []
            b_full = []
            
            for i_con in range(len(A_ub)):
                n_x = -A_ub[i_con, 0] # A_ub is -n, so n = -A_ub
                n_y = -A_ub[i_con, 1]
                d = b_ub[i_con]
                A_full.append([-n_x, -n_y, 1.0])
                b_full.append(-d)
                
            A_full.append([0.0, 0.0, -1.0])
            b_full.append(0.0)
            
            A_full = np.array(A_full)
            b_full = np.array(b_full)
            
            # Bounds: x, y in [-1, 1], r in [0, 1]
            bounds = [(-1, 1), (-1, 1), (0, 1)]
            
            try:
                res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
                if res.success:
                    new_pts[i] = res.x[:2]
                else:
                    # Fallback: keep current point
                    pass
            except Exception:
                pass
        
        pts = new_pts
        
        # Optional: Small random perturbation to escape local minima
        # pts += np.random.randn(N, 2) * 0.001
        # Project back to circle
        r = np.linalg.norm(pts, axis=1)
        mask = r > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / r[mask, None]
            
    return pts

# EVOLVE-BLOCK-END
