import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    eps = 1e-7
    sqrt3 = np.sqrt(3)
    
    # Standard hexagonal lattice cluster: 11 hexagons
    # Ring 0: center
    # Ring 1: 6 hexagons at distance sqrt(3), angles 0,60,...,300
    # Ring 2: 4 hexagons at distance 3, angles 30,90,150,210
    # All inner hexagons use flat-top orientation (angle 30 deg means the hexagon's 
    # "pointy" direction is at 30 deg, but let's define vertices consistently)
    
    # For a regular hexagon of side 1 with center at origin and "pointy-top" orientation:
    # vertices at angles 0, 60, 120, 180, 240, 300 from the positive x-axis, 
    # at distance 1 (circumradius = side length for regular hexagon)
    # Actually, for a regular hexagon with side s, the circumradius is s.
    # Vertices: (s*cos(k*60°), s*sin(k*60°)) for k=0..5
    
    # Let's use a consistent definition:
    # A unit regular hexagon centered at (cx, cy) with rotation angle theta (degrees):
    # vertices at (cx + cos(theta + k*60°), cy + sin(theta + k*60°)) for k=0..5
    
    # The minimum distance between centers of two non-overlapping unit hexagons 
    # (same orientation) is sqrt(3) when they share an edge.
    
    # Initial configuration
    centers = np.zeros((11, 2))
    angles = np.full(11, 30.0)  # all rotated 30 degrees
    
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = k * 60.0
        centers[k+1, 0] = (sqrt3 + eps) * np.cos(np.deg2rad(ang))
        centers[k+1, 1] = (sqrt3 + eps) * np.sin(np.deg2rad(ang))
    
    # Ring 2: 4 hexagons at angles 30, 90, 150, 210
    ring2_angles = [30, 90, 150, 210]
    for i, ang in enumerate(ring2_angles):
        idx = 7 + i
        centers[idx, 0] = (3.0 + eps) * np.cos(np.deg2rad(ang))
        centers[idx, 1] = (3.0 + eps) * np.sin(np.deg2rad(ang))
    
    def get_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon centered at (cx,cy) with rotation angle_deg."""
        a = np.deg2rad(angle_deg)
        verts = []
        for k in range(6):
            va = a + k * np.pi / 3
            verts.append([cx + np.cos(va), cy + np.sin(va)])
        return np.array(verts)
    
    def check_disjoint(centers, angles, tol=1e-6):
        """Check if all hexagons are disjoint (no overlapping interiors).
        Returns True if valid, False otherwise."""
        n = len(centers)
        for i in range(n):
            vi = get_vertices(centers[i, 0], centers[i, 1], angles[i])
            for j in range(i+1, n):
                vj = get_vertices(centers[j, 0], centers[j, 1], angles[j])
                # Check if any vertex of vi is inside vj or vice versa
                # Simple check: compute minimum distance between edges
                # For speed, use a bounding box pre-check
                if np.max(vi[:,0]) < np.min(vj[:,0]) - tol or np.min(vi[:,0]) > np.max(vj[:,0]) + tol:
                    continue
                if np.max(vi[:,1]) < np.min(vj[:,1]) - tol or np.min(vi[:,1]) > np.max(vj[:,1]) + tol:
                    continue
                # Check point-in-polygon for vertices
                for p in vi:
                    if _point_in_convex_poly(p, vj, tol):
                        return False
                for p in vj:
                    if _point_in_convex_poly(p, vi, tol):
                        return False
                # Also check edge intersections
                if _edges_intersect(vi, vj, tol):
                    return False
        return True
    
    def _point_in_convex_poly(p, poly, tol=1e-7):
        """Check if point p is strictly inside convex polygon poly."""
        n = len(poly)
        sign = 0
        for i in range(n):
            a = poly[i]
            b = poly[(i+1) % n]
            cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
            if abs(cross) < tol:
                return False  # on boundary
            if sign == 0:
                sign = 1 if cross > 0 else -1
            elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                return False
        return True
    
    def _edges_intersect(vi, vj, tol=1e-7):
        """Check if any edge of vi intersects any edge of vj."""
        for i in range(6):
            a1 = vi[i]
            a2 = vi[(i+1) % 6]
            for j in range(6):
                b1 = vj[j]
                b2 = vj[(j+1) % 6]
                if _seg_intersect(a1, a2, b1, b2, tol):
                    return True
        return False
    
    def _seg_intersect(p1, p2, p3, p4, tol=1e-7):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def orient(a, b, c):
            val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if abs(val) < tol:
                return 0
            return 1 if val > 0 else -1
        
        d1 = orient(p3, p4, p1)
        d2 = orient(p3, p4, p2)
        d3 = orient(p1, p2, p3)
        d4 = orient(p1, p2, p4)
        
        if d1 != d2 and d3 != d4:
            return True
        return False
    
    def min_outer_side(centers, angles, cx, cy, rot_deg, max_iter=100):
        """Binary search for minimum outer side S such that all inner hexagons fit inside."""
        rot_rad = np.deg2rad(rot_deg)
        
        def fits(S):
            # Outer hexagon centered at (cx, cy), side S, rotation rot_rad
            # Vertices of outer hexagon
            outer_verts = []
            for k in range(6):
                va = rot_rad + k * np.pi / 3
                outer_verts.append([cx + S * np.cos(va), cy + S * np.sin(va)])
            outer_verts = np.array(outer_verts)
            
            # Check all inner vertices are inside outer hexagon
            for i in range(len(centers)):
                vi = get_vertices(centers[i, 0], centers[i, 1], angles[i])
                for p in vi:
                    if not _point_in_convex_poly(p, outer_verts, -1e-9):
                        # Allow points on boundary (use non-strict)
                        if _point_in_convex_poly_inclusive(p, outer_verts):
                            continue
                        return False
            return True
        
        def _point_in_convex_poly_inclusive(p, poly):
            n = len(poly)
            sign = 0
            for i in range(n):
                a = poly[i]
                b = poly[(i+1) % n]
                cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross < -1e-9:
                    if sign == 0:
                        sign = -1
                    elif sign > 0:
                        return False
                elif cross > 1e-9:
                    if sign == 0:
                        sign = 1
                    elif sign < 0:
                        return False
            return True
        
        lo, hi = 3.0, 6.0
        if not fits(hi):
            hi = 8.0
            if not fits(hi):
                return hi
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-7:
                break
        return hi
    
    # Objective function for optimization
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e9
        
        # params: [cx, cy, rot_deg, S]
        cx, cy, rot_deg, S = params
        if S < 0:
            return 1e9
        
        rot_rad = np.deg2rad(rot_deg)
        
        # Outer hexagon vertices
        outer_verts = []
        for k in range(6):
            va = rot_rad + k * np.pi / 3
            outer_verts.append([cx + S * np.cos(va), cy + S * np.sin(va)])
        outer_verts = np.array(outer_verts)
        
        # Check all inner vertices are inside outer hexagon
        violation = 0.0
        for i in range(len(centers)):
            vi = get_vertices(centers[i, 0], centers[i, 1], angles[i])
            for p in vi:
                # Compute max violation for point p against outer hexagon
                max_viol = _max_point_violation(p, outer_verts)
                violation += max(max_viol, 0.0)**2
        
        return S + 100.0 * violation
    
    def _max_point_violation(p, outer_verts):
        """Return how much point p violates the outer hexagon boundary (positive if outside)."""
        n = len(outer_verts)
        max_viol = 0.0
        for i in range(n):
            a = outer_verts[i]
            b = outer_verts[(i+1) % n]
            # Edge normal (inward)
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            # Ensure normal points inward
            center = np.mean(outer_verts, axis=0)
            mid = (a + b) / 2
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            # Signed distance from p to edge line (positive if on inward side)
            dist = np.dot(p - a, normal)
            if dist < 0:
                max_viol = max(max_viol, -dist)
        return max_viol
    
    # Initial guess
    all_pts = centers
    cx0, cy0 = np.mean(all_pts, axis=0)
    
    # Try multiple rotations for outer hexagon
    best_S = float('inf')
    best_params = None
    
    for rot_init in [0.0, 30.0, 15.0, 45.0]:
        if time.time() - start_time > DEADLINE - 5:
            break
        x0 = np.array([cx0, cy0, rot_init, 4.0])
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except:
            pass
    
    if best_params is None:
        # Fallback
        S = 4.0
        return [[c[0], c[1], 30.0] for c in centers], (cx0, cy0), S, 0.0
    
    cx_opt, cy_opt, rot_opt, S_opt = best_params
    S_opt += 1e-5  # safety margin
    
    inner = []
    for i in range(len(centers)):
        inner.append([centers[i, 0], centers[i, 1], angles[i]])
    
    return inner, (cx_opt, cy_opt), S_opt, rot_opt
