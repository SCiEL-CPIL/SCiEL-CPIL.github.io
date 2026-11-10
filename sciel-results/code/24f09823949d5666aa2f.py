"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Hexagon geometry helpers
    def hex_vertices(cx, cy, s, angle_deg=0.0):
        """Return 6 vertices of regular hexagon with center (cx,cy), side s, rotation angle_deg."""
        angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3.0
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def point_in_hexagon(px, py, cx, cy, s, angle_deg=0.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s and rotation angle_deg."""
        # Transform to hexagon's local frame
        dx, dy = px - cx, py - cy
        a = np.radians(angle_deg)
        dx_l = dx * np.cos(-a) - dy * np.sin(-a)
        dy_l = dx * np.sin(-a) + dy * np.cos(-a)
        # For a regular hexagon with flat top (vertices at 0,60,120,180,240,300 deg),
        # the constraints in local frame (vertex at angle 0 is (s,0)):
        # |x| <= s, |x/2 + sqrt(3)/2 * y| <= s, |-x/2 + sqrt(3)/2 * y| <= s
        return (abs(dx_l) <= s + 1e-9 and 
                abs(dx_l/2.0 + np.sqrt(3)/2.0 * dy_l) <= s + 1e-9 and
                abs(-dx_l/2.0 + np.sqrt(3)/2.0 * dy_l) <= s + 1e-9)
    
    def hexagon_in_hexagon(inner_verts, outer_cx, outer_cy, outer_s, outer_angle_deg=0.0):
        """Check if all vertices of inner hexagon are inside outer hexagon."""
        for vx, vy in inner_verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_angle_deg):
                return False
        return True
    
    def min_separation_hexagons(v1, v2, s1, s2, a1, a2):
        """
        Compute minimum distance between two hexagons.
        If negative, they overlap. We compute the minimum over all vertex-edge pairs.
        For same-orientation hexagons on a lattice, this simplifies.
        Here we do a general check via vertices.
        """
        # For simplicity and speed, check if any vertex of one is inside the other
        # This is necessary but not sufficient for general polygons, but for convex
        # shapes with same orientation it's a good approximation.
        # Better: compute min distance between edges
        min_dist = float('inf')
        # Check vertices of v1 against edges of v2, and vice versa
        for i in range(6):
            p = v1[i]
            for j in range(6):
                # Distance from p to edge v2[j]-v2[(j+1)%6]
                a = v2[j]
                b = v2[(j+1) % 6]
                ab = b - a
                ap = p - a
                t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                d = np.linalg.norm(p - closest)
                if d < min_dist:
                    min_dist = d
            # Also check if p is inside v2
            if point_in_hexagon(p[0], p[1], *np.mean(v2, axis=0), s2, a2):
                return -min_dist  # Overlap
        for i in range(6):
            p = v2[i]
            for j in range(6):
                a = v1[j]
                b = v1[(j+1) % 6]
                ab = b - a
                ap = p - a
                t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                d = np.linalg.norm(p - closest)
                if d < min_dist:
                    min_dist = d
            if point_in_hexagon(p[0], p[1], *np.mean(v1, axis=0), s1, a1):
                return -min_dist
        return min_dist
    
    def compute_min_separation(positions, angles, side=1.0):
        """Compute minimum separation between all pairs of inner hexagons."""
        n = len(positions)
        min_sep = float('inf')
        for i in range(n):
            vi = hex_vertices(positions[i][0], positions[i][1], side, angles[i])
            for j in range(i+1, n):
                vj = hex_vertices(positions[j][0], positions[j][1], side, angles[j])
                sep = min_separation_hexagons(vi, vj, side, side, angles[i], angles[j])
                if sep < min_sep:
                    min_sep = sep
        return min_sep
    
    def outer_hex_side_for_positions(positions, angles, side=1.0, outer_angle=0.0):
        """Find minimum outer hexagon side length (same orientation as outer) to contain all inner hexagons."""
        # The outer hexagon must contain all vertices of all inner hexagons
        # For a hexagon centered at (cx,cy) with side s and angle a, a point (px,py) is inside iff:
        # max over 6 directions of (projection) <= s
        # The 6 normal directions for a hexagon with vertex at angle a are at a+30, a+90, ... 
        # Actually, for a regular hexagon, the 6 supporting lines are at angles a+30k for k=0..5
        # The distance from center to each supporting line is s * cos(30°) = s * sqrt(3)/2 ... 
        # Wait, let me reconsider.
        
        # A regular hexagon with side s, center at origin, one vertex at angle 0:
        # Vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5
        # The 6 edges have outward normals at angles 30°, 90°, 150°, 210°, 270°, 330°
        # The distance from center to each edge (apothem) = s * cos(30°) = s * sqrt(3)/2
        # So a point (x,y) is inside iff for each normal direction n_k: dot((x,y), n_k) <= s*sqrt(3)/2
        
        # For outer hexagon with center (cx,cy) and rotation angle_a:
        # Normals are at angles angle_a + 30° + k*60° for k=0..5
        # Constraint: dot((px-cx, py-cy), n_k) <= s_outer * sqrt(3)/2 for all k and all points
        
        # So s_outer >= max over all points p and all k of dot((p - c), n_k) / (sqrt(3)/2)
        
        # We also need to center the outer hexagon optimally.
        # For a given set of points, the minimal enclosing hexagon (fixed orientation) 
        # can be found by: s = max_k [ (max_p dot(p, n_k) - min_p dot(p, n_k)) / (2 * sqrt(3)/2) ]
        # and center at the midpoint of the projections.
        
        # Collect all vertices of all inner hexagons
        all_pts = []
        for i in range(len(positions)):
            vi = hex_vertices(positions[i][0], positions[i][1], side, angles[i])
            all_pts.append(vi)
        all_pts = np.vstack(all_pts)
        
        # Normal directions for outer hexagon
        base_angle = np.radians(outer_angle)
        normal_angles = base_angle + np.pi/6 + np.arange(6) * np.pi/3
        
        # For each normal direction, compute max and min projection
        max_proj = np.zeros(6)
        min_proj = np.zeros(6)
        for k in range(6):
            n = np.array([np.cos(normal_angles[k]), np.sin(normal_angles[k])])
            projs = all_pts @ n
            max_proj[k] = np.max(projs)
            min_proj[k] = np.min(projs)
        
        # The required side length for each pair of opposite faces
        # s_outer * sqrt(3)/2 >= (max_proj[k] - min_proj[k]) / 2 for each k
        # So s_outer >= (max_proj[k] - min_proj[k]) / (2 * sqrt(3)/2) = (max_proj[k] - min_proj[k]) / sqrt(3)
        
        s_required = np.max((max_proj - min_proj) / np.sqrt(3))
        
        # Center of outer hexagon: midpoint of projections for each direction
        # cx = (max_proj[k] + min_proj[k]) / 2 projected... actually:
        # The center should satisfy: dot(c, n_k) = (max_proj[k] + min_proj[k]) / 2 for all k
        # This is overdetermined, so we find the least-squares solution
        # Or simply: the center is the average of the "midpoints"
        
        # Actually, for a hexagon, the center is determined by the constraint that all 6 
        # face distances are equal. The optimal center minimizes the maximum face distance.
        # For simplicity, use the centroid of all points as initial center, then refine.
        
        # The center (cx, cy) should satisfy for each k:
        # dot((c), n_k) = (max_proj[k] + min_proj[k]) / 2
        # This gives 6 equations in 2 unknowns. Use least squares.
        A = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        b = (max_proj + min_proj) / 2.0
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return s_required, (center[0], center[1])
    
    # Initial configuration: hexagonal lattice cluster
    # 12 hexagons in a compact arrangement
    # Use a 3-4-3-2 row arrangement or similar
    # Hexagonal lattice: neighbors at distance sqrt(3) in 6 directions
    # Directions: 0°, 60°, 120°, 180°, 240°, 300°
    
    # Let's use a known compact arrangement:
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # Hexagonal lattice spacing: distance between centers of touching hexagons = sqrt(3)
    # For flat-top hexagons, horizontal spacing = sqrt(3), vertical spacing = 1.5 (for adjacent rows)
    
    # Actually, for unit hexagons with flat top:
    # Width (horizontal) = sqrt(3), height (vertical) = 2
    # Horizontal distance between centers in same row = sqrt(3)
    # Vertical distance between rows = 1.5
    # Offset between rows = sqrt(3)/2
    
    # Let me use pointy-top hexagons for easier lattice math
    # Pointy-top: width = sqrt(3), height = 2
    # Same as flat-top but rotated 30°
    
    # Let's just use a simple grid-like arrangement and let the optimizer fix it
    
    # Initial guess: place hexagons in a 4x3 grid on hexagonal lattice
    # Using axial coordinates
    positions0 = []
    angles0 = []
    
    # Hexagonal lattice positions for 12 hexagons
    # Use a compact cluster
    hex_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2),
        (-2, 0), (-2, 1)
    ]
    
    # Convert axial to Cartesian for pointy-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    for q, r in hex_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions0.append([x, y])
        angles0.append(0.0)  # Same orientation as outer
    
    positions0 = np.array(positions0)
    angles0 = np.array(angles0)
    
    # Verify initial configuration is valid
    min_sep_init = compute_min_separation(positions0, angles0)
    
    # Get initial outer hexagon size
    s_init, c_init = outer_hex_side_for_positions(positions0, angles0, outer_angle=0.0)
    
    best_positions = positions0.copy()
    best_angles = angles0.copy()
    best_s = s_init
    best_center = c_init
    
    # Objective function: minimize outer hexagon side length, with penalty for overlap
    def objective(params):
        # params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12]
        pos = np.array(params[0:3*N]).reshape(N, 2)
        ang = np.array(params[3*N:6*N])
        
        # Check minimum separation
        min_sep = compute_min_separation(pos, ang)
        
        # Penalty for overlap
        if min_sep < 0:
            penalty = 100.0 * (-min_sep) ** 2
        else:
            penalty = 0.0
        
        # Compute outer hexagon size
        s, c = outer_hex_side_for_positions(pos, ang, outer_angle=0.0)
        
        return s + penalty
    
    # Initial parameters
    x0 = np.concatenate([positions0.flatten(), angles0])
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if result.fun < best_s:
            best_params = result.x
            best_positions = np.array(best_params[0:3*N]).reshape(N, 2)
            best_angles = np.array(best_params[3*N:6*N])
            best_s, best_center = outer_hex_side_for_positions(best_positions, best_angles, outer_angle=0.0)
    except Exception:
        pass
    
    # Try a few more local optimizations from perturbed starts
    for restart in range(3):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + restart)
        perturbation = np.random.normal(0, 0.1, 6*N)
        x0_pert = x0 + perturbation
        
        try:
            result = minimize(objective, x0_pert, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
            
            if result.fun < best_s:
                best_params = result.x
                best_positions = np.array(best_params[0:3*N]).reshape(N, 2)
                best_angles = np.array(best_params[3*N:6*N])
                best_s, best_center = outer_hex_side_for_positions(best_positions, best_angles, outer_angle=0.0)
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, np.concatenate([best_positions.flatten(), best_angles]), 
                            method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                     'adaptive': True})
            if result.fun < best_s:
                best_positions = np.array(result.x[0:3*N]).reshape(N, 2)
                best_angles = np.array(result.x[3*N:6*N])
                best_s, best_center = outer_hex_side_for_positions(best_positions, best_angles, outer_angle=0.0)
        except Exception:
            pass
    
    # Build final answer
    inner = [[float(best_positions[i][0]), float(best_positions[i][1]), float(best_angles[i])] 
             for i in range(N)]
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_s), 0.0
# EVOLVE-BLOCK-END
