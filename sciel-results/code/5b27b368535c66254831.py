import time
import math
import numpy as np

N = 11
T0 = time.time()
DEADLINE = 145.0  # safety margin before 150s kill
SQRT3 = math.sqrt(3.0)
PI = math.pi
TWO_PI = 2.0 * PI


def _hex_vertices(cx, cy, theta):
    """Vertices of unit hexagon (circumradius=1) at center (cx,cy) rotated by theta."""
    ang = np.arange(6) * (PI / 3.0) + theta
    return np.column_stack([cx + np.cos(ang), cy + np.sin(ang)])


def _outer_hex_vertices(cx, cy, side, theta_outer):
    """Vertices of outer hexagon with center (cx,cy), side length 'side', rotated by theta_outer."""
    # circumradius = side
    ang = np.arange(6) * (PI / 3.0) + theta_outer
    return np.column_stack([cx + side * np.cos(ang), cy + side * np.sin(ang)])


def _point_in_hexagon(px, py, cx, cy, side, theta_outer):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to local coords
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_t = math.cos(-theta_outer)
    sin_t = math.sin(-theta_outer)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # For a regular hexagon with circumradius r, the apothem is r * sqrt(3)/2
    # The hexagon can be described by 6 half-planes
    # Normal directions: 0, 60, 120, 180, 240, 300 degrees (in local frame)
    # But the flat sides are perpendicular to the directions to vertices
    # Actually, for point-in-hexagon, we check distance to each of the 6 sides
    # Side normals point outward at angles: 30, 90, 150, 210, 270, 330 degrees
    # (i.e., halfway between vertex directions)
    for k in range(6):
        normal_angle = k * (PI / 3.0) + PI / 6.0
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        proj = lx * nx + ly * ny
        if proj > side * SQRT3 / 2.0:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(verts1, verts2, tol=1e-9):
    """Check if two hexagons are disjoint (no overlapping area, touching is ok)."""
    # Check if any vertex of hex1 is strictly inside hex2
    for v in verts1:
        inside = True
        # Check if v is inside convex polygon verts2
        # Use ray casting or check all edges
        n = len(verts2)
        for i in range(n):
            p1 = verts2[i]
            p2 = verts2[(i + 1) % n]
            # Check if v is on the wrong side of edge p1->p2
            d = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if d < -tol:
                inside = False
                break
        if inside:
            return False
    
    # Check if any vertex of hex2 is strictly inside hex1
    for v in verts2:
        inside = True
        n = len(verts1)
        for i in range(n):
            p1 = verts1[i]
            p2 = verts1[(i + 1) % n]
            d = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[1])
            if d < -tol:
                inside = False
                break
        if inside:
            return False
    
    # Check edge intersections
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        for j in range(n2):
            if _segments_intersect(verts1[i], verts1[(i+1)%n1], verts2[j], verts2[(j+1)%n2]):
                # Check if it's a proper intersection (not just touching at a point)
                # For simplicity, if segments cross, consider it invalid
                return False
    
    return True


def _all_disjoint(centers, angles, tol=1e-8):
    """Check if all N hexagons are pairwise disjoint."""
    verts_list = []
    for i in range(N):
        v = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        verts_list.append(v)
    
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(verts_list[i], verts_list[j], tol):
                return False
    return True


def _inner_in_outer(centers, angles, cx, cy, side, theta_outer, tol=1e-7):
    """Check if all inner hexagons are contained in outer hexagon."""
    for i in range(N):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, side, theta_outer):
                return False
    return True


def _min_outer_side_lp(centers, angles, theta_outer, cx=0.0, cy=0.0, tol=1e-9):
    """
    Find minimum side length of outer hexagon (centered at cx,cy, rotated theta_outer)
    that contains all inner hexagons.
    Uses linear programming: minimize s such that all vertices satisfy
    n_k . (v - center) <= s * apothem_factor for all 6 sides.
    
    Actually, for a hexagon with circumradius s, the distance from center to each side
    (apothem) is s * sqrt(3)/2.
    So constraint: n_k . (v - center) <= s * sqrt(3)/2 for each of 6 outward normals n_k.
    Minimize s.
    """
    from scipy.optimize import linprog
    
    # Collect all vertices
    all_verts = []
    for i in range(N):
        v = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (6N, 2)
    
    # Outer normals
    normal_angles = np.arange(6) * (PI / 3.0) + theta_outer + PI / 6.0
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each vertex v and each normal n_k:
    # n_k . (v - center) <= s * (sqrt(3)/2)
    # => -n_k . v + n_k . center <= s * (sqrt(3)/2)
    # => -n_k . v + n_k . center - s * (sqrt(3)/2) <= 0
    
    # We want to minimize s >= 0
    # Variables: [s, v_1, v_2, ..., v_{6N}] where v_k are vertex coordinates (fixed, but we include as vars for LP)
    # Actually, vertex coordinates are fixed. So the constraints are:
    # s * (sqrt(3)/2) >= n_k . (v - center) for all k, all v
    # => s >= (2/sqrt(3)) * n_k . (v - center)
    
    # So s >= max over all (k, v) of (2/sqrt(3)) * n_k . (v - center)
    
    min_s = 0.0
    for v in all_verts:
        for k in range(6):
            dot = normals[k, 0] * (v[0] - cx) + normals[k, 1] * (v[1] - cy)
            s_required = (2.0 / SQRT3) * dot
            if s_required > min_s:
                min_s = s_required
    
    return max(0.0, min_s)


def _compute_outer_for_positions(centers, angles, theta_outer, cx=0.0, cy=0.0):
    """Compute minimum outer side length for given inner positions."""
    return _min_outer_side_lp(centers, angles, theta_outer, cx, cy)


def construct():
    global T0
    
    # Strategy: discrete grid search over center positions and angles
    # Inner hexagons: centers on a fine grid, angles from a discrete set
    # Use greedy placement with pruning
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_theta_outer = 0.0
    
    # Outer rotation: try a few
    theta_outers = [0.0, PI/6, PI/3, PI/4, PI/2]
    
    # Inner angles: try a few discrete values
    angle_options = [0.0, PI/6, PI/3, PI/4, PI/12, PI/24]
    
    # Grid for centers: centered at origin, range [-3, 3]
    grid_step = 0.25
    grid_range = 3.5
    grid_vals = np.arange(-grid_range, grid_range + grid_step, grid_step)
    
    for theta_o in theta_outers:
        if time.time() - T0 > DEADLINE - 10:
            break
        
        for ang_inner in angle_options:
            if time.time() - T0 > DEADLINE - 10:
                break
            
            # Greedy placement: place hexagons one by one at best positions
            # Start with center at origin, then add others
            
            placed_centers = [np.array([0.0, 0.0])]
            placed_angles = [ang_inner]
            
            for hex_idx in range(1, N):
                if time.time() - T0 > DEADLINE - 5:
                    break
                
                best_pos = None
                best_dist = float('inf')
                
                # Candidate positions on grid
                for gx in grid_vals:
                    for gy in grid_vals:
                        cand_center = np.array([gx, gy])
                        
                        # Check against all placed hexagons
                        valid = True
                        min_sep = float('inf')
                        
                        for pc, pa in zip(placed_centers, placed_angles):
                            # Quick reject: center distance must be at least ~2
                            d = np.linalg.norm(cand_center - pc)
                            if d < 1.5:  # conservative lower bound
                                valid = False
                                break
                            min_sep = min(min_sep, d)
                        
                        if not valid:
                            continue
                        
                        # Check actual disjointness
                        cand_verts = _hex_vertices(gx, gy, ang_inner)
                        for pc, pa in zip(placed_centers, placed_angles):
                            pc_verts = _hex_vertices(pc[0], pc[1], pa)
                            if not _hexagons_disjoint(cand_verts, pc_verts):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Compute current outer side if we add this
                        test_centers = np.array(placed_centers + [cand_center])
                        test_angles = np.array(placed_angles + [ang_inner])
                        s_test = _compute_outer_for_positions(test_centers, test_angles, theta_o)
                        
                        # Prefer positions that minimize outer side, with some preference for compactness
                        score = s_test + 0.1 * min_sep
                        
                        if score < best_dist:
                            best_dist = score
                            best_pos = cand_center
                
                if best_pos is None:
                    # Try finer grid or skip this angle
                    break
                
                placed_centers.append(best_pos)
                placed_angles.append(ang_inner)
            
            if len(placed_centers) < N:
                continue
            
            centers_arr = np.array(placed_centers)
            angles_arr = np.array(placed_angles)
            
            # Verify all disjoint
            if not _all_disjoint(centers_arr, angles_arr):
                continue
            
            s = _compute_outer_for_positions(centers_arr, angles_arr, theta_o)
            
            if s < best_side:
                best_side = s
                best_centers = centers_arr.copy()
                best_angles = angles_arr.copy()
                best_theta_outer = theta_o
    
    # If we found a solution, refine with local search
    if best_centers is not None and time.time() - T0 < DEADLINE - 5:
        # Try small perturbations of the best solution
        for _ in range(20):
            if time.time() - T0 > DEADLINE - 3:
                break
            
            perturbed_centers = best_centers + np.random.normal(0, 0.1, size=(N, 2))
            perturbed_angles = best_angles + np.random.normal(0, 0.05, size=N)
            perturbed_angles = perturbed_angles % TWO_PI
            
            if not _all_disjoint(perturbed_centers, perturbed_angles):
                continue
            
            s = _compute_outer_for_positions(perturbed_centers, perturbed_angles, best_theta_outer)
            if s < best_side:
                best_side = s
                best_centers = perturbed_centers.copy()
                best_angles = perturbed_angles.copy()
    
    # Format output
    if best_centers is None:
        # Fallback: simple hexagonal grid
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(1, min(N, 7)):
            ang = (i - 1) * PI / 3.0
            centers[i] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(7, N):
            ang = (i - 7) * PI / 3.0 + PI / 6.0
            centers[i] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
        angles = np.zeros(N)
        theta_o = 0.0
        s = _compute_outer_for_positions(centers, angles, theta_o)
        best_centers = centers
        best_angles = angles
        best_theta_outer = theta_o
        best_side = s
    
    inner = []
    for i in range(N):
        x = float(best_centers[i, 0])
        y = float(best_centers[i, 1])
        ang_deg = float(math.degrees(best_angles[i]))
        inner.append((x, y, ang_deg))
    
    return inner, (0.0, 0.0), float(best_side), float(math.degrees(best_theta_outer))
