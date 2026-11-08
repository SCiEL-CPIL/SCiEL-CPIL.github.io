import numpy as np
from scipy.optimize import minimize
import time
import math
import random
import itertools

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_edge_normals(verts):
    """Return outward normals of each edge of a hexagon (as numpy array of shape (6,2))."""
    normals = np.zeros((6, 2))
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # Outward normal: rotate edge 90 degrees clockwise (for CCW ordered vertices)
        # For CCW polygon, outward normal is (dy, -dx) normalized... wait, let me think.
        # If vertices are CCW, the interior is to the left of each edge.
        # So outward normal is to the right: (edge_y, -edge_x)
        n = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            normals[i] = n / norm
    return normals

def _min_enclosing_side_lp(positions, outer_angle=0.0):
    """
    Compute the minimum side length of a regular hexagon centered at origin with given angle
    that contains all given positions (as centers of unit hexagons).
    
    For a regular hexagon with side s and center at origin, angle a, the constraint for
    a point (px, py) to be inside is:
    For each of the 6 edge normals n_i, n_i · (px, py) <= s * apothem_factor
    
    Actually, let me think more carefully. A regular hexagon with side s has:
    - Apothem (distance from center to edge) = s * sqrt(3) / 2
    - The 6 edges are at distance apothem from center, in 6 directions
    
    For a hexagon with vertices at angles angle + i*pi/3, the edge normals point in
    directions angle + pi/6 + i*pi/3 (perpendicular to edges).
    
    The constraint is: for each edge normal direction n_i, n_i · p <= apothem = s * sqrt(3)/2
    
    So: s >= 2 / sqrt(3) * max_i(n_i · p) for all points p.
    
    But we need this for ALL points, so:
    s >= max over all points p and all normals n_i of (2/sqrt(3)) * (n_i · p)
    
    Wait, that's not quite right either. Let me reconsider.
    
    For a regular hexagon centered at origin with side s and rotation angle a:
    The vertices are at s * (cos(a + k*pi/3), sin(a + k*pi/3)) for k=0..5
    The edges connect adjacent vertices. The outward normal of the edge between
    vertex k and vertex k+1 points in direction a + pi/6 + k*pi/3.
    The distance from center to each edge is the apothem = s * sqrt(3) / 2.
    
    So a point p is inside iff for all k: n_k · p <= s * sqrt(3) / 2
    where n_k is the unit outward normal of edge k.
    
    Therefore, for all points to be inside:
    s * sqrt(3) / 2 >= max over all points p and all k of (n_k · p)
    
    So: s >= (2 / sqrt(3)) * max_{p, k} (n_k · p)
    
    But wait, we also need to account for the fact that the inner hexagons have
    size 1. The centers must be such that the entire inner hexagon is inside the outer.
    So for each inner hexagon with center c and angle a_inner, we need all its
    vertices to be inside the outer hexagon.
    
    Actually, for the enclosing hexagon to contain a unit hexagon with center c and
    angle a_inner, we need: for each edge normal n_k of the outer hexagon,
    n_k · c + (max over vertices v of inner hex of n_k · v - n_k · c) <= s * sqrt(3) / 2
    
    The max over vertices v of inner hex of n_k · v - n_k · c is the support function
    of the unit hexagon in direction n_k, which equals the apothem of the unit hexagon
    if n_k aligns with an edge normal, or something else otherwise.
    
    Actually, the support function of a regular hexagon with side 1 in any direction
    is at most 1 (the circumradius). More precisely, for a unit regular hexagon,
    the support function in direction u is:
    - If u aligns with a vertex direction: 1
    - If u aligns with an edge normal direction: sqrt(3)/2
    - In between: varies
    
    For simplicity and correctness, let me just check all vertices of all inner
    hexagons against the outer hexagon constraints.
    """
    angle = math.radians(outer_angle)
    # Compute the 6 outward normals of the outer hexagon
    normals = []
    for i in range(6):
        # Edge normal direction: angle + pi/6 + i*pi/3
        theta = angle + math.pi / 6 + i * math.pi / 3
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_val = 0.0
    for pos in positions:
        x, y, ang = pos
        # Get all vertices of this inner hexagon
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            for n in normals:
                dot = n[0] * v[0] + n[1] * v[1]
                if dot > max_val:
                    max_val = dot
    
    # s * sqrt(3) / 2 >= max_val
    # s >= 2 * max_val / sqrt(3)
    return 2.0 * max_val / math.sqrt(3)

def _hex_overlap(v1, v2, eps=1e-9):
    """
    Check if two hexagons (as vertex arrays) have overlapping interiors using SAT.
    Returns True if they overlap.
    """
    # SAT: check all 12 edge normals (6 from each hexagon)
    for i in range(6):
        # Get edge normal of hex1
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        
        # Project both hexagons onto this normal
        proj1_min = np.min(v1 @ n)
        proj1_max = np.max(v1 @ n)
        proj2_min = np.min(v2 @ n)
        proj2_max = np.max(v2 @ n)
        
        # Check for separation
        if proj1_max <= proj2_min + eps or proj2_max <= proj1_min + eps:
            return False
    
    # No separating axis found, so they overlap
    return True

def _check_validity(positions, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and mutually disjoint."""
    # Check all inside outer
    for pos in positions:
        x, y, ang = pos
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            # Check if v is inside outer hexagon
            if not _point_in_hex(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    
    # Check mutual disjointness
    inner_verts = [_hex_vertices(pos[0], pos[1], 1.0, pos[2]) for pos in positions]
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if _hex_overlap(inner_verts[i], inner_verts[j]):
                return False
    
    return True

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    angle = math.radians(angle_deg)
    # Transform to local frame
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2) scaled by side
    # The hexagon is defined by 6 half-planes
    # For each edge, check if point is on the correct side
    for i in range(6):
        theta = i * math.pi / 3
        # Edge normal in local frame
        theta_n = theta + math.pi / 6
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        # Constraint: nx * lx + ny * ly <= side * sqrt(3) / 2
        if nx * lx + ny * ly > side * math.sqrt(3) / 2 + 1e-9:
            return False
    return True

def _optimize_outer(positions):
    """Optimize the outer hexagon center and angle to minimize side length."""
    # Try different angles for the outer hexagon
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Compute bounding box of all inner hexagon vertices
    all_verts = []
    for pos in positions:
        x, y, ang = pos
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # Center should be near the centroid of all vertices
    centroid = np.mean(all_verts, axis=0)
    
    # Try a range of angles
    for angle_deg in range(0, 180, 15):
        # For this angle, compute the optimal center
        # The optimal center minimizes the maximum over all vertices of the support function
        # This is a convex optimization problem (minimize max of linear functions)
        
        # Use scipy to solve: minimize t subject to n_k · (c - v) <= t for all k, v
        # where n_k are the 6 normals, v are all vertices, t = s * sqrt(3) / 2
        
        angle = math.radians(angle_deg)
        normals = []
        for i in range(6):
            theta = angle + math.pi / 6 + i * math.pi / 3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        # We want to find center c and side s such that for all vertices v and all normals n:
        # n · (v - c) <= s * sqrt(3) / 2
        # i.e., n · v - n · c <= s * sqrt(3) / 2
        # i.e., s >= (2/sqrt(3)) * (n · v - n · c) for all n, v
        
        # This is equivalent to: minimize s subject to s * sqrt(3)/2 + n · c >= n · v for all n, v
        # Variables: c_x, c_y, s
        # Constraints: s * sqrt(3)/2 + n_x * c_x + n_y * c_y >= n_x * v_x + n_y * v_y for all n, v
        
        n_constraints = 6 * len(all_verts)
        
        def objective(c_params):
            return c_params[2]  # s
        
        def constraints(c_params):
            cx, cy, s = c_params
            cons = []
            for n in normals:
                for v in all_verts:
                    # s * sqrt(3)/2 + n · c >= n · v
                    # s * sqrt(3)/2 + n_x * cx + n_y * cy - n_x * v_x - n_y * v_y >= 0
                    val = s * math.sqrt(3) / 2 + n[0] * cx + n[1] * cy - (n[0] * v[0] + n[1] * v[1])
                    cons.append({'type': 'ineq', 'fun': lambda val=val: val})
            return cons
        
        # Initial guess
        x0 = np.array([centroid[0], centroid[1], 5.0])
        
        try:
            result = minimize(objective, x0, method='SLSQP', constraints=constraints,
                             options={'maxiter': 50, 'ftol': 1e-6})
            if result.success and result.fun < best_side:
                best_side = result.fun
                best_center = (result.x[0], result.x[1])
                best_angle = angle_deg
        except:
            pass
    
    if best_side == float('inf'):
        # Fallback: just use origin and angle 0
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = _min_enclosing_side_lp(positions, 0.0)
    
    return best_center, best_side, best_angle

def _try_arrangement(positions):
    """Try to optimize the given arrangement."""
    outer_center, outer_side, outer_angle = _optimize_outer(positions)
    
    # Check validity
    if _check_validity(positions, outer_center, outer_side, outer_angle):
        return positions, outer_center, outer_side, outer_angle
    
    # If invalid, try to fix by moving hexagons apart
    # This is a fallback
    return None

def _get_valid_fallback():
    """Return a valid but suboptimal arrangement."""
    # Place hexagons in a line with enough spacing
    positions = []
    for i in range(N):
        x = i * 3.0  # 3 units apart ensures no overlap
        y = 0.0
        ang = 0.0
        positions.append([x, y, ang])
    
    outer_center = (0.0, 0.0)
    outer_side = 20.0
    outer_angle = 0.0
    
    return positions, outer_center, outer_side, outer_angle

def _force_directed_relax(positions, max_iter=100):
    """
    Use force-directed relaxation to resolve overlaps between inner hexagons.
    For each pair of overlapping hexagons, compute the penetration depth
    and push them apart.
    """
    positions = [list(p) for p in positions]
    
    for iteration in range(max_iter):
        moved = False
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                
                if _hex_overlap(vi, vj):
                    # Compute the minimum translation vector to separate
                    # Use SAT to find the axis of minimum penetration
                    min_pen = float('inf')
                    min_axis = None
                    
                    for k in range(6):
                        # Check axis from hex i
                        p1 = vi[k]
                        p2 = vi[(k + 1) % 6]
                        edge = p2 - p1
                        n = np.array([edge[1], -edge[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12:
                            continue
                        n = n / norm
                        
                        proj_i_min = np.min(vi @ n)
                        proj_i_max = np.max(vi @ n)
                        proj_j_min = np.min(vj @ n)
                        proj_j_max = np.max(vj @ n)
                        
                        # Penetration depth
                        pen = min(proj_i_max, proj_j_max) - max(proj_i_min, proj_j_min)
                        if pen < min_pen:
                            min_pen = pen
                            # Direction to push: from j to i if i is "above" j on this axis
                            if proj_i_max < proj_j_max:
                                min_axis = n
                            else:
                                min_axis = -n
                    
                    if min_axis is not None and min_pen > 0:
                        # Push apart
                        push = min_axis * min_pen / 2.0
                        positions[i][0] += push[0]
                        positions[i][1] += push[1]
                        positions[j][0] -= push[0]
                        positions[j][1] -= push[1]
                        moved = True
        
        if not moved:
            break
    
    return [tuple(p) for p in positions]

def _simulate_annealing(positions, outer_center, outer_side, outer_angle, max_iter=5000, t_start=1.0, t_end=0.01):
    """
    Simulated annealing to optimize the arrangement.
    """
    best_positions = [list(p) for p in positions]
    best_score = outer_side
    current_positions = [list(p) for p in positions]
    current_score = outer_side
    
    for iteration in range(max_iter):
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (iteration / max_iter)
        
        # Generate a random perturbation
        new_positions = [list(p) for p in current_positions]
        idx = random.randint(0, len(new_positions) - 1)
        dx = random.gauss(0, t * 0.5)
        dy = random.gauss(0, t * 0.5)
        dang = random.gauss(0, t * 10)
        new_positions[idx][0] += dx
        new_positions[idx][1] += dy
        new_positions[idx][2] += dang
        
        # Compute new score
        nc, ns, na = _optimize_outer(new_positions)
        
        # Check validity
        if not _check_validity(new_positions, nc, ns, na):
            # Invalid configuration, penalize
            new_score = ns + 10.0
        else:
            new_score = ns
        
        # Accept or reject
        if new_score < current_score:
            current_positions = new_positions
            current_score = new_score
            if new_score < best_score:
                best_positions = [list(p) for p in new_positions]
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
                current_positions = new_positions
                current_score = new_score
    
    return best_positions, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Initial arrangement: hexagonal lattice pattern
    # For unit hexagons, minimum center distance for touching is sqrt(3)
    d = sqrt3
    dy_row = d * math.sqrt(3) / 2  # = 1.5
    
    positions = []
    # 4-3-4 pattern
    for i in range(4):
        positions.append([i * d - 1.5 * d, 0.0, 0.0])
    for i in range(3):
        positions.append([i * d - d, dy_row, 0.0])
    for i in range(4):
        positions.append([i * d - 1.5 * d, 2 * dy_row, 0.0])
    
    # First, try to get a valid arrangement via force-directed relaxation
    positions = _force_directed_relax(positions, max_iter=50)
    
    # Optimize outer hexagon
    outer_center, outer_side, outer_angle = _optimize_outer(positions)
    
    # Check validity
    if not _check_validity(positions, outer_center, outer_side, outer_angle):
        # Try to fix
        positions = _force_directed_relax(positions, max_iter=100)
        outer_center, outer_side, outer_angle = _optimize_outer(positions)
        if not _check_validity(positions, outer_center, outer_side, outer_angle):
            # Fallback
            return _get_valid_fallback()
    
    best_positions = [list(p) for p in positions]
    best_score = outer_side
    
    # Simulated annealing to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        max_iter = min(5000, int(remaining_time * 100))  # Rough estimate
        best_positions, best_score = _simulate_annealing(
            positions, outer_center, outer_side, outer_angle,
            max_iter=max_iter, t_start=0.5, t_end=0.01
        )
        # Recompute outer for best positions
        outer_center, outer_side, outer_angle = _optimize_outer(best_positions)
    
    # Final check
    if not _check_validity(best_positions, outer_center, outer_side, outer_angle):
        # Fallback
        return _get_valid_fallback()
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return immediately
        return best_positions, outer_center, outer_side, outer_angle
    
    # Try a few more random restarts if time permits
    random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a random initial arrangement
        rand_positions = []
        for i in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            ang = random.uniform(0, 60)
            rand_positions.append([x, y, ang])
        
        # Relax
        rand_positions = _force_directed_relax(rand_positions, max_iter=50)
        rc, rs, ra = _optimize_outer(rand_positions)
        
        if _check_validity(rand_positions, rc, rs, ra) and rs < best_score:
            best_positions = [list(p) for p in rand_positions]
            best_score = rs
            outer_center, outer_side, outer_angle = rc, rs, ra
    
    return best_positions, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
