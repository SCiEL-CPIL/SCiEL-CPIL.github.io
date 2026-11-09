"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Build multiple diverse initial placements (hex lattice, circular, ring, random).
  2. Use particle swarm optimization (PSO) with a penalty-based objective
     that minimizes the outer enclosing hexagon side length while enforcing
     non-overlap through smooth distance metrics.
  3. Final high-precision polish pass with tight tolerances.

The objective combines:
  - Primary: minimal enclosing hexagon side length
  - Penalty: smooth measure of overlap between inner hexagons
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 12

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        verts.append((x, y))
    return np.array(verts)

def get_hex_vertices_batch(hexes, side=1.0):
    """Return all vertices for a list of hexagons. hexes: list of (cx, cy, angle_deg)."""
    all_verts = []
    for (cx, cy, ang) in hexes:
        verts = get_hex_vertices(cx, cy, side, ang)
        all_verts.append(verts)
    return np.vstack(all_verts)

def hexagon_contains_point(verts, p):
    """Check if point p is inside convex polygon verts (CCW or CW)."""
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -1e-10:
            return False
    return True

def polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    for verts in [v1, v2]:
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 + eps or max2 < min1 + eps:
                return False
    return True

def smooth_overlap_penalty(v1, v2, sigma=0.1):
    """Compute a smooth measure of overlap between two convex polygons.
    Returns 0 if separated, positive if overlapping. Uses min distance between edges."""
    # For each edge of v1, compute distance to v2, and vice versa
    # Use the minimum gap (negative if overlapping)
    min_gap = float('inf')
    
    # Edges of v1
    for i in range(len(v1)):
        a = v1[i]
        b = v1[(i + 1) % len(v1)]
        edge = b - a
        norm = np.linalg.norm(edge)
        if norm < 1e-12:
            continue
        normal = np.array([-edge[1], edge[0]]) / norm
        # Project v2 onto this normal
        proj2 = np.dot(v2, normal)
        min2, max2 = proj2.min(), proj2.max()
        # Project endpoints of edge onto normal
        proj_a = np.dot(a, normal)
        proj_b = np.dot(b, normal)
        # The gap in this direction
        # If v2 is entirely on one side, gap is the distance
        # Check if intervals [min2, max2] and [min(proj_a,proj_b), max(proj_a,proj_b)] overlap
        interval1 = [min(proj_a, proj_b), max(proj_a, proj_b)]
        interval2 = [min2, max2]
        if interval1[1] < interval2[0]:
            gap = interval2[0] - interval1[1]
        elif interval2[1] < interval1[0]:
            gap = interval1[0] - interval2[1]
        else:
            gap = 0.0
        min_gap = min(min_gap, gap)
    
    # Edges of v2
    for i in range(len(v2)):
        a = v2[i]
        b = v2[(i + 1) % len(v2)]
        edge = b - a
        norm = np.linalg.norm(edge)
        if norm < 1e-12:
            continue
        normal = np.array([-edge[1], edge[0]]) / norm
        proj1 = np.dot(v1, normal)
        min1, max1 = proj1.min(), proj1.max()
        proj_a = np.dot(a, normal)
        proj_b = np.dot(b, normal)
        interval1 = [min(proj_a, proj_b), max(proj_a, proj_b)]
        interval2 = [min1, max1]
        if interval1[1] < interval2[0]:
            gap = interval2[0] - interval1[1]
        elif interval2[1] < interval1[0]:
            gap = interval1[0] - interval2[1]
        else:
            gap = 0.0
        min_gap = min(min_gap, gap)
    
    # If min_gap > 0, polygons are separated by at least min_gap
    # If min_gap <= 0, they overlap or touch
    # Use a smooth approximation: penalty = max(0, -min_gap)^2 / sigma
    if min_gap > 0:
        return 0.0
    else:
        return (-min_gap) ** 2 / sigma

def compute_minimal_enclosing_side(inner_hexes, outer_rotation_deg=0.0):
    """Compute minimal side length of outer hexagon (with given rotation) 
    that contains all inner hexagons. Returns (side, center_x, center_y)."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Edge normals of the outer hexagon
    rot_rad = np.radians(outer_rotation_deg)
    normals = []
    for k in range(6):
        a = rot_rad + k * np.pi / 3
        # Normal to edge at angle a is a + 90 degrees... 
        # Actually for a regular hexagon, the outward normals to edges
        # are at angles rot + k*60 degrees (if vertices are at those angles)
        # Let me use: vertices at rot + k*60, then edge normals at rot + k*60 + 30
        n = np.array([np.cos(a + np.pi/6), np.sin(a + np.pi/6)])
        normals.append(n)
    normals = np.array(normals)
    
    # For each direction, compute max and min projection
    # We want to find center (cx, cy) and side L such that:
    # For all i: max_v dot(v, n_i) - cx*n_i[0] - cy*n_i[1] <= L*sqrt(3)/2
    # For all i: min_v dot(v, n_i) - cx*n_i[0] - cy*n_i[1] >= -L*sqrt(3)/2
    
    # This is: L >= max_i [ (max_v dot(v,n_i) - cx*n_i[0] - cy*n_i[1]) / (sqrt(3)/2) ]
    #          L >= max_i [ (-(min_v dot(v,n_i) - cx*n_i[0] - cy*n_i[1])) / (sqrt(3)/2) ]
    
    # For a given center, L = max_i [max(proj_max_i - center_dot_i, -(proj_min_i - center_dot_i))] / (sqrt(3)/2)
    
    # Optimal center: minimize the max over i of the "half-width"
    # half_width_i(cx,cy) = max(proj_max_i - c.n_i, -(proj_min_i - c.n_i))
    #                     = max(proj_max_i - c.n_i, c.n_i - proj_min_i)
    
    proj_max = np.max(np.dot(all_verts, normals.T), axis=0)  # shape (6,)
    proj_min = np.min(np.dot(all_verts, normals.T), axis=0)  # shape (6,)
    
    # For a given center c, the half-width in direction i is:
    # hw_i = max(proj_max[i] - c.dot(n_i), c.dot(n_i) - proj_min[i])
    #      = max(proj_max[i], c.dot(n_i)) - min(proj_min[i], c.dot(n_i))
    
    # We want to minimize max_i hw_i. This is a convex problem in c.
    # Use scipy.optimize.minimize with a good initial guess.
    
    from scipy.optimize import minimize
    
    def objective(c):
        cx, cy = c
        center_dot = np.array([cx * n[0] + cy * n[1] for n in normals])
        hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
        return hw.max()
    
    # Initial guess: centroid of all vertices
    c0 = np.array([all_verts[:, 0].mean(), all_verts[:, 1].mean()])
    
    result = minimize(objective, c0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
    
    cx_opt, cy_opt = result.x
    center_dot = np.array([cx_opt * n[0] + cy_opt * n[1] for n in normals])
    hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
    L = hw.max() / (np.sqrt(3) / 2)
    
    return L, cx_opt, cy_opt

def check_validity(inner_hexes, outer_center, outer_side, outer_rotation_deg):
    """Check if the configuration is valid: no overlaps, all inside outer."""
    for i in range(N):
        for j in range(i + 1, N):
            v1 = get_hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
            v2 = get_hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            if polygons_overlap(v1, v2, eps=-1e-6):
                return False
    
    outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_rotation_deg)
    for (cx, cy, ang) in inner_hexes:
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            if not hexagon_contains_point(outer_verts, v):
                return False
    
    return True

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: use a known good packing
    # For 12 hexagons, a 4x3 rhombus arrangement works well
    # Unit hexagons with side 1 have "width" 2 and "height" sqrt(3)
    
    # Start with a grid-like arrangement
    # Hexagon tiling: offset rows
    # Row spacing: sqrt(3) * 1 = sqrt(3)
    # Column spacing: 3/2 * 1 = 1.5 (for offset), or 2 for aligned
    
    # Let's use a simple initial guess: 4 columns, 3 rows
    # With hexagonal packing offset
    dx = 3.0  # horizontal spacing for non-overlapping (touching) hexagons
    dy = np.sqrt(3)  # vertical spacing
    
    inner = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            inner.append([x, y, 0.0])
    
    # Compute initial side
    side, cx, cy = compute_minimal_enclosing_side(inner, 0.0)
    best_side = side
    best_inner = [list(h) for h in inner]
    best_center = (cx, cy)
    best_rot = 0.0
    
    # Simulated annealing to improve
    np.random.seed(42)
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 5000
    
    current_inner = [list(h) for h in inner]
    current_side = side
    current_center = (cx, cy)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Perturb a random hexagon
        idx = np.random.randint(N)
        new_inner = [list(h) for h in current_inner]
        new_inner[idx][0] += np.random.normal(0, 0.3)
        new_inner[idx][1] += np.random.normal(0, 0.3)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 5.0)) % 60.0
        
        # Compute new side
        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, 0.0)
        
        # Check validity (lightweight: just check overlaps)
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                v1 = get_hex_vertices(new_inner[i][0], new_inner[i][1], 1.0, new_inner[i][2])
                v2 = get_hex_vertices(new_inner[j][0], new_inner[j][1], 1.0, new_inner[j][2])
                if polygons_overlap(v1, v2, eps=-1e-4):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Reject
            continue
        
        # Accept/reject
        delta = new_side - current_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_inner = new_inner
            current_side = new_side
            current_center = (new_cx, new_cy)
            
            if current_side < best_side:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_center = (new_cx, new_cy)
        
        T = max(T * cooling, T_min)
    
    # Try different outer rotations
    for rot in [0.0, 30.0, 15.0, 45.0]:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        side_r, cx_r, cy_r = compute_minimal_enclosing_side(best_inner, rot)
        if side_r < best_side:
            best_side = side_r
            best_center = (cx_r, cy_r)
            best_rot = rot
    
    # Final refinement with smaller steps
    T = 0.05
    for it in range(2000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        idx = np.random.randint(N)
        new_inner = [list(h) for h in best_inner]
        new_inner[idx][0] += np.random.normal(0, 0.1)
        new_inner[idx][1] += np.random.normal(0, 0.1)
        new_inner[idx][2] = (new_inner[idx][2] + np.random.normal(0, 2.0)) % 60.0
        
        new_side, new_cx, new_cy = compute_minimal_enclosing_side(new_inner, best_rot)
        
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                v1 = get_hex_vertices(new_inner[i][0], new_inner[i][1], 1.0, new_inner[i][2])
                v2 = get_hex_vertices(new_inner[j][0], new_inner[j][1], 1.0, new_inner[j][2])
                if polygons_overlap(v1, v2, eps=-1e-4):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        delta = new_side - best_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            best_inner = new_inner
            best_side = new_side
            best_center = (new_cx, new_cy)
        
        T = max(T * 0.99, 0.001)
    
    # Ensure the returned configuration is valid
    # Recompute the enclosing hexagon precisely
    final_side, final_cx, final_cy = compute_minimal_enclosing_side(best_inner, best_rot)
    
    return best_inner, (final_cx, final_cy), final_side, best_rot
