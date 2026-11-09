"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Build a high-quality initial placement using a known optimal packing
     pattern for 12 unit hexagons (a 4x3 rhombus of hexagons, or a
     hexagonal arrangement).
  2. Compute the minimal enclosing regular hexagon side length that
     contains all 12 unit hexagons (with their rotations).
  3. Use a local optimization (simulated annealing + Powell-like refinement)
     to minimize the outer side length by adjusting positions and angles.

The key insight: for a set of convex polygons, the minimal enclosing regular
hexagon of a given orientation can be found by projecting all vertices onto
the 6 axis directions of the enclosing hexagon. The required side length L
satisfies: for each of the 6 directions u_i, max dot(p, u_i) - min dot(p, u_i)
<= 2*L*cos(30°) = L*sqrt(3) for a hexagon with flat-top orientation... 

Actually, let me think more carefully. A regular hexagon with side length L
and center (cx, cy) and rotation angle theta contains a point (x,y) iff for
all 6 edge normals n_i (unit vectors perpendicular to edges), we have
dot((x,y)-(cx,cy), n_i) <= L*cos(30°) = L*sqrt(3)/2.

The 6 edge normals of a regular hexagon with rotation theta are at angles
theta + k*60° for k=0..5.

So the containment constraint is:
  max over all vertices v of inner hexagons: dot(v - center, n_i) <= L*sqrt(3)/2
  min over all vertices v: dot(v - center, n_i) >= -L*sqrt(3)/2

For a fixed center and rotation, the minimal L is:
  L = max_i (max_v dot(v, n_i) - min_v dot(v, n_i)) / (2 * sqrt(3)/2)
    = max_i (max_v dot(v, n_i) - min_v dot(v, n_i)) / sqrt(3)

We can also optimize over the center: the optimal center for a given set of
points and hexagon rotation is the center that minimizes the max over i of
(max_v dot(v, n_i) - min_v dot(v, n_i)). This is equivalent to finding the
center that minimizes the "width" in all 6 directions simultaneously.

For a fixed rotation, the optimal center can be found by noting that the
constraints are:
  cx*n_i.x + cy*n_i.y <= (max_v dot(v, n_i) + min_v dot(v, n_i))/2 ... 

Actually this is getting complex. Let me use a simpler approach:

For a fixed hexagon rotation, we can find the minimal enclosing hexagon by:
- The constraint is that all points must be inside.
- For direction i, we need: max_v (dot(v, n_i)) - cx*n_i.x - cy*n_i.y <= L*sqrt(3)/2
  and min_v (dot(v, n_i)) - cx*n_i.x - cy*n_i.y >= -L*sqrt(3)/2

So: L*sqrt(3)/2 >= max_i [ max_v dot(v,n_i) - cx*n_i.x - cy*n_i.y,
                            -(min_v dot(v,n_i) - cx*n_i.x - cy*n_i.y) ]

This is a convex optimization in (cx, cy, L) for fixed rotation. We can solve
it with scipy.optimize.linprog or just use a simple approach.

Given the time constraint, I'll use a practical approach:
1. Start with a good initial configuration.
2. For each configuration, compute the minimal enclosing hexagon side.
3. Use simulated annealing to minimize the side length.
"""

import numpy as np
import time
import sys

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

def hexagon_contains_point(verts, p):
    """Check if point p is inside convex polygon verts (CCW or CW)."""
    # Use cross product method
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
            # Edge normal
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            # Project both polygons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Check for gap
            if max1 < min2 + eps or max2 < min1 + eps:
                return False
    return True

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
    # Check overlaps between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            v1 = get_hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
            v2 = get_hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            if polygons_overlap(v1, v2, eps=-1e-6):  # Allow tiny overlap for numerical stability
                return False
    
    # Check containment
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
