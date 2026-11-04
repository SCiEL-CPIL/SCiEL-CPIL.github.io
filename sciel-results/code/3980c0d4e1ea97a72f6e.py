"""
Strategy: Simulated Annealing for Direct Geometric Packing.

This approach abandons the topological constraints (fixed contact graph) used in
previous iterations. Instead, it treats the problem as a global optimization
landscape where:
1. The objective is the side length of the minimal enclosing regular hexagon.
2. Collisions are handled via a hard penalty (SAT-based overlap detection).
3. Simulated Annealing is used to escape local minima, allowing hexagons to
   rotate and translate freely.
4. The outer hexagon center and rotation are also optimized to tighten the
   fit.

This is materially different from the seed program because it does not assume
a specific arrangement (e.g., "center + ring"). It allows the packer to discover
optimal structures, including non-symmetric or shifted configurations, that
minimize the bounding box.
"""

import time
import numpy as np
import random

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_and_center(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Compute the minimum side length and optimal center for the outer hexagon
    containing the given inner hexagons.
    
    Returns: (side_length, optimal_center)
    """
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    
    # For a regular hexagon with side S centered at (cx, cy) and rotation theta,
    # a point (x, y) is inside if its distance to each of the 6 sides is <= apothem = S*sqrt(3)/2.
    # Alternatively, using the support function formulation in the rotated frame:
    # Let u, v be coordinates in the rotated frame.
    # The hexagon is defined by |u| <= S, |u + v/sqrt3| <= S, |u - v/sqrt3| <= S ? 
    # No, that's for a hexagon with flat top/bottom aligned with axes?
    # Let's use the standard definition:
    # Vertices at angle theta + k*pi/3.
    # The constraint for a point p to be inside hexagon H(center c, side S, angle theta) is:
    # max_{k=0..5} dot(p-c, n_k) <= S * sqrt(3) / 2, where n_k are normals to sides.
    # Wait, the support function of a regular hexagon with side S in direction n is:
    # h(n) = S * sqrt(3) / 2 * max(1, 2/3 * |dot(n, e_0)| + 1/3 * |dot(n, e_1)| ... )? 
    # Simpler: The width of the hexagon in any direction is between S*sqrt(3) and 2*S.
    # Actually, let's just iterate over the 6 possible orientations of the bounding box?
    # No, the outer hexagon orientation is a variable.
    
    # Let's reformulate.
    # A point p is inside a regular hexagon of side S centered at c with orientation phi if:
    # dist(p, c) in the "hexagonal norm" is <= S * sqrt(3) / 2 ?
    # The hexagonal norm is defined by the dual hexagon.
    # For a hexagon with vertices at angles phi + k*pi/3, the inradius is R_in = S * sqrt(3) / 2.
    # The condition is max_{k=0..5} dot(p-c, n_k) <= R_in, where n_k are unit normals to the sides.
    # The sides are perpendicular to directions phi + pi/6 + k*pi/3.
    
    # So, for a given outer angle, we can compute the required S for each vertex and take the max.
    # S_required = max( dot(p-c, n_k) / (sqrt(3)/2) ) for all p, k.
    
    # Since we want to minimize S, and we can also shift the center, this is a Chebyshev-like problem.
    # However, for speed in SA, let's fix the outer center to the centroid of the inner hexagons' centroids
    # and only optimize the outer angle? Or optimize center too?
    # Optimizing center adds 2 dimensions. Let's try to optimize center and angle.
    
    # To make it fast, we can approximate the optimal center as the centroid of all vertices.
    # Then we only need to find the min S for a given angle.
    # Actually, the min S for a given angle is determined by the max projection.
    # Let's compute the min S for a given angle assuming center is fixed at (0,0) for now, 
    # but we can shift the inner points to center them.
    
    # Strategy: 
    # 1. Compute centroid of all inner vertices.
    # 2. Shift inner vertices so centroid is at (0,0).
    # 3. For a given outer angle, compute max projection onto 6 normal directions.
    # 4. S = max_proj / (sqrt(3)/2).
    # 5. The optimal center for the outer hexagon is the centroid.
    
    centroid_x = np.mean(all_verts_x)
    centroid_y = np.mean(all_verts_y)
    
    dx = all_verts_x - centroid_x
    dy = all_verts_y - centroid_y
    
    # Transform to outer hexagon frame (rotated by -outer_angle)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # Normals to the sides of a hexagon aligned with angle 0 (flat top? pointy top?)
    # Our hex_vertices uses pointy top when angle=0? 
    # i=0: theta=0 -> (S, 0). This is a pointy right vertex.
    # So the sides are at angles 30, 90, 150, 210, 270, 330.
    # Normals are at 30, 90, 150, 210, 270, 330? No, normal to side connecting (S,0) and (S/2, S*sqrt3/2) is...
    # Let's use the property: max_{k} dot(p, n_k) <= R_in.
    # The normals n_k are at angles phi + pi/6 + k*pi/3.
    
    normal_angles = a_outer + np.pi/6 + np.arange(6) * np.pi/3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    # Project all points onto all normals
    # proj[i, k] = lx[i]*nx[k] + ly[i]*ny[k]
    # We need max over i and k.
    # Since we want S, and R_in = S * sqrt(3) / 2,
    # S = max_proj / (sqrt(3) / 2)
    
    # Vectorized:
    # lx is (N*6,), ly is (N*6,)
    # nx is (6,), ny is (6,)
    # projections = lx[:, None] * nx[None, :] + ly[:, None] * ny[None, :]
    # max_proj = np.max(projections)
    
    # To save memory/time, we can just compute the max for each normal and take the global max.
    max_proj = 0.0
    for k in range(6):
        proj = lx * nx[k] + ly * ny[k]
        m = np.max(proj)
        if m > max_proj:
            max_proj = m
            
    side = max_proj / (SQRT3 / 2.0)
    
    return side, (centroid_x, centroid_y)

def _check_overlap_fast(pos, ang):
    """
    Check for overlaps using a fast bounding box check first, then SAT if necessary.
    Returns penalty (0 if no overlap, >0 if overlap).
    """
    penalty = 0.0
    # Pre-compute vertices? No, SAT needs vertices.
    # Let's just do SAT for all pairs. N=11, 55 pairs.
    # To speed up, check center distance first.
    # Min distance between centers of two non-overlapping unit hexagons is 2*sqrt(3) if aligned,
    # but can be less if rotated? No, the inradius is sqrt(3)/2.
    # Distance between centers must be at least 2 * inradius = sqrt(3) if they touch at vertices?
    # No, if they touch at vertices, distance is 2. If they touch at sides, distance is 2*sqrt(3).
    # So if center distance < 2, they definitely overlap.
    # If center distance < 2*sqrt(3), they *might* overlap.
    
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i]))
        
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 4.0: # 2^2
                # Definitely overlap
                penalty += 100.0
                continue
            if dist_sq < 12.0: # (2*sqrt(3))^2
                # Check SAT
                v1 = verts_list[i]
                v2 = verts_list[j]
                axes = []
                for v in [v1, v2]:
                    for k in range(6):
                        p1 = v[k]
                        p2 = v[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            axes.append(normal / norm)
                
                overlap = True
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
                        overlap = False
                        break
                if overlap:
                    # Estimate penalty based on depth? Just use constant for now.
                    penalty += 100.0
    return penalty

def _objective(pos, ang, outer_angle):
    """
    Objective function: Outer side length + Overlap Penalty.
    """
    side, _ = _min_outer_side_and_center(pos, ang, (0.0, 0.0), outer_angle)
    overlap_penalty = _check_overlap_fast(pos, ang)
    return side + overlap_penalty

def construct():
    global START_TIME, DEADLINE
    
    # Initial configuration: Random
    # Or use the seed's initial configuration as a good start
    sqrt3 = np.sqrt(3)
    pos_init = np.zeros((N, 2))
    ang_init = np.zeros(N)
    
    # 0: Center
    pos_init[0] = [0, 0]
    for i in range(6):
        a = i * np.pi / 3
        pos_init[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos_init[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    best_pos = pos_init.copy()
    best_ang = ang_init.copy()
    best_outer_angle = 0.0
    best_side = _min_outer_side_and_center(best_pos, best_ang, (0.0, 0.0), 0.0)[0]
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    steps = 50000
    
    current_pos = pos_init.copy()
    current_ang = ang_init.copy()
    current_outer_angle = 0.0
    current_score = _objective(current_pos, current_ang, current_outer_angle)
    
    i = 0
    while i < steps and time.time() < DEADLINE - 5.0:
        i += 1
        
        # Generate neighbor
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb a random hexagon
        k = random.randint(0, N-1)
        # Move
        new_pos[k, 0] += random.gauss(0, 0.2)
        new_pos[k, 1] += random.gauss(0, 0.2)
        # Rotate
        new_ang[k] += random.gauss(0, 0.2)
        
        # Occasionally perturb outer angle
        if random.random() < 0.1:
            new_outer_angle += random.gauss(0, 0.1)
            
        # Occasionally perturb global center (shift all)
        if random.random() < 0.05:
            shift_x = random.gauss(0, 0.1)
            shift_y = random.gauss(0, 0.1)
            new_pos[:, 0] += shift_x
            new_pos[:, 1] += shift_y
            
        new_score = _objective(new_pos, new_ang, new_outer_angle)
        
        # Accept if better or with probability exp(-(delta)/T)
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_pos = new_pos
            current_ang = new_ang
            current_outer_angle = new_outer_angle
            current_score = new_score
            
            # Update best
            if current_score < best_side:
                best_pos = current_pos.copy()
                best_ang = current_ang.copy()
                best_outer_angle = current_outer_angle
                best_side = current_score
                
        T *= cooling_rate
        
    # Final refinement: try to optimize outer angle for the best configuration
    # We can do a small local search on outer angle
    if time.time() < DEADLINE - 2.0:
        for _ in range(100):
            # Try small variations of outer angle
            for da in [-0.1, 0.1, -0.05, 0.05]:
                test_angle = best_outer_angle + da
                side, _ = _min_outer_side_and_center(best_pos, best_ang, (0.0, 0.0), test_angle)
                if side < best_side:
                    best_side = side
                    best_outer_angle = test_angle
                    
    # Final validity check
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    
    # We need to return the outer center as well.
    # The _min_outer_side_and_center returns the optimal center.
    final_side, final_center = _min_outer_side_and_center(best_pos, best_ang, (0.0, 0.0), best_outer_angle)
    
    # Verify no overlaps in final config
    if _check_overlap_fast(best_pos, best_ang) > 0:
        # Fallback to initial
        pos_f, ang_f = pos_init, ang_init
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        final_side, final_center = _min_outer_side_and_center(pos_f, ang_f, (0.0, 0.0), 0.0)
        best_outer_angle = 0.0
        
    return inner_list, (float(final_center[0]), float(final_center[1])), float(final_side), float(best_outer_angle)

if __name__ == "__main__":
    construct()
