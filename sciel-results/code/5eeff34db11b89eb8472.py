"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Use a Multistart Nelder-Mead approach.
  2. Define a smooth objective function that penalizes overlaps based on penetration depth.
  3. This allows the optimizer to navigate through overlapping states smoothly,
     which is a structural change from the hard-constraint SA approach.
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 135.0


def _elapsed():
    return time.time() - START_TIME


def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, rotation."""
    angle_rad = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts[k, 0] = cx + side * np.cos(a)
        verts[k, 1] = cy + side * np.sin(a)
    return verts


def compute_minimal_enclosing_side(inner_hexes, outer_rotation_deg=0.0):
    """Compute minimal side length of outer hexagon (with given rotation)
    that contains all inner hexagons. Returns (side, center_x, center_y)."""
    all_verts = np.vstack([
        get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner_hexes
    ])

    rot_rad = np.radians(outer_rotation_deg)
    normals = np.empty((6, 2))
    for k in range(6):
        a = rot_rad + k * np.pi / 3 + np.pi / 6
        normals[k, 0] = np.cos(a)
        normals[k, 1] = np.sin(a)

    proj_max = np.max(np.dot(all_verts, normals.T), axis=0)
    proj_min = np.min(np.dot(all_verts, normals.T), axis=0)

    def objective(c):
        cx, cy = c
        center_dot = cx * normals[:, 0] + cy * normals[:, 1]
        hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
        return hw.max()

    c0 = np.array([all_verts[:, 0].mean(), all_verts[:, 1].mean()])
    result = minimize(objective, c0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})

    cx_opt, cy_opt = result.x
    center_dot = cx_opt * normals[:, 0] + cy_opt * normals[:, 1]
    hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
    L = hw.max() / (np.sqrt(3) / 2)

    return L, cx_opt, cy_opt


def get_penetration_depth(v1, v2):
    """Calculate the minimum translation distance to separate two convex polygons.
    Returns 0 if separated, positive value if overlapping (penetration depth)."""
    min_sep = np.inf
    for verts in (v1, v2):
        for i in range(6):
            a = verts[i]
            b = verts[(i + 1) % 6]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            # Separation distance: if max of one is less than min of other, they are separated
            # We want the minimum "overlap" which is max(0, min(proj1) - max(proj2)) etc?
            # Actually, for separation, we look for an axis where intervals do not intersect.
            # If they intersect on all axes, they overlap.
            # The "depth" of overlap along an axis is how much we need to push them apart.
            # If proj1_max < proj2_min, separated by proj2_min - proj1_max.
            # If proj2_max < proj1_min, separated by proj1_min - proj2_max.
            # If they overlap, the "separation" is negative.
            # We want the minimum separation distance (most negative) across all axes.
            # If min_sep > 0, they are separated.
            # If min_sep < 0, they overlap by -min_sep.
            
            sep1 = proj2.min() - proj1.max()
            sep2 = proj1.min() - proj2.max()
            
            # If either sep is positive, they are separated on this axis.
            # We are looking for the axis that allows the *least* separation (i.e., the tightest fit).
            # If they are separated on this axis, the distance is positive.
            # If they overlap, the distance is negative.
            # The overall separation distance is the max of all axis separations?
            # No, if they are separated on ANY axis, they are disjoint.
            # The distance between them is the max of (sep1, sep2) over all axes?
            # Actually, the minimum translation vector magnitude is the max of the "negative overlaps" if we consider the worst case?
            # Let's use the standard definition: 
            # If they are disjoint, distance = min over axes of (separation on that axis).
            # If they are overlapping, distance = 0 (or we define penetration depth).
            # For a smooth penalty, we want a value that is 0 when disjoint and >0 when overlapping.
            # Let d_axis = min(proj1) - max(proj2) for one orientation? No.
            # Let's define overlap_axis = max(0, max(proj1) - min(proj2), max(proj2) - min(proj1)) is not right.
            
            # Correct SAT logic for distance:
            # For each axis, compute the gap.
            # gap = min(proj1, proj2) - max(proj1, proj2) ... no.
            # gap = min( min(proj1) - max(proj2), min(proj2) - max(proj1) )
            # If gap > 0, they are separated by gap.
            # If gap < 0, they overlap by -gap on this axis.
            # If they are disjoint, the distance is the MAX of the gaps over all axes?
            # No, if they are disjoint, there exists at least one axis with gap > 0.
            # The actual distance is the minimum gap over all axes? No, that would be negative if they overlap on some axes.
            # Wait, if they are disjoint, ALL axes have gap > 0? No, only the separating axis has gap > 0.
            # Actually, for convex polygons, if they are disjoint, there is at least one axis where they are separated.
            # The distance between the polygons is the minimum over all axes of the "separation" if we consider the projection intervals?
            # No, the distance is the length of the minimum translation vector.
            # A simpler smooth approximation for penalty:
            # Penalty = max(0, -min_gap) where min_gap is the minimum gap over all axes.
            # If min_gap > 0, they are separated.
            # If min_gap < 0, they overlap. The magnitude of the most negative gap is a proxy for penetration depth.
            
            gap = min(proj1.min() - proj2.max(), proj2.min() - proj1.max())
            min_sep = min(min_sep, gap)
            
    if min_sep > 0:
        return 0.0
    else:
        return -min_sep


def smooth_overlap_penalty(inner_hexes):
    """Calculate a smooth penalty for overlaps."""
    verts_list = [get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner_hexes]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = get_penetration_depth(verts_list[i], verts_list[j])
            if depth > 0:
                # Use a quadratic penalty to make it smooth and strong
                penalty += depth * depth
    return penalty


def construct():
    global START_TIME
    START_TIME = time.time()

    np.random.seed(42)

    # Initial configuration: hexagonal grid arrangement
    dx = 3.0
    dy = np.sqrt(3)

    inner = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            inner.append([x, y, 0.0])

    best_side = np.inf
    best_inner = [list(h) for h in inner]
    best_center = [0.0, 0.0]
    best_rot = 0.0

    # Multistart Nelder-Mead
    # We will run NM from the initial grid and a few random perturbations.
    
    def objective_wrapper(params, rot):
        if _elapsed() > TIME_LIMIT - 5:
            return 1e6
        hexes = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
        
        # Compute penalty for overlaps
        pen = smooth_overlap_penalty(hexes)
        
        if pen > 1e-6:
            # If overlapping, return a large value plus the penalty
            # We still compute the side length to provide gradient info if possible, 
            # but for NM, a large constant is often enough to push away.
            # However, adding the penalty helps navigate.
            s, _, _ = compute_minimal_enclosing_side(hexes, rot)
            return s + 10.0 * pen
        else:
            s, _, _ = compute_minimal_enclosing_side(hexes, rot)
            return s

    # Try a few rotations
    rotations = [0.0, 30.0, 15.0]
    
    for rot in rotations:
        if _elapsed() > TIME_LIMIT - 20:
            break
            
        # Start from the grid
        params0 = np.array([p for h in inner for p in h])
        
        # Run Nelder-Mead
        try:
            result = minimize(objective_wrapper, params0, args=(rot,), method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < best_side and result.fun < 1e5: # Check if valid (not huge penalty)
                # Verify no overlap with tight tolerance
                final_hexes = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
                pen = smooth_overlap_penalty(final_hexes)
                if pen < 1e-6:
                    s, cx, cy = compute_minimal_enclosing_side(final_hexes, rot)
                    if s < best_side:
                        best_side = s
                        best_inner = final_hexes
                        best_center = [cx, cy]
                        best_rot = rot
        except Exception as e:
            pass

        # Try a random start
        if _elapsed() > TIME_LIMIT - 20:
            break
        random_params = np.random.normal(0, 5, size=(N * 3,))
        # Scale angles to be reasonable
        random_params[2::3] = np.random.uniform(0, 60, size=N)
        
        try:
            result = minimize(objective_wrapper, random_params, args=(rot,), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < best_side and result.fun < 1e5:
                final_hexes = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
                pen = smooth_overlap_penalty(final_hexes)
                if pen < 1e-6:
                    s, cx, cy = compute_minimal_enclosing_side(final_hexes, rot)
                    if s < best_side:
                        best_side = s
                        best_inner = final_hexes
                        best_center = [cx, cy]
                        best_rot = rot
        except Exception as e:
            pass

    # Final precise computation
    final_side, final_cx, final_cy = compute_minimal_enclosing_side(best_inner, best_rot)

    return best_inner, (final_cx, final_cy), final_side, best_rot
