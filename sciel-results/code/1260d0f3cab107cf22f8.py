"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Flat-Top hexagon grid: x = 1.5*q, y = sqrt(3)*(r + q/2)
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))

    # Normal directions for flat-top outer hexagon (normals at 30,90,150,210,270,330 deg)
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)

    # Precompute: for a point (px,py), the required L is max_j(px*cos(nj)+py*sin(nj)) / (sqrt3/2)
    # We need this for all 6 vertices of each unit hexagon.

    def min_L_for_vertex(px, py):
        """Minimum L for a single point to be inside outer hex of side L."""
        projs = px * normal_cos + py * normal_sin
        return np.max(projs) / (sqrt3 / 2.0)

    def min_L_for_hexagon(cx, cy, theta):
        """Minimum L for a unit hexagon at (cx,cy) with rotation theta."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        # For each vertex, compute required L; take max
        # Vectorize: projs[i,j] = vx[i]*cos(nj) + vy[i]*sin(nj)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def total_L_for_config(centers, thetas):
        """Compute the outer hex side L needed for all hexagons."""
        max_L = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            L_i = min_L_for_hexagon(cx, cy, thetas[i])
            if L_i > max_L:
                max_L = L_i
        return max_L

    def check_non_overlap(centers, thetas, margin=0.001):
        """Check that all pairs of hexagons are disjoint (strictly)."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                # Two unit hexagons are disjoint if distance between centers > 2*inradius
                # For same orientation, min center distance for non-overlap is sqrt(3) (edge-to-edge)
                # For different orientations, we need to check more carefully.
                # Conservative: require center distance > 2 * (sqrt(3)/2) + margin = sqrt(3) + margin
                # But with rotation, the actual minimum distance can be smaller.
                # Use a safer check: compute max over all vertex-edge pairs.
                # For speed, use the conservative bound: if dist > 2.0 + margin, definitely safe.
                # If dist < sqrt(3) - margin, definitely overlapping for same orientation.
                # For safety with rotations, require dist > sqrt(3) - 0.001 for same-orientation,
                # and for different orientations, do a proper check.
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Conservative: for any two unit hexagons, if dist > 2.0 they can't overlap
                # (max width of unit hexagon is 2*sqrt(3)/2*... actually diameter is 2)
                # The circumradius is 1, so two hexagons with centers dist > 2 cannot overlap.
                # But they can overlap if dist < sqrt(3) even for different orientations.
                # Let's use: if dist > 2.0, safe. Otherwise do vertex-edge check.
                if dist > 2.0 + margin:
                    continue
                # Do a proper overlap check: for each vertex of hex i, check if it's inside hex j
                # and for each vertex of hex j, check if it's inside hex i
                # Also check edge-edge intersections (skip for now, vertex-in-polygon is usually sufficient)
                vi_angles = np.array([math.radians(60 * k) + thetas[i] for k in range(6)])
                vj_angles = np.array([math.radians(60 * k) + thetas[j] for k in range(6)])
                vi = np.column_stack([centers[i][0] + np.cos(vi_angles), centers[i][1] + np.sin(vi_angles)])
                vj = np.column_stack([centers[j][0] + np.cos(vj_angles), centers[j][1] + np.sin(vj_angles)])
                
                # Check if any vertex of i is inside hex j
                for k in range(6):
                    px, py = vi[k]
                    if point_in_hexagon(px, py, centers[j], thetas[j]):
                        return False
                # Check if any vertex of j is inside hex i
                for k in range(6):
                    px, py = vj[k]
                    if point_in_hexagon(px, py, centers[i], thetas[i]):
                        return False
        return True

    def point_in_hexagon(px, py, center, theta):
        """Check if point (px,py) is inside unit hexagon at center with rotation theta."""
        cx, cy = center
        # Transform to hexagon-local coordinates
        dx = px - cx
        dy = py - cy
        # Rotate by -theta
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        lx = dx * cos_t - dy * sin_t
        ly = dx * sin_t + dy * cos_t
        # Unit flat-top hexagon: vertices at angles 0,60,...,300
        # The hexagon is the intersection of 6 half-planes
        # Normal directions at 30,90,150,210,270,330
        for k in range(6):
            ang = math.radians(60 * k + 30)
            proj = lx * math.cos(ang) + ly * math.sin(ang)
            if proj > sqrt3 / 2.0 + 1e-10:
                return False
        return True

    # Phase 1: Enumerate all C(19,12) = 50388 subsets, find best with optimal rotations
    best_L = float('inf')
    best_subset = None
    best_thetas = None

    # For each subset, compute the L with optimal per-hexagon rotation
    # Optimal rotation for a hexagon at (cx,cy): we want to minimize
    # max_j [cx*cos(nj) + cy*sin(nj) + max_k cos(60k + theta - nj)] / (sqrt3/2)
    # The term max_k cos(60k + theta - nj) = cos(min angular distance from (theta-nj) to nearest 60deg multiple)
    # This is a piecewise function with period 60 degrees.
    # We can find the optimal theta by sampling.

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees (period 60 due to hex symmetry)
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine
        for _ in range(2):
            step = 0.1 if _ == 0 else 0.01
            t_lo = best_theta - step * 10
            t_hi = best_theta + step * 10
            for t_deg in np.arange(t_lo, t_hi + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    for subset in combinations(range(19), 12):
        if time.time() > deadline - 30.0:  # Leave 30s for refinement
            break
        # Compute L with optimal rotations
        L_subset = 0.0
        thetas_subset = []
        for idx in subset:
            cx, cy = all_centers[idx]
            L_h, theta_h = optimal_theta_for_hexagon(cx, cy)
            thetas_subset.append(theta_h)
            if L_h > L_subset:
                L_subset = L_h
        if L_subset < best_L:
            best_L = L_subset
            best_subset = subset
            best_thetas = thetas_subset

    if best_subset is None:
        best_subset = tuple(range(12))
        best_thetas = [0.0] * 12
        best_L = total_L_for_config([all_centers[i] for i in best_subset], best_thetas)

    # Phase 2: Local continuous refinement using Nelder-Mead
    # Variables: 12*(x,y,theta) = 36 variables
    # Objective: outer hex side L
    # Constraint: non-overlap (enforced via penalty)

    def objective(params):
        centers = []
        thetas = []
        for i in range(12):
            x = params[3 * i]
            y = params[3 * i + 1]
            theta = params[3 * i + 2]
            centers.append((x, y))
            thetas.append(theta)
        L = total_L_for_config(centers, thetas)
        # Penalty for overlap
        if not check_non_overlap(centers, thetas, margin=0.005):
            L += 10.0
        return L

    # Initial params from grid + optimal rotations
    x0 = []
    for i, idx in enumerate(best_subset):
        cx, cy = all_centers[idx]
        x0.extend([cx, cy, best_thetas[i]])
    x0 = np.array(x0)

    # Try Nelder-Mead refinement
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        if result.fun < best_L:
            # Verify non-overlap
            centers = []
            thetas = []
            for i in range(12):
                centers.append((result.x[3 * i], result.x[3 * i + 1]))
                thetas.append(result.x[3 * i + 2])
            if check_non_overlap(centers, thetas, margin=0.005):
                best_L = result.fun
                best_subset = best_subset  # keep for reference
                best_thetas = thetas
                best_centers = centers
    except Exception:
        pass

    # If refinement succeeded, use refined positions; otherwise use grid positions
    try:
        if 'best_centers' in dir():
            inner = []
            for i in range(12):
                inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_thetas[i])])
        else:
            inner = []
            for i, idx in enumerate(best_subset):
                cx, cy = all_centers[idx]
                inner.append([cx, cy, math.degrees(best_thetas[i])])
    except Exception:
        inner = []
        for i, idx in enumerate(best_subset):
            cx, cy = all_centers[idx]
            inner.append([cx, cy, 0.0])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
