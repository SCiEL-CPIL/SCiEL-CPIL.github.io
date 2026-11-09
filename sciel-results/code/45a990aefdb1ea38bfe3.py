"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        # Vertex angles relative to hexagon center
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        # Projection of each vertex onto each normal direction
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees (period is 60 degrees due to hexagonal symmetry)
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine around best
        for _ in range(2):
            step = 0.1 if _ == 0 else 0.01
            t_start = best_theta - step * 10
            t_end = best_theta + step * 10 + step
            for t_deg in np.arange(t_start, t_end, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Small scale to ensure strict non-overlap
    SCALE = 1.000001
    
    # Precompute optimal theta for each center (independent of subset)
    center_optimal = []
    for i in range(19):
        x, y, _, _ = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
        center_optimal.append((L_h, theta_h))
    
    # Try all subsets of size 12 from 19 centers
    # C(19,12) = C(19,7) = 50388
    from itertools import combinations
    
    best_L = float('inf')
    best_subset = None
    best_thetas = None
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline:
            break
        # L is the max over all hexagons in subset
        L = max(center_optimal[i][0] for i in subset)
        if L < best_L:
            best_L = L
            best_subset = subset
            best_thetas = [center_optimal[i][1] for i in subset]
    
    if best_subset is None:
        # Fallback: use first 12
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)
    
    # Construct inner hexagons
    inner = []
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        inner.append([x * SCALE, y * SCALE, math.degrees(best_thetas[idx])])
    
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
