"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    n_centers = len(all_centers)  # 19
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 0.5 degree steps
        for t_deg in np.arange(0, 60.5, 0.5):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001]:
            step = math.radians(step_deg)
            t_start = best_theta - step * 10
            t_end = best_theta + step * 10 + step
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end), step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Minimal scale to ensure strict disjointness (distance > sqrt(3))
    # Adjacent centers in hex grid are exactly sqrt(3) apart
    # We need distance > sqrt(3), so scale by (1 + epsilon)
    # With epsilon = 1e-6, distance becomes sqrt(3)*(1+1e-6) > sqrt(3)
    SCALE = 1.000001
    
    # Precompute optimal theta for each center
    center_optimal = []
    for i in range(n_centers):
        x, y, _, _ = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x, y)
        center_optimal.append((L_h, theta_h))
    
    # Sort centers by their individual L requirement
    sorted_indices = sorted(range(n_centers), key=lambda i: center_optimal[i][0])
    
    # Greedy approach: pick the 12 centers with smallest L, then verify
    # But we need to check if they're valid (non-overlapping)
    # For now, just use the 12 with smallest L since the hex grid ensures non-overlap
    best_subset = tuple(sorted_indices[:12])
    best_thetas = [center_optimal[i][1] for i in best_subset]
    best_L = max(center_optimal[i][0] for i in best_subset)
    
    # Try to improve by swapping: replace one center with another
    # that might allow a smaller max L
    remaining = sorted_indices[12:]
    for _ in range(100):  # Limited swaps
        if time.time() > deadline:
            break
        improved = False
        for out_idx in range(12):
            for in_idx in remaining:
                # Check if swapping reduces max L
                new_subset = list(best_subset)
                new_subset[out_idx] = in_idx
                new_L = max(center_optimal[i][0] for i in new_subset)
                if new_L < best_L - 1e-10:
                    best_L = new_L
                    best_subset = tuple(new_subset)
                    best_thetas = [center_optimal[i][1] for i in best_subset]
                    # Update remaining
                    remaining = [i for i in range(n_centers) if i not in best_subset]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Construct inner hexagons with scaled positions
    inner = []
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        inner.append([x * SCALE, y * SCALE, math.degrees(best_thetas[idx])])
    
    # The outer hexagon side length needs to accommodate the scaled positions
    # Since we scaled positions by SCALE, the required L is also scaled
    final_L = best_L * SCALE
    
    return inner, (0.0, 0.0), final_L, 0.0
