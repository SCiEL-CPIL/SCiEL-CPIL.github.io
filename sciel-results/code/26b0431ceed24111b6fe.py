"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Generate the 19 centers of a side-3 regular hexagon grid (flat-top).
  2. For each cluster rotation angle, rotate all centers.
  3. For each rotated center, precompute the optimal per-hexagon rotation
     that minimizes its maximum projection onto the outer hexagon edges.
  4. Enumerate all subsets of size 12 from these 19 centers.
  5. For each subset, the required outer side L is the max of the individual
     required L values.
  6. Use a tiny scale factor to ensure strict disjointness.
"""
import time
import math
import numpy as np
from itertools import combinations

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
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))

    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta
        to fit in outer hex of side L (flat-top, centered at origin)."""
        # Vertex angles relative to hexagon center
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])

        # Projection of each vertex onto each normal direction
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 2-degree steps (60-degree period)
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine around best with finer steps
        for _ in range(2):
            step = 0.5 if _ == 0 else 0.1
            t_start = best_theta - step * 6
            t_end = best_theta + step * 6
            for t_deg in np.arange(t_start, t_end + step * 0.5, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict non-overlap (adjacent hexagons in grid touch at distance sqrt(3))
    SCALE = 1.00001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    # The cluster has 6-fold symmetry, so search 0 to 60 degrees
    # Coarse search first, then refine
    cluster_angles_coarse = [math.radians(a) for a in range(0, 61, 5)]
    # Add 30 degrees explicitly (often near-optimal)
    if math.radians(30) not in cluster_angles_coarse:
        cluster_angles_coarse.append(math.radians(30))

    # Precompute for each cluster angle: the optimal theta and L for each of the 19 centers
    # We'll do this lazily as we iterate through cluster angles

    for cluster_angle in cluster_angles_coarse:
        if time.time() > deadline:
            break

        cos_ca = math.cos(cluster_angle)
        sin_ca = math.sin(cluster_angle)

        # Rotate all centers by cluster_angle
        rotated_centers = []
        for x, y in all_centers:
            rx = x * cos_ca - y * sin_ca
            ry = x * sin_ca + y * cos_ca
            rotated_centers.append((rx, ry))

        # Precompute optimal theta for each center
        center_optimal = []
        for i in range(19):
            x, y = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))

        # Try all subsets of size 12 from 19 centers
        for subset in combinations(range(19), 12):
            if time.time() > deadline:
                break
            # L is the max over all hexagons in subset
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle

    # Refine around the best cluster angle found
    if best_cluster_angle > 0 and time.time() < deadline:
        # Fine search around best cluster angle
        step_deg = 1.0
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            ca = math.radians(math.degrees(best_cluster_angle) + offset * step_deg)
            if time.time() > deadline:
                break
            cos_ca = math.cos(ca)
            sin_ca = math.sin(ca)

            rotated_centers = []
            for x, y in all_centers:
                rx = x * cos_ca - y * sin_ca
                ry = x * sin_ca + y * cos_ca
                rotated_centers.append((rx, ry))

            center_optimal = []
            for i in range(19):
                x, y = rotated_centers[i]
                L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
                center_optimal.append((L_h, theta_h))

            for subset in combinations(range(19), 12):
                if time.time() > deadline:
                    break
                L = max(center_optimal[i][0] for i in subset)
                if L < best_L:
                    best_L = L
                    best_subset = subset
                    best_thetas = [center_optimal[i][1] for i in subset]
                    best_cluster_angle = ca

    if best_subset is None:
        # Fallback: use first 12 with no cluster rotation
        best_cluster_angle = 0.0
        rotated_centers = all_centers
        center_optimal = []
        for i in range(19):
            x, y = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)

    # Construct inner hexagons
    cos_ca = math.cos(best_cluster_angle)
    sin_ca = math.sin(best_cluster_angle)
    inner = []
    for idx, i in enumerate(best_subset):
        x, y = all_centers[i]
        rx = x * cos_ca - y * sin_ca
        ry = x * sin_ca + y * cos_ca
        inner.append([rx * SCALE, ry * SCALE, math.degrees(best_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
