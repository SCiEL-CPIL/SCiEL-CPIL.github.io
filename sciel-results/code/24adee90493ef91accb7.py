"""Packing 12 unit hexagons in minimal regular hexagon - optimized."""
import math
import time
import numpy as np
from itertools import combinations

N = 12
SQRT3 = math.sqrt(3.0)
APH = SQRT3 / 2.0  # apothem factor: apothem = L * APH

# Face normals of flat-top outer hexagon (outward), at 30,90,...,330 degrees
NORM_ANG = np.array([math.radians(30 + 60 * j) for j in range(6)])
NCOS = np.cos(NORM_ANG)
NSIN = np.sin(NORM_ANG)
# Precompute vertex direction cos/sin for 60-degree steps
KANG = np.array([math.radians(60 * k) for k in range(6)])
KCOS = np.cos(KANG)
KSIN = np.sin(KANG)


def min_L_at(cx, cy, theta):
    """Minimum outer side L for a unit hexagon at (cx,cy) with rotation theta."""
    # Vertex angles: theta + 60*k
    # cos(theta + 60k) = cos(theta)cos(60k) - sin(theta)sin(60k)
    ct, st = math.cos(theta), math.sin(theta)
    vx = cx + ct * KCOS - st * KSIN
    vy = cy + ct * KSIN + st * KCOS
    # projections onto 6 normals
    projs = vx[:, None] * NCOS[None, :] + vy[:, None] * NSIN[None, :]
    return float(np.max(projs)) / APH


def min_L_center(cx, cy):
    """Minimize L over rotation theta (period 60 deg) for center (cx,cy)."""
    # Golden section search on [0, pi/3]
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = 0.0, math.pi / 3.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = min_L_at(cx, cy, c)
    fd = min_L_at(cx, cy, d)
    for _ in range(50):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = min_L_at(cx, cy, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = min_L_at(cx, cy, d)
    theta = 0.5 * (a + b)
    return min_L_at(cx, cy, theta), theta


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # Generate all 19 centers of the side-3 hexagonal cluster (axial coords)
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = SQRT3 * (r + 0.5 * q)
                centers.append((x, y))
    centers = np.array(centers, dtype=float)  # (19, 2)

    best_L = float('inf')
    best_pos = None
    best_theta = None
    best_cluster = 0.0

    # Cluster rotation search: 0..60 deg (6-fold symmetry)
    # Coarse then fine
    cluster_angles = []
    for a_deg in range(0, 61, 3):
        cluster_angles.append(math.radians(a_deg))
    # Add key symmetric angles
    for a_deg in [15, 30, 45]:
        cluster_angles.append(math.radians(a_deg))
    cluster_angles = sorted(set(cluster_angles))

    # Precompute per-center optimal L and theta for each cluster angle
    for cluster_angle in cluster_angles:
        if time.time() > deadline:
            break
        ca, sa = math.cos(cluster_angle), math.sin(cluster_angle)
        # Rotate centers
        rot = np.empty_like(centers)
        rot[:, 0] = centers[:, 0] * ca - centers[:, 1] * sa
        rot[:, 1] = centers[:, 0] * sa + centers[:, 1] * ca

        # Optimal L and theta per center
        opt_L = np.empty(19)
        opt_th = np.empty(19)
        for i in range(19):
            opt_L[i], opt_th[i] = min_L_center(rot[i, 0], rot[i, 1])

        # Select 12 centers minimizing max L.
        # Sort by L descending; the max over any 12-subset is minimized by
        # dropping the 7 largest-L centers.
        order = np.argsort(opt_L)  # ascending
        # The 12 chosen = the 12 with smallest L? No: max over subset.
        # To minimize max, drop the 7 with largest L.
        keep = order[:12]
        L_candidate = float(np.max(opt_L[keep]))
        if L_candidate < best_L - 1e-12:
            best_L = L_candidate
            best_pos = rot[keep].copy()
            best_theta = opt_th[keep].copy()
            best_cluster = cluster_angle
            if best_L < 3.001:
                break

    # If coarse search didn't find good, do fine refinement around best cluster
    if best_L > 3.001 and time.time() < deadline:
        # Fine search around best_cluster
        fine_angles = np.linspace(best_cluster - 0.15, best_cluster + 0.15, 15)
        for cluster_angle in fine_angles:
            if time.time() > deadline:
                break
            ca, sa = math.cos(cluster_angle), math.sin(cluster_angle)
            rot = np.empty_like(centers)
            rot[:, 0] = centers[:, 0] * ca - centers[:, 1] * sa
            rot[:, 1] = centers[:, 0] * sa + centers[:, 1] * ca
            opt_L = np.empty(19)
            opt_th = np.empty(19)
            for i in range(19):
                opt_L[i], opt_th[i] = min_L_center(rot[i, 0], rot[i, 1])
            order = np.argsort(opt_L)
            keep = order[:12]
            L_candidate = float(np.max(opt_L[keep]))
            if L_candidate < best_L - 1e-12:
                best_L = L_candidate
                best_pos = rot[keep].copy()
                best_theta = opt_th[keep].copy()
                best_cluster = cluster_angle

    # Local refinement: nudge positions/angles to reduce max L further
    if time.time() < deadline - 3.0 and best_pos is not None:
        # Coordinate descent: for each hexagon, re-optimize its angle and
        # small position nudge to reduce its individual L (which may reduce max)
        improved = True
        for _ in range(6):
            if not improved:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3.0:
                    break
                # Re-optimize angle for this center
                new_L, new_th = min_L_center(best_pos[i, 0], best_pos[i, 1])
                # Small radial nudge toward center to reduce L
                r = math.hypot(best_pos[i, 0], best_pos[i, 1])
                if r > 1e-9:
                    for frac in [0.995, 0.99, 0.985, 0.98]:
                        nx = best_pos[i, 0] * frac
                        ny = best_pos[i, 1] * frac
                        L_try, th_try = min_L_center(nx, ny)
                        if L_try < new_L - 1e-12:
                            new_L, new_th = L_try, th_try
                            best_pos[i, 0] = nx
                            best_pos[i, 1] = ny
                            improved = True
                            break
                if new_L < min_L_at(best_pos[i, 0], best_pos[i, 1], best_theta[i]) - 1e-12:
                    best_theta[i] = new_th
                    improved = True
            # Recompute best_L
            cur_L = 0.0
            for i in range(N):
                cur_L = max(cur_L, min_L_at(best_pos[i, 0], best_pos[i, 1], best_theta[i]))
            best_L = cur_L

    # Final L computation
    final_L = 0.0
    for i in range(N):
        final_L = max(final_L, min_L_at(best_pos[i, 0], best_pos[i, 1], best_theta[i]))
    best_L = final_L

    # Build output
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]),
                      math.degrees(best_theta[i])])

    return inner, (0.0, 0.0), float(best_L), 0.0
