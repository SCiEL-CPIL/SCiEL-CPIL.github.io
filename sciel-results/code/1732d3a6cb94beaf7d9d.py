import math
import numpy as np
import time
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    SQRT3 = math.sqrt(3)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def compute_S_aligned(pts):
        """Compute S for aligned hexagons (beta=0), center at centroid."""
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        centered = pts - np.array([cx, cy])
        # 6 normals for beta=0: angles pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
        normals = np.array([
            [math.cos(math.pi/6), math.sin(math.pi/6)],
            [math.cos(math.pi/2), math.sin(math.pi/2)],
            [math.cos(5*math.pi/6), math.sin(5*math.pi/6)],
            [math.cos(7*math.pi/6), math.sin(7*math.pi/6)],
            [math.cos(3*math.pi/2), math.sin(3*math.pi/2)],
            [math.cos(11*math.pi/6), math.sin(11*math.pi/6)],
        ])
        projs = centered @ normals.T  # (n, 6)
        D = float(np.max(np.abs(projs)))
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy)

    def compute_S_optimal_center(pts, beta=0.0):
        """Optimal center for fixed beta: minimize max |projection|."""
        # For fixed beta, optimal center minimizes max over normals of max|proj|
        # This is a 2D LP. But for aligned hexagons, centroid is often optimal.
        # Try centroid first, then small perturbation search.
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        S0, _ = compute_S_aligned(pts)
        
        # Quick perturbation: try small offsets
        best_S = S0
        best_c = (cx0, cy0)
        for dx in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            for dy in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                c = (cx0 + dx, cy0 + dy)
                centered = pts - np.array(c)
                normals = np.array([
                    [math.cos(math.pi/6), math.sin(math.pi/6)],
                    [math.cos(math.pi/2), math.sin(math.pi/2)],
                    [math.cos(5*math.pi/6), math.sin(5*math.pi/6)],
                    [math.cos(7*math.pi/6), math.sin(7*math.pi/6)],
                    [math.cos(3*math.pi/2), math.sin(3*math.pi/2)],
                    [math.cos(11*math.pi/6), math.sin(11*math.pi/6)],
                ])
                projs = centered @ normals.T
                D = float(np.max(np.abs(projs)))
                S = (D + APOTHEM) / APOTHEM
                if S < best_S:
                    best_S = S
                    best_c = c
        return best_S, best_c

    def compute_S_with_rotation(pts, beta):
        """Compute S for given rotation beta, center at centroid."""
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        centered = pts - np.array([cx0, cy0])
        normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        projs = centered @ normals.T
        D = float(np.max(np.abs(projs)))
        S = (D + APOTHEM) / APOTHEM
        return S, (cx0, cy0)

    # --- Generate diverse candidate configurations ---
    candidates = []

    # Hexagonal cluster (radius 1 + extras)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Star: center + 6 neighbors + 4 opposite
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # 3x3 block + 2 extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # 4x3 block + 1 extra
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # 3x4 block + 1 extra
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # 3x3 + 2 top
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    # Cross-like
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # 5+6 row
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    # Center + 6 + 4 asymmetric
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # 3x3 + 2 top (variant)
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # 2+4+4+1
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # 4+2+2+2+1
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # Center + 6 + 4 (variant)
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Center + 6 + 4 (variant)
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # 3x3 with extras
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Staircase
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Center + 6 + 4 (variant)
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # 4+3+3
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # 2+3+2+3+1
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Center + 6 + 4 (variant)
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # 4+4+3
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    # 3+4+3+1
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    # Center + 6 + 4 (variant)
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    # 2+2+3+2+2
    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    # 3+2+3+3
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    # 4+3+4+2
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    # Center + 6 + 4 (variant)
    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    # 2+3+2+3+3
    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # More diverse configurations
    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 1), (0, 2), (1, -2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, 2)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    # --- Evaluate all candidates quickly ---
    best_S = float('inf')
    best_pts = None
    best_center = None
    best_beta = 0.0

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        
        # Try aligned (beta=0) with centroid
        S_aligned, c_aligned = compute_S_aligned(pts)
        if S_aligned < best_S:
            best_S = S_aligned
            best_pts = pts
            best_center = c_aligned
            best_beta = 0.0
        
        # Try a few rotations
        for beta_deg in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = math.radians(beta_deg)
            S_rot, c_rot = compute_S_with_rotation(pts, beta)
            if S_rot < best_S:
                best_S = S_rot
                best_pts = pts
                best_center = c_rot
                best_beta = beta

    # --- Refine best candidate with center optimization ---
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Try perturbation around best center for beta=0
        S_refined, c_refined = compute_S_optimal_center(best_pts, 0.0)
        if S_refined < best_S:
            best_S = S_refined
            best_center = c_refined
            best_beta = 0.0
        
        # Try perturbation for best_beta if not 0
        if best_beta != 0.0:
            cx0, cy0 = best_center
            for dx in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                for dy in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    c = (cx0 + dx, cy0 + dy)
                    centered = best_pts - np.array(c)
                    normals = np.array([
                        [math.cos(best_beta + math.pi/6 + k*math.pi/3), math.sin(best_beta + math.pi/6 + k*math.pi/3)]
                        for k in range(6)
                    ])
                    projs = centered @ normals.T
                    D = float(np.max(np.abs(projs)))
                    S = (D + APOTHEM) / APOTHEM
                    if S < best_S:
                        best_S = S
                        best_center = c

    # --- Fallback ---
    if best_center is None or best_S == float('inf'):
        best_pts = axial_to_cartesian(candidates[0])
        best_center = (float(np.mean(best_pts[:, 0])), float(np.mean(best_pts[:, 1])))
        best_beta = 0.0
        S_aligned, _ = compute_S_aligned(best_pts)
        best_S = S_aligned

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # --- Final: compute exact S with tiny safety scale ---
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    normals = np.array([
        [math.cos(best_beta + math.pi/6 + k*math.pi/3), math.sin(best_beta + math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    centered = pts_scaled - np.array([cx, cy])
    projs = centered @ normals.T
    D_scaled = float(np.max(np.abs(projs)))
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
