import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 2.0, 0.0

    # Define candidate point sets (axial coordinates for hex lattice)
    # Each candidate is a set of 11 axial coordinates (q, r)
    candidates = []

    # Candidate 1: Center + 6 neighbors + 4 from second ring
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    candidates.append(axial_1)

    # Candidate 2: Center + 6 neighbors + 4 different second ring positions
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_2)

    # Candidate 3: Linear-ish arrangement
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # Candidate 4: 4-4-3 arrangement
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # Candidate 5: 3-4-4 arrangement
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_5)

    # Candidate 6: 3-3-3-2 arrangement
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    # Candidate 7: Cross pattern
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # Candidate 8: Long line
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    # Candidate 9: Another ring variant
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # Candidate 10: 3-3-3-2 variant
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # Candidate 11: 2-5-4 arrangement
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # Candidate 12: Staircase
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # Candidate 13: Asymmetric ring
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Candidate 14: Another asymmetric
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # Candidate 15: 3-2-3-3 arrangement
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Candidate 16: 2-3-2-3-1
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Candidate 17: Ring with gaps
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # Candidate 18: 4-3-3-1
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # Candidate 19: 2-3-3-2-1
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Candidate 20: Symmetric ring variant
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # Candidate 21: 4-4-3 variant
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    # Candidate 22: 3-4-3-1
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    # Candidate 23: Plus sign
    axial_23 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    # Candidate 24: 2-2-3-2-2
    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    # Candidate 25: 3-2-3-3
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """Use LP to find minimum container size for given orientation beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D (half-width in each direction)
        # Minimize D
        # Constraints: for each normal (nx, ny) and each point (px, py):
        #   (px - cx) * nx + (py - cy) * ny <= D
        #   -(px - cx) * nx - (py - cy) * ny <= D
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_shape(pts):
        """Optimize container orientation for a given point set."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_D = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # Try 37 orientations (every 1.5 degrees)
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
                best_D = D
        
        # Refine with local search around best
        if best_S < float('inf') and best_center is not None:
            for i in range(11):
                beta = best_beta + (i - 5) * (math.pi / 6) / 36 / 5
                S, center, D = calc_S_for_beta(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = center
                    best_D = D
        
        return best_S, best_beta, best_center, best_D

    def check_disjoint(pts):
        """Check if all hexagons are disjoint (centers at least sqrt(3) apart)."""
        min_dist = math.sqrt(3) - 1e-6
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                if dx * dx + dy * dy < min_dist * min_dist:
                    return False
        return True

    def check_fit(pts, center, S, beta):
        """Check if all hexagons fit inside the container."""
        cx, cy = center
        for i in range(6):
            ang = beta + math.pi / 6.0 + i * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in pts:
                proj = (px - cx) * nx + (py - cy) * ny
                if abs(proj) > S * APOTHEM + 1e-6:
                    return False
        return True

    # Main optimization loop
    best_S = float('inf')
    best_pts = None
    best_center = None
    best_beta = 0.0

    for cand_idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        
        # Center the point set
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts = pts - np.array([cx, cy])
        
        # Check disjointness
        if not check_disjoint(pts):
            continue
        
        # Optimize container
        S, beta, center, D = optimize_shape(pts)
        
        if S < best_S and center is not None:
            # Verify fit
            if check_fit(pts, center, S, beta):
                best_S = S
                best_pts = pts.copy()
                best_center = center
                best_beta = beta
    
    # If we found a good solution, try to improve with local perturbation
    if best_pts is not None and best_S < float('inf'):
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 20:
            # Try perturbing points slightly to see if we can improve
            for attempt in range(5):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 10:
                    break
                
                perturbed = best_pts + np.random.randn(N, 2) * 0.01
                # Re-center
                cx = np.mean(perturbed[:, 0])
                cy = np.mean(perturbed[:, 1])
                perturbed = perturbed - np.array([cx, cy])
                
                if not check_disjoint(perturbed):
                    continue
                
                S, beta, center, D = optimize_shape(perturbed)
                if S < best_S and center is not None and check_fit(perturbed, center, S, beta):
                    best_S = S
                    best_pts = perturbed.copy()
                    best_center = center
                    best_beta = beta
    
    # Construct the final answer
    if best_pts is not None:
        inner = []
        for i in range(N):
            x = best_pts[i, 0] - best_center[0]
            y = best_pts[i, 1] - best_center[1]
            inner.append([x, y, 0.0])
        
        # The outer hexagon is centered at best_center
        # But we need to express it in the same coordinate system
        # The inner hexagons are relative to the outer center
        # So inner[i] = (x_i - cx, y_i - cy)
        
        return inner, list(best_center), best_S, math.degrees(best_beta)
    else:
        # Fallback: simple hexagonal arrangement
        sp = math.sqrt(3)
        fallback_inner = []
        positions = [
            (0, 0),
            (sp, 0), (sp/2, sp*1.5), (-sp/2, sp*1.5), (-sp, 0), (-sp/2, -sp*1.5), (sp/2, -sp*1.5),
            (2*sp, 0), (0, 2*sp*1.5), (-2*sp, sp*1.5), (2*sp, -sp*1.5)
        ]
        for x, y in positions[:N]:
            fallback_inner.append([x, y, 0.0])
        
        # Compute container size
        best_S = float('inf')
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(np.array([[x, y] for x, y, _ in fallback_inner]), beta)
            if S < best_S:
                best_S = S
                best_center = center
                best_beta = beta
        
        if best_center is not None:
            inner = []
            for x, y, _ in fallback_inner:
                inner.append([x - best_center[0], y - best_center[1], 0.0])
            return inner, list(best_center), best_S, math.degrees(best_beta)
        else:
            return fallback_inner, [0.0, 0.0], 3.0, 0.0
