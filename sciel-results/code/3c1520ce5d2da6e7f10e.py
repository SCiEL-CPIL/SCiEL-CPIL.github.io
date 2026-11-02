import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate configurations (axial coordinates)
    # We want compact shapes.
    candidates = []
    
    # 1. 3x3 Rhombus (9) + 2
    # Rhombus: q in [0,2], r in [0,2] is 9 points?
    # Axial: (0,0)..(2,2)
    # Let's add 2 to make it 11.
    # Best to add to the "middle" of the long side to keep it compact.
    c1 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (-1,1), (3,1) # Extend the middle row
    ]
    candidates.append(c1)
    
    # 2. Side-2 Hexagon (7) + 4
    # Center (0,0), Ring (1,0)..(1,-1)
    # Add 4 to the sides.
    c2 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2) # Extend 4 sides
    ]
    candidates.append(c2)
    
    # 3. 2x4 Rectangle (8) + 3
    # q in [0,3], r in [0,1] -> 8 points
    # Add 3 to make 11.
    c3 = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (1,2), (2,2), (1,-1) # Add to top and bottom
    ]
    candidates.append(c3)
    
    # 4. 3x3 Square (9) + 2
    # q in [0,2], r in [0,2] is a rhombus, not a square in Euclidean sense, but 3x3 grid.
    # Add 2 to corners?
    c4 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (-1,0), (3,2) # Add to opposite corners
    ]
    candidates.append(c4)
    
    # 5. 5+6 Strip
    c5 = [
        (0,0), (1,0), (2,0), (3,0), (4,0),
        (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)
    ]
    candidates.append(c5)
    
    # 6. 4+3+4
    c6 = [
        (0,0), (1,0), (2,0), (3,0),
        (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2),
        (1,-1) # Add one below
    ]
    # Wait, that's 12. Remove one.
    c6 = [
        (0,0), (1,0), (2,0), (3,0),
        (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2)
    ]
    # This is 11? 4+3+4 = 11. Yes.
    candidates.append(c6)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S(pts, beta):
        # pts is Nx2
        # Normals for outer hexagon at rotation beta
        # Outer hexagon sides are perpendicular to normals.
        # Normals are at angles beta + pi/6 + k*pi/3
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        # Project points onto normals
        # projs is Nx6
        projs = pts @ normals.T
        
        # Width in each direction
        widths = np.max(projs, axis=0) - np.min(projs, axis=0)
        
        # Required S for each direction
        # S * sqrt(3) >= width + sqrt(3) (apothem of inner hex)
        # S >= width/sqrt(3) + 1
        S_vals = widths / math.sqrt(3) + 1.0
        
        return np.max(S_vals)

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    for axial in candidates:
        pts = axial_to_cartesian(axial)
        
        # Find center (mean)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Grid search for beta
        best_local_S = float('inf')
        best_local_beta = 0.0
        
        # 100 steps in 0..30 deg
        for i in range(100):
            beta = i * (math.pi / 6.0) / 100.0
            S = calc_S(pts, beta)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = beta
        
        # Polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize_scalar(lambda b: calc_S(pts, b),
                                     bounds=(0, math.pi / 6.0), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_beta = res.x
            except Exception:
                pass

        if best_local_S < best_S:
            best_S = best_local_S
            best_pts = pts
            best_beta = best_local_beta
            best_center = (cx, cy)

    if best_pts is None:
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        best_pts = pts
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_beta = 0.0
        best_S = calc_S(pts, best_beta)

    # Finalize
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Scale points slightly to ensure fit?
    # The calc_S already accounts for the apothem.
    # The returned side length is best_S.
    # We just need to return the coordinates.
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
