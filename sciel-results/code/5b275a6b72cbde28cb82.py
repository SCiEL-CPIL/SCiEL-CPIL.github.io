import math
import numpy as np
import time
from itertools import permutations
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# The 11 cells of the minimal 3x3 hexagonal lattice block (axial coordinates)
# These form a hexagon of "radius" 2 in axial coordinates
CELLS = [
    (0, 0),
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    (1, 1), (-1, 2), (-2, 1), (2, -1)
]

def axial_to_cartesian(axial_list):
    """Convert axial coordinates to Cartesian coordinates for unit hexagons."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def calc_enclosing_hexagon_side(pts, center, beta):
    """
    Calculate the side length of the smallest enclosing regular hexagon
    with center, rotation beta, that contains all points.
    
    Returns (S, D) where S is the side length and D is the apothem distance.
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    cx, cy = center
    for nx, ny in normals:
        for px, py in pts:
            proj = abs((px - cx) * nx + (py - cy) * ny)
            if proj > max_proj:
                max_proj = proj
    
    D = max_proj
    S = (D + APOTHEM) / APOTHEM
    return S, D

def optimize_center_and_rotation(pts):
    """
    For a given set of points, find the optimal center and rotation
    that minimizes the enclosing hexagon side length.
    Uses Linear Programming for exact center for a given angle, and scans angles.
    """
    best_S = float('inf')
    best_center = None
    best_beta = 0.0
    
    # The problem has 6-fold symmetry, so beta in [0, pi/6) is sufficient
    # Scan a range of angles
    n_beta = 60
    for i in range(n_beta):
        beta = i * (math.pi / 6.0) / n_beta
        
        # Normals for this beta
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # LP formulation:
        # Minimize D
        # Variables: c_x, c_y, D
        # Constraints:
        # (p_i - c) . n_k <= D  =>  -p_ix*n_kx - p_iy*n_ky + c_x*n_kx + c_y*n_ky - D <= 0
        # -(p_i - c) . n_k <= D =>  p_ix*n_kx + p_iy*n_ky - c_x*n_kx - c_y*n_ky - D <= 0
        
        A_ub = []
        b_ub = []
        
        for px, py in pts:
            for nx, ny in normals:
                # Constraint 1: (px - cx)nx + (py - cy)ny <= D
                # -nx*px - ny*py + nx*cx + ny*cy - D <= 0
                # Coeffs for [cx, cy, D]: [nx, ny, -1]
                A_ub.append([nx, ny, -1])
                b_ub.append(nx*px + ny*py)
                
                # Constraint 2: -(px - cx)nx - (py - cy)ny <= D
                # nx*px + ny*py - nx*cx - ny*cy - D <= 0
                # Coeffs for [cx, cy, D]: [-nx, -ny, -1]
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-(nx*px + ny*py))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Objective: minimize D => [0, 0, 1]
        c_obj = [0, 0, 1]
        
        # Bounds: cx, cy unbounded (or large), D >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                D_val = res.x[2]
                S_val = (D_val + APOTHEM) / APOTHEM
                if S_val < best_S:
                    best_S = S_val
                    best_center = (res.x[0], res.x[1])
                    best_beta = beta
        except Exception:
            pass

    # Refine beta around the best found using a finer grid if time permits
    if best_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Fine scan around best_beta
        step = (math.pi / 6.0) / 600
        start = max(0, best_beta - 2*step)
        end = min(math.pi / 6.0, best_beta + 2*step)
        for beta in np.arange(start, end, step):
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            A_ub = []
            b_ub = []
            for px, py in pts:
                for nx, ny in normals:
                    A_ub.append([nx, ny, -1])
                    b_ub.append(nx*px + ny*py)
                    A_ub.append([-nx, -ny, -1])
                    b_ub.append(-(nx*px + ny*py))
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            c_obj = [0, 0, 1]
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    D_val = res.x[2]
                    S_val = (D_val + APOTHEM) / APOTHEM
                    if S_val < best_S:
                        best_S = S_val
                        best_center = (res.x[0], res.x[1])
                        best_beta = beta
            except Exception:
                pass

    return best_S, best_center, best_beta

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    pts = axial_to_cartesian(CELLS)
    
    best_S, best_center, best_beta = optimize_center_and_rotation(pts)
    
    if best_center is None or best_S == float('inf'):
        best_center = (0.0, 0.0)
        best_beta = 0.0
        best_S = calc_enclosing_hexagon_side(pts, best_center, best_beta)[0]
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    S_final, D_final = calc_enclosing_hexagon_side(pts, best_center, best_beta)
    
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    S_scaled, _ = calc_enclosing_hexagon_side(pts_scaled, best_center, best_beta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
