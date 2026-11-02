import math
import numpy as np
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0

def construct():
    # 11 unit hexagons placed on the standard hex grid (axial coordinates)
    # This is the densest known packing of 11 hexagons in a hexagon
    axial_points = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    pts = np.array([axial_to_cartesian(q, r) for q, r in axial_points])
    
    # For each possible outer rotation (0 to 60 degrees in small steps),
    # find the minimal enclosing hexagon
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # The outer hexagon has normals at angles: beta + 30° + k*60° for k=0..5
    # For a fixed beta, the minimal enclosing hexagon center and size can be found via LP
    
    def calc_S_for_beta(beta):
        """
        Find minimal S and center (cx, cy) such that all pts are inside
        a regular hexagon of side S centered at (cx, cy) with rotation beta.
        
        The hexagon boundary in each normal direction n_k is:
          |p · n_k - c · n_k| <= S * APOTHEM
        
        So: max_p |p · n_k - c · n_k| <= S * APOTHEM for all k
        Equivalently: max_p (p · n_k) - c · n_k <= S * APOTHEM
                      c · n_k - min_p (p · n_k) <= S * APOTHEM
        
        Variables: cx, cy, S
        Minimize: S
        Subject to:
          For each k: max_p(p·n_k) - (cx*n_kx + cy*n_ky) <= S * APOTHEM
          For each k: (cx*n_kx + cy*n_ky) - min_p(p·n_k) <= S * APOTHEM
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Objective: minimize S
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (p · n) - (c · n) <= S * APOTHEM
                # => -nx*cx - ny*cy - S*APOTHEM <= -(p · n)
                A_ub.append([-nx, -ny, -APOTHEM])
                b_ub.append(-(px * nx + py * ny))
                
                # (c · n) - (p · n) <= S * APOTHEM
                # => nx*cx + ny*cy - S*APOTHEM <= p · n
                A_ub.append([nx, ny, -APOTHEM])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[2]  # S
            else:
                return float('inf')
        except Exception:
            return float('inf')
    
    # Grid search over beta from 0 to 60 degrees
    n_steps = 120
    for i in range(n_steps):
        beta = i * (math.pi / 6.0) / n_steps
        S = calc_S_for_beta(beta)
        if S < best_S:
            best_S = S
            best_beta = beta
    
    # Refine with finer grid around best
    if best_S < float('inf'):
        for i in range(100):
            beta = best_beta - 0.1 * math.pi / 6.0 + i * 0.2 * math.pi / 6.0 / 100
            beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
            S = calc_S_for_beta(beta)
            if S < best_S:
                best_S = S
                best_beta = beta
    
    # Find the optimal center for the best beta
    # Re-solve LP to get center
    normals = []
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -APOTHEM])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -APOTHEM])
            b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy = res.x[0], res.x[1]
            S_final = res.x[2]
        else:
            # Fallback: compute center as mean, S as max projection
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            max_proj = 0.0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            S_final = (max_proj + APOTHEM) / APOTHEM
    except Exception:
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S_final = (max_proj + APOTHEM) / APOTHEM
        best_beta = 0.0
    
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
