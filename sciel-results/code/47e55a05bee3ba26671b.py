"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def solve_min_enclosing_hexagon(pts, beta):
    """
    Solves the LP to find the minimum side length S of a regular hexagon
    with orientation beta that contains all points in 'pts'.
    
    Args:
        pts: np.array of shape (M, 2)
        beta: orientation angle in radians
    
    Returns:
        S: minimum side length, or inf if infeasible
    """
    if len(pts) == 0:
        return 0.0
        
    # Normals for the outer hexagon
    # The vertices of a unit hexagon (side 1) are at angles 0, 60, ...
    # The outward normals are at 30, 90, ...
    # With rotation beta, normals are at beta + 30 + k*60
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # Variables: cx, cy, D
    # D is the diameter (2 * apothem). S = D / SQRT3
    # Minimize D
    c_obj = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            # (px - cx)*nx + (py - cy)*ny <= D/2
            # -nx*cx - ny*cy - 0.5*D <= -px*nx - py*ny
            A_ub.append([-nx, -ny, -0.5])
            b_ub.append(-(px * nx + py * ny))
            
            # -(px - cx)*nx - (py - cy)*ny <= D/2
            # nx*cx + ny*cy - 0.5*D <= px*nx + py*ny
            A_ub.append([nx, ny, -0.5])
            b_ub.append(px * nx + py * ny)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            S = D / SQRT3
            return S
        else:
            return float('inf')
    except Exception:
        return float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Generate candidate arrangements from hex lattice
    # We use a pool of axial coordinates within a reasonable range
    axial_pool = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            if max(abs(q), abs(r), abs(q+r)) <= 4:
                axial_pool.append((q, r))
                
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
        
    # Candidate sets of 11 axial coordinates
    # We define several known compact shapes
    candidates = []
    
    # Shape 1: Center + Ring of 6 + 4 extensions
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extensions from ring:
    # (2,0), (1,1), (0,2), (-1,2) -- One side
    candidates.append([
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ])
    
    # Shape 2: More symmetric extension
    candidates.append([
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2)
    ])
    
    # Shape 3: Elongated
    candidates.append([
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2)
    ])
    
    # Shape 4: 
    candidates.append([
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (2,-1), (0,2)
    ])
    
    # Shape 5:
    candidates.append([
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-1,2), (1,1), (0,2)
    ])
    
    # Shape 6:
    candidates.append([
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,1), (1,2), (-1,2), (0,3)
    ])
        
    # Add random perturbations to lattice positions? 
    # No, lattice packing is usually optimal for hexagons due to symmetry.
    # We will just optimize the outer container for these fixed inner positions.
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # We need to return the inner hexagons' positions and angles.
    # Since we are using lattice positions with angle 0, we just convert axial to cartesian.
    # The outer center and angle are determined by the LP solution.
    # Note: The LP gives us the optimal center and side for a fixed angle.
    # We need to find the best angle.
    
    # Pre-compute points for each candidate
    candidate_pts = []
    for coords in candidates:
        pts = []
        for q, r in coords:
            cx, cy = axial_to_cartesian(q, r)
            # Get vertices of the unit hexagon at (cx, cy) with angle 0
            verts = get_hex_vertices(cx, cy, 1.0, 0.0)
            for v in verts:
                pts.append(v)
        candidate_pts.append(np.array(pts))
        
    # Optimize outer angle for each candidate
    for i, pts in enumerate(candidate_pts):
        if time.time() > deadline - 10:
            break
            
        def objective_angle(beta):
            return solve_min_enclosing_hexagon(pts, beta)
            
        # The function is periodic with period pi/6.
        # We search in [0, pi/6)
        # Use minimize_scalar for global search within the period
        try:
            res = minimize_scalar(objective_angle, bounds=(0, math.pi/6), method='bounded', 
                                  options={'maxiter': 100, 'xatol': 1e-10})
            if res.fun < best_side:
                best_side = res.fun
                best_outer_angle = res.x
                # We need to recover the center from the LP solution
                # Let's re-run the LP to get the center
                # Actually, solve_min_enclosing_hexagon only returns S.
                # We need a version that returns center too, or we re-solve.
                # Let's modify solve_min_enclosing_hexagon to return center as well for the final best.
                
                # For now, store the candidate index
                # We will re-solve at the end to get the center
                best_candidate_idx = i
        except Exception:
            continue
            
    if best_inner is None and best_candidate_idx is not None:
        # Re-solve to get center
        pts = candidate_pts[best_candidate_idx]
        # We need a helper to get center
        def solve_and_get_center(pts, beta):
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
                
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for nx, ny in normals:
                for px, py in pts:
                    A_ub.append([-nx, -ny, -0.5])
                    b_ub.append(-(px * nx + py * ny))
                    A_ub.append([nx, ny, -0.5])
                    b_ub.append(px * nx + py * ny)
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[0], res.x[1], res.x[2] / SQRT3
            return None, None, float('inf')
            
        cx, cy, S = solve_and_get_center(pts, best_outer_angle)
        if S < best_side:
            best_side = S
            best_outer_center = (cx, cy)
            
        # Construct inner list
        coords = candidates[best_candidate_idx]
        inner = []
        for q, r in coords:
            cx_h, cy_h = axial_to_cartesian(q, r)
            inner.append([cx_h, cy_h, 0.0])
        best_inner = inner
        
    # Fallback if no candidate worked
    if best_inner is None:
        # Use the first candidate
        coords = candidates[0]
        inner = []
        for q, r in coords:
            cx_h, cy_h = axial_to_cartesian(q, r)
            inner.append([cx_h, cy_h, 0.0])
        best_inner = inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        # Calculate side
        pts = candidate_pts[0]
        best_side = solve_min_enclosing_hexagon(pts, 0.0)
        
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
