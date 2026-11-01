"""
Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.
Combines L-BFGS-B for inner position optimization with Linear Programming for exact outer container sizing.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# Precompute hexagon vertex offsets for unit hexagon at angle 0
_HEX_ANGLES = np.pi / 3.0 * np.arange(6)
_HEX_COS = np.cos(_HEX_ANGLES)
_HEX_SIN = np.sin(_HEX_ANGLES)

# Outer hexagon support normals (inward-pointing) relative to local frame
# The outer hexagon vertices are at angles 0, 60, 120... 
# The normals to the sides are at 30, 90, 150...
_OUTER_NORMALS = np.array([[math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] for i in range(6)])

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + _HEX_ANGLES
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts_list.append(get_hex_vertices(ix, iy, 1.0, math.radians(iangle)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def solve_outer_lp(inner_positions, timeout=0.5):
    """
    Solve LP to find minimum outer hexagon size containing inner hexagons.
    inner_positions: list of (x, y) for N hexagons.
    Returns (S, ox, oy, angle_rad) or None.
    """
    start_time = time.time()
    
    # We need to check all 6 possible orientations of the outer hexagon?
    # No, the LP can optimize the rotation if we parameterize it, but LP is linear.
    # The constraint is:
    # Max over k=0..5 of | (p_i - c) . n_k(theta) | <= D
    # This is non-linear in theta.
    # However, for a fixed set of inner hexagons, the optimal outer hexagon
    # will likely have its sides aligned with the "extreme" directions of the cluster.
    # 
    # Strategy: Try a few candidate rotations for the outer hexagon.
    # The most likely candidates are 0, 30, 60, 90, 120, 150 degrees.
    # Or, we can iterate: 
    # 1. Assume rotation theta.
    # 2. Solve LP for c, D.
    # 3. The active constraints define the "corners".
    # 
    # A simpler and robust approach for this specific problem:
    # The optimal rotation is likely one of the 6 symmetries of the hexagon grid.
    # Let's test 6 rotations: 0, 30, 60, 90, 120, 150 degrees.
    
    best_S = float('inf')
    best_params = None
    
    # Candidate rotations in radians
    candidate_angles = [k * math.pi / 6.0 for k in range(6)]
    
    for theta in candidate_angles:
        if time.time() - start_time > timeout:
            break
            
        # Normals for this rotation
        # The normals of the outer hexagon at rotation theta are at theta + 30 + k*60
        normals = np.array([
            [math.cos(theta + math.pi/6 + k*math.pi/3), math.sin(theta + math.pi/6 + k*math.pi/3)] 
            for k in range(6)
        ])
        
        # Variables: ox, oy, D
        # Objective: Minimize D
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        # Constraints:
        # For each point p_i and each normal n_k:
        # (p_i - c) . n_k <= D  =>  -c . n_k - D <= -p_i . n_k
        # -(p_i - c) . n_k <= D => c . n_k - D <= p_i . n_k
        
        for px, py in inner_positions:
            for nx, ny in normals:
                # 1. (px - ox)nx + (py - oy)ny <= D
                # -ox*nx - oy*ny - D <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-px*nx - py*ny)
                
                # 2. -((px - ox)nx + (py - oy)ny) <= D
                # ox*nx + oy*ny - D <= px*nx + py*ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px*nx + py*ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                # S is the side length.
                # D is the distance from center to side (apothem of outer hexagon).
                # Apothem = S * sqrt(3) / 2.
                # So S = 2 * D / sqrt(3).
                S = 2.0 * D / SQRT3
                if S < best_S:
                    best_S = S
                    best_params = (res.x[0], res.x[1], theta)
        except Exception:
            pass
            
    if best_params is None:
        return None
        
    ox, oy, theta = best_params
    return best_S, ox, oy, math.degrees(theta) % 360.0

def objective_hybrid(params):
    """
    Objective function for L-BFGS-B.
    params: [cx1, cy1, ang1, ..., cxN, cyN, angN]
    We optimize inner positions.
    The outer container is solved exactly via LP for the current inner positions.
    """
    n = N
    inner_positions = []
    inner_angles = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner_positions.append((cx, cy))
        inner_angles.append(ang)
        
    # Solve LP for outer container
    try:
        res = solve_outer_lp(inner_positions, timeout=0.2)
        if res is None:
            return 1e6
        S, ox, oy, outer_angle = res
    except Exception:
        return 1e6
        
    # Check for overlaps
    penalty = 0.0
    verts_list = []
    for i in range(n):
        verts_list.append(get_hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, math.radians(inner_angles[i])))
    
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1 = inner_positions[i]
            cx2, cy2 = inner_positions[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                if sat_overlap_check(verts_list[i], verts_list[j], eps=1e-8):
                    penalty += (2.0 - d) ** 2 * 100.0
                    # If overlap is severe, return large value early
                    if penalty > 10.0:
                        return S + penalty

    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Axial coordinates for 11 hexagons in a compact cluster
    # Center + 6 in ring 1 + 4 in ring 2
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # Arrangement 1: Standard hexagonal lattice, scale 1.0
    scale1 = 1.0 + 1e-6
    centers1 = [axial_to_cartesian(q, r, scale1) for q, r in axial_coords]
    ic1 = np.zeros(3 * N)
    for i, (px, py) in enumerate(centers1):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    initial_conditions.append(ic1)
    
    # Arrangement 2: Slightly tighter scale
    scale2 = 0.99
    centers2 = [axial_to_cartesian(q, r, scale2) for q, r in axial_coords]
    ic2 = np.zeros(3 * N)
    for i, (px, py) in enumerate(centers2):
        ic2[3*i] = px
        ic2[3*i+1] = py
        ic2[3*i+2] = 0.0
    initial_conditions.append(ic2)
    
    # Arrangement 3: Perturbed standard lattice
    np.random.seed(42)
    ic3 = ic1 + np.random.randn(3*N) * 0.05
    initial_conditions.append(ic3)
    
    # Arrangement 4: Perturbed with different seed
    np.random.seed(123)
    ic4 = ic1 + np.random.randn(3*N) * 0.1
    initial_conditions.append(ic4)
    
    # Arrangement 5: Rotated inner hexagons slightly
    ic5 = ic1.copy()
    for i in range(N):
        ic5[3*i+2] = np.radians(10)
    initial_conditions.append(ic5)
    
    # Arrangement 6: Different cluster shape? 
    # Let's try a more linear arrangement to see if it helps escape local minima
    # Though for N=11, the compact cluster is usually best.
    # Let's stick to variations of the compact cluster.
    
    best_result = None
    best_score = float('inf')
    
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 10:
            break
            
        try:
            # Optimize inner positions
            # L-BFGS-B is good for this
            res = minimize(objective_hybrid, ic, 
                          method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                # Extract parameters
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                
                # Solve LP again to get exact outer params
                inner_positions = [(p[0], p[1]) for p in inner]
                lp_res = solve_outer_lp(inner_positions, timeout=1.0)
                if lp_res is not None:
                    S, ox, oy, outer_angle = lp_res
                    
                    # Validate
                    if check_validity(inner, (ox, oy), S, outer_angle):
                        if S < best_score:
                            best_score = S
                            best_result = (inner, (ox, oy), S, outer_angle)
                            
        except Exception:
            pass
            
    # Fallback
    if best_result is None:
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = 4.0 # Rough estimate
        # Try to solve LP for fallback
        lp_res = solve_outer_lp([(p[0], p[1]) for p in inner], timeout=1.0)
        if lp_res:
            S, ox, oy, outer_angle = lp_res
        else:
            ox, oy, outer_angle = 0.0, 0.0, 0.0
        best_result = (inner, (ox, oy), S, outer_angle)
        
    # Final polish: Try to refine the best result found with a tighter tolerance
    if time.time() < deadline - 5 and best_result is not None:
        try:
            inner, (ox, oy), S, outer_angle = best_result
            params = np.zeros(3*N)
            for i, (cx, cy, ang) in enumerate(inner):
                params[3*i] = cx
                params[3*i+1] = cy
                params[3*i+2] = np.radians(ang)
                
            # Run a short, high-precision optimization
            res = minimize(objective_hybrid, params, 
                          method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                
                inner_positions = [(p[0], p[1]) for p in inner]
                lp_res = solve_outer_lp(inner_positions, timeout=1.0)
                if lp_res is not None:
                    S, ox, oy, outer_angle = lp_res
                    if check_validity(inner, (ox, oy), S, outer_angle):
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
