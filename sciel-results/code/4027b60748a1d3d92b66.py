import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 580  # Leave 20s margin for 600s limit

    # Constants
    n_inner = 11
    epsilon_check = 1e-6
    epsilon_opt = 1e-5  # Stricter for optimization to ensure validity
    apothem_ratio = np.sqrt(3) / 2.0
    
    # Outer hexagon normals (for angle 0, flat top)
    # Vertices at 0, 60, 120...
    # Normals to edges:
    # Edge 0-1 (right-top): Normal is 30 deg?
    # Let's check. Vertex 0 (1,0), Vertex 1 (0.5, 0.866).
    # Midpoint (0.75, 0.433). Normal vector (0.866, 0.5) -> 30 deg.
    # Edge 1-2 (top): Normal 90 deg.
    # So normals are at 30, 90, 150, 210, 270, 330.
    outer_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    
    # Inner hexagon normals (same set)
    inner_normals = outer_normals

    def get_hex_vertices(x, y):
        angles = np.radians([0, 60, 120, 180, 240, 300])
        return np.column_stack([x + np.cos(angles), y + np.sin(angles)])

    def check_validity(inner, outer, side):
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11) - 1e-9:
            return False
            
        ox, oy, oa = outer
        inner_shifted = inner.copy()
        inner_shifted[:, 0] -= ox
        inner_shifted[:, 1] -= oy
        
        # Check edge lengths
        for i in range(11):
            v = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1])
            diffs = np.diff(v, axis=0, append=v[:1])
            lengths = np.linalg.norm(diffs, axis=1)
            if not np.all(np.abs(lengths - 1.0) < 1e-8):
                return False

        # Check non-overlap
        for i in range(11):
            vi = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1])
            for j in range(i+1, 11):
                vj = get_hex_vertices(inner_shifted[j, 0], inner_shifted[j, 1])
                separated = False
                for n in inner_normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    gap = min(min_j - max_i, min_i - max_j)
                    if gap > epsilon_check:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment
        oa_rad = np.radians(oa)
        outer_v = np.column_stack([
            ox + side * np.cos(oa_rad + np.radians(60*k)),
            oy + side * np.sin(oa_rad + np.radians(60*k))
        ] for k in range(6))
        
        for i in range(6):
            p1 = outer_v[i]
            p2 = outer_v[(i+1)%6]
            edge_vec = p2 - p1
            for i_h in range(11):
                vi = get_hex_vertices(inner_shifted[i_h, 0], inner_shifted[i_h, 1])
                for vp in vi:
                    c = np.cross(p2 - p1, vp - p1)
                    if c < -epsilon_check:
                        return False
        return True

    def compute_min_side(inner):
        max_proj = 0
        for n in outer_normals:
            for i in range(11):
                vi = get_hex_vertices(inner[i, 0], inner[i, 1])
                proj = np.dot(vi, n)
                max_proj = max(max_proj, np.max(proj))
        
        S = 2 * max_proj / np.sqrt(3)
        return S

    # Construct a valid "3x3 + 2" configuration with gaps
    # Use axial coordinates scaled by 1.01 to ensure gaps
    scale = 1.01
    # Axial coords for 3x3 grid centered at (0,0) in "flat top" layout?
    # Let's use Cartesian directly for simplicity.
    # Flat top hexes.
    # Row 0 (top): y = sqrt(3) * 1.5 * scale
    # Row 1 (mid): y = 0
    # Row 2 (bot): y = -sqrt(3) * 1.5 * scale
    
    # We need 11 hexes.
    # Let's place them in a hexagonal shape of radius 1 (7 hexes) and add 4 more.
    # Or just a 3x3 grid (9) + 2.
    
    # 3x3 grid of flat-top hexes:
    # x coords: -sqrt(3)*scale, 0, sqrt(3)*scale
    # y coords for x=0: -1.5*sqrt(3)*scale, 0, 1.5*sqrt(3)*scale
    # y coords for x!=0: -sqrt(3)*scale, 0, sqrt(3)*scale ? No.
    
    # Standard flat-top grid:
    # x = sqrt(3) * (q + r/2) * scale
    # y = 1.5 * r * scale
    
    # Let's pick 11 axial coords (q, r):
    # 0,0
    # 1,0; -1,0
    # 0,1; 0,-1
    # 1,1; -1,1; 1,-1; -1,-1
    # 0,2; 0,-2
    # This is a hexagon of radius 2? No, that's 19.
    
    # Let's use a compact 11-set:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1) -> 8 neighbors?
    # Distance from (0,0) to (1,1) is sqrt(3) * sqrt(1 + 0.25) ? No.
    # Axial distance formula: (|q| + |r| + |q+r|) / 2.
    # (1,1): (1+1+2)/2 = 2. Distance = 2 * sqrt(3) * scale.
    # (1,0): (1+0+1)/2 = 1. Distance = 1 * sqrt(3) * scale.
    
    # Let's use the 7 hexes of radius 1 (center + 6 neighbors) and 4 more at distance 2?
    # (2,0), (-2,0), (0,2), (0,-2).
    # Total 11.
    # This is a "plus" shape with a hex core.
    
    coords = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1),
        (2,0), (-2,0) # Only 10? 
        # Wait. 1 center + 6 neighbors + 4 more.
        # Neighbors of (0,0) in flat top axial:
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Let's list 11:
        # (0,0)
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # (2,0), (2,-1), (0,-2), (0,2) ?
    ]
    
    # Let's just use a simple grid that is definitely valid.
    # 3x3 grid of hexes (9) + 2 more at the sides.
    # Axial coords:
    # q in [-1, 0, 1], r in [-1, 0, 1] -> 9 hexes.
    # Add (2,0) and (-2,0).
    
    axial_coords = [
        (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1),
        (2,0), (-2,0)
    ]
    
    inner = np.zeros((11, 3))
    for i, (q, r) in enumerate(axial_coords):
        x = np.sqrt(3) * (q + r/2) * scale
        y = 1.5 * r * scale
        inner[i, 0] = x
        inner[i, 1] = y
        inner[i, 2] = 0

    outer = np.array([0, 0, 0])
    side = compute_min_side(inner)
    
    # Verify validity
    if not check_validity(inner, outer, side):
        # If still invalid, increase scale
        scale = 1.05
        for i, (q, r) in enumerate(axial_coords):
            x = np.sqrt(3) * (q + r/2) * scale
            y = 1.5 * r * scale
            inner[i, 0] = x
            inner[i, 1] = y
        side = compute_min_side(inner)
        if not check_validity(inner, outer, side):
            # Fallback: return a very large valid config
            scale = 2.0
            for i, (q, r) in enumerate(axial_coords):
                x = np.sqrt(3) * (q + r/2) * scale
                y = 1.5 * r * scale
                inner[i, 0] = x
                inner[i, 1] = y
            side = compute_min_side(inner)

    best_result = (inner, outer, side)
    best_score = 1.0 / side

    # Optimization
    def objective(x):
        return x[-1]

    def make_containment_constraint(i, k):
        n = outer_normals[k]
        hex_v = get_hex_vertices(0, 0)
        max_proj = np.max(np.dot(hex_v, n))
        
        def constraint(x):
            S = x[-1]
            cx, cy = x[2*i], x[2*i+1]
            return S * apothem_ratio - (cx * n[0] + cy * n[1] + max_proj)
        return constraint

    def make_non_overlap_constraint(i, j):
        def constraint(x):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            
            vi = get_hex_vertices(ci[0], ci[1])
            vj = get_hex_vertices(cj[0], cj[1])
            
            max_gap = -np.inf
            for n in inner_normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                gap = min(min_j - max_i, min_i - max_j)
                if gap > max_gap:
                    max_gap = gap
            return max_gap - epsilon_opt
        
        return constraint

    cons = []
    for i in range(11):
        for k in range(6):
            cons.append({'type': 'ineq', 'fun': make_containment_constraint(i, k)})
            
    for i in range(11):
        for j in range(i+1, 11):
            cons.append({'type': 'ineq', 'fun': make_non_overlap_constraint(i, j)})

    x0 = np.zeros(23)
    for i in range(11):
        x0[2*i] = inner[i, 0]
        x0[2*i+1] = inner[i, 1]
    x0[-1] = side

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=[(-10, 10)]*22 + [(2, 10)], constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})
        if res.success and res.fun < side:
            x_opt = res.x
            inner_opt = np.zeros((11, 3))
            for i in range(11):
                inner_opt[i, 0] = x_opt[2*i]
                inner_opt[i, 1] = x_opt[2*i+1]
                inner_opt[i, 2] = 0
            outer_opt = np.array([0, 0, 0])
            side_opt = x_opt[-1]
            
            if check_validity(inner_opt, outer_opt, side_opt):
                score_opt = 1.0 / side_opt
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = (inner_opt, outer_opt, side_opt)
    except Exception as e:
        pass

    return best_result
