import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590

    sqrt3 = np.sqrt(3)

    # Precompute outer hexagon normals (for containment check)
    # Outer hexagon centered at (0,0), side S, angle 0 (vertices at 0, 60, ...)
    # Normals are at 30, 90, ...
    outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])

    def get_hex_vertices(cx, cy, angle_deg):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def get_all_vertices(params):
        """
        params: (33,) array [x1, y1, a1, x2, y2, a2, ...]
        returns: (11, 6, 2) array of vertices
        """
        verts = np.zeros((11, 6, 2))
        for i in range(11):
            cx = params[i*3]
            cy = params[i*3 + 1]
            ang = params[i*3 + 2]
            verts[i] = get_hex_vertices(cx, cy, ang)
        return verts

    def calc_S(verts):
        """Calculate minimum side length S of outer hexagon (angle 0, center 0,0) to contain verts."""
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        global_max = np.max(max_proj)
        return 2.0 * global_max / sqrt3

    def overlap_penalty(params):
        """
        Fast overlap penalty based on center distances.
        Min center distance for non-overlap is sqrt(3).
        """
        p = params.reshape(11, 3)
        xs = p[:, 0]
        ys = p[:, 1]
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < sqrt3:
                    penalty += (sqrt3 - d)**2
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        verts = get_all_vertices(params)
        S = calc_S(verts)
        pen = overlap_penalty(params)
        return S + 100.0 * pen

    def exact_check(params):
        """
        Perform exact validation.
        Returns (is_valid, S, min_gap)
        """
        verts = get_all_vertices(params)
        S = calc_S(verts)
        
        # Check separation with SAT
        min_gap = np.inf
        for i in range(11):
            v1 = verts[i]
            for j in range(i+1, 11):
                v2 = verts[j]
                # SAT check
                # Get normals for v1
                edges1 = np.roll(v1, -1, axis=0) - v1
                normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
                norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
                norms1[norms1 < 1e-12] = 1.0
                normals1 /= norms1
                
                # Get normals for v2
                edges2 = np.roll(v2, -1, axis=0) - v2
                normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
                norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
                norms2[norms2 < 1e-12] = 1.0
                normals2 /= norms2
                
                all_normals = np.vstack([normals1, normals2])
                
                proj1 = v1 @ all_normals.T
                proj2 = v2 @ all_normals.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                gaps = np.maximum(min2 - max1, min1 - max2)
                gap = np.min(gaps)
                if gap < min_gap:
                    min_gap = gap
        
        if min_gap < 1e-6:
            return False, S, min_gap
            
        # Check edge lengths
        for i in range(11):
            for j in range(6):
                p1 = verts[i, j]
                p2 = verts[i, (j+1)%6]
                if abs(np.linalg.norm(p2-p1) - 1.0) > 1e-8:
                    return False, S, min_gap
                    
        # Check containment (cross product check as per spec)
        # Outer vertices
        outer_verts = get_hex_vertices(0, 0, 0) * (S) # Wait, get_hex_vertices returns unit hex.
        # Actually, let's just use the projection check which is equivalent for convex sets.
        # The spec says: cross(p2 - p1, p - p1) >= -1e-6 for each outer edge.
        # This is the standard half-plane check.
        # My calc_S ensures that the max projection on normals is <= S*sqrt(3)/2.
        # This is equivalent to being inside the hexagon.
        # However, to be safe, let's verify the cross product condition explicitly for the final result.
        
        # Outer vertices of the bounding hexagon
        outer_angles = np.deg2rad(np.arange(0, 360, 60))
        outer_v = np.column_stack([S * np.cos(outer_angles), S * np.sin(outer_angles)])
        
        for i in range(11):
            for p in verts[i]:
                for k in range(6):
                    p1 = outer_v[k]
                    p2 = outer_v[(k+1)%6]
                    cross = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
                    if cross < -1e-6:
                        return False, S, min_gap

        return True, S, min_gap

    def get_initial_guesses():
        guesses = []
        
        # Guess 1: Symmetric 1+6+4 (Honeycomb-like)
        p = np.zeros(33)
        p[0:3] = [0, 0, 0]
        
        d1 = sqrt3
        for k in range(6):
            ang = np.deg2rad(k * 60)
            idx = (k+1)*3
            p[idx] = d1 * np.cos(ang)
            p[idx+1] = d1 * np.sin(ang)
            p[idx+2] = 0 
            
        # Positions for the 4 outer hexes in a honeycomb arrangement
        # Center (0,0), Ring 1 (6 hexes), Ring 2 (4 hexes at "corners" of the hex grid)
        # In axial coords (q,r) with pointy top:
        # (0,0) -> (0,0)
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2 candidates: (2,-1), (1,1), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2)
        # We want 4 more. Let's pick symmetric ones.
        # (2,-1) -> x = sqrt3*(2-0.5) = 1.5*sqrt3, y = 1.5*(-1) = -1.5
        # (1,1) -> x = sqrt3*(1+0.5) = 1.5*sqrt3, y = 1.5
        # (-1,2) -> x = sqrt3*(-1+1) = 0, y = 3.0
        # (-2,1) -> x = sqrt3*(-2+0.5) = -1.5*sqrt3, y = 1.5
        # This set {(2,-1), (1,1), (-1,2), (-2,1)} is not symmetric under 60 deg rotation.
        # Let's try: (2,0), (0,2), (-2,0), (0,-2) in "flat top" axial?
        # Let's just use Cartesian distances.
        # Distance from center to ring 2 in honeycomb is 2*sqrt(3) along axes?
        # No, let's just use the positions from the seed which worked, but maybe rotated?
        # Seed used: (2*sqrt3, 0), (sqrt3, 3), (-2*sqrt3, 0), (-sqrt3, -3)
        
        positions_r2 = [
            (2*sqrt3, 0),
            (sqrt3, 3),
            (-2*sqrt3, 0),
            (-sqrt3, -3)
        ]
        
        for k in range(4):
            idx = (k+7)*3
            p[idx] = positions_r2[k][0]
            p[idx+1] = positions_r2[k][1]
            p[idx+2] = 0
            
        guesses.append(p)
        
        # Guess 2: Jittered
        p2 = p.copy()
        np.random.seed(42)
        p2[0:33:3] += np.random.uniform(-0.1, 0.1, 11)
        p2[1:33:3] += np.random.uniform(-0.1, 0.1, 11)
        p2[2:33:3] += np.random.uniform(-10, 10, 11)
        guesses.append(p2)
        
        # Guess 3: Rotated angles
        p3 = p.copy()
        p3[2:33:3] += np.random.uniform(-5, 5, 11)
        guesses.append(p3)
        
        # Guess 4: More aggressive jitter
        p4 = p.copy()
        np.random.seed(123)
        p4[0:33:3] += np.random.uniform(-0.2, 0.2, 11)
        p4[1:33:3] += np.random.uniform(-0.2, 0.2, 11)
        p4[2:33:3] += np.random.uniform(-15, 15, 11)
        guesses.append(p4)

        return guesses

    best_S = 1e10
    best_params = None
    best_valid = False

    guesses = get_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline - 60:
            break
            
        try:
            # Use Nelder-Mead with tight tolerances, as suggested by inspiration
            # It handles non-smoothness better than Powell in some cases
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
                
            # Check if this result is valid
            valid, S_val, gap_val = exact_check(res.x)
            if valid:
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x.copy()
                    best_valid = True
                    
        except Exception as e:
            pass

    # If we have a valid solution, try to refine it further
    if best_valid:
        try:
            # Run one more pass with even tighter tolerances on the best solution
            res2 = minimize(objective, best_params, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            valid, S_val, gap_val = exact_check(res2.x)
            if valid and S_val < best_S:
                best_S = S_val
                best_params = res2.x.copy()
        except Exception:
            pass

    # Fallback to a known valid packing if optimization failed
    if best_params is None or not best_valid:
        # Use the initial guess 1 which is likely valid or close
        p = guesses[0]
        valid, S_val, gap_val = exact_check(p)
        if valid:
            best_params = p
            best_S = S_val
            best_valid = True
        else:
            # Construct a very safe, large hexagon
            inner_hex_data = np.array(
                [
                    [0, 0, 0],
                    [-2.5, 0, 0],
                    [2.5, 0, 0],
                    [-1.25, 2.17, 0],
                    [1.25, 2.17, 0],
                    [-1.25, -2.17, 0],
                    [1.25, -2.17, 0],
                    [-3.75, 2.17, 0],
                    [3.75, 2.17, 0],
                    [-3.75, -2.17, 0],
                    [3.75, -2.17, 0],
                ]
            )
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 8.0
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Format output
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
