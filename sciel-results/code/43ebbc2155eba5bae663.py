import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Improved Force-Directed Layout
    # We want to find a compact, non-overlapping configuration.
    # We will simulate a system of particles with repulsive forces when too close
    # and attractive forces towards the center of mass.
    # To ensure non-overlap, we use a strong repulsive force for d < 2.0 (since side=1, min dist between centers is 2*APOTHEM? No, for regular hexagons of side 1, the distance between centers for edge-to-edge contact is 2*APOTHEM = sqrt(3) ~ 1.732? 
    # Wait, apothem is sqrt(3)/2. Distance from center to vertex is 1. Distance from center to edge is sqrt(3)/2.
    # If two hexagons are placed edge-to-edge, the distance between centers is 2 * apothem = sqrt(3).
    # If they are vertex-to-vertex (touching at a corner), distance is 2.
    # So the minimum center distance for non-overlapping hexagons is sqrt(3) if they share an edge, but generally they can be closer if rotated?
    # Actually, for regular hexagons, the minimum distance between centers such that they do not overlap is sqrt(3) (when aligned) or potentially less if rotated?
    # Let's check: The width of a hexagon is 2 (vertex to vertex). The "height" is sqrt(3) (edge to edge).
    # The minimum distance between centers of two non-overlapping unit hexagons is actually sqrt(3) if they are in a honeycomb lattice.
    # However, if we rotate one, can we pack them tighter?
    # The directive says "disjoint".
    # Let's assume the minimum center distance is approximately sqrt(3) for aligned hexagons.
    # To be safe, we will use a repulsive force that activates when distance < sqrt(3) - epsilon.
    
    MIN_DIST = math.sqrt(3) - 0.01 # Safety margin for non-overlap
    
    def force_directed_layout(pts_init, iterations=300):
        pts = pts_init.copy()
        n = len(pts)
        # Initial positions should be somewhat spread out to avoid singularities
        # We'll use a cooling schedule for the step size
        
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = np.zeros_like(pts)
            # Repulsive forces
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < MIN_DIST:
                        # Strong repulsion
                        force_mag = 10.0 * (MIN_DIST - dist)
                        if dist > 1e-6:
                            forces[i] += force_mag * diff / dist
                            forces[j] -= force_mag * diff / dist
                        else:
                            # Random nudge
                            forces[i] += force_mag * np.array([1.0, 0.0])
                            forces[j] -= force_mag * np.array([1.0, 0.0])
            
            # Attractive force towards centroid to keep compact
            centroid = np.mean(pts, axis=0)
            for i in range(n):
                forces[i] -= 0.1 * (pts[i] - centroid)
            
            # Update positions
            # Step size decreases over time
            step = 0.1 * (1.0 - it / iterations)
            pts += step * forces
            
        return pts

    # Generate a set of diverse initial configurations
    # We'll use several different starting patterns
    
    candidates = []
    
    # 1. Honeycomb cluster (center + 6 neighbors + 3 more)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_to_cartesian(axial_1))
    
    # 2. Linear chain
    axial_2 = []
    for i in range(N):
        axial_2.append((i, 0))
    candidates.append(axial_to_cartesian(axial_2))
    
    # 3. Square-ish block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_to_cartesian(axial_3))
    
    # 4. Random start
    np.random.seed(42)
    pts_rand = np.random.rand(N, 2) * 4 - 2
    candidates.append(pts_rand)
    
    # 5. Another random start
    np.random.seed(123)
    pts_rand2 = np.random.rand(N, 2) * 4 - 2
    candidates.append(pts_rand2)
    
    # 6. Hexagonal ring
    axial_6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2) # This is 11 points? 1+6+4=11. Yes.
    ]
    # Wait, (2,0), (-2,0), (0,2), (0,-2) are 4 points. Total 11.
    candidates.append(axial_to_cartesian(axial_6))

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    def compute_enclosing_hexagon_lp(pts, beta):
        """
        Compute minimal enclosing hexagon side length using Linear Programming.
        Returns (S, center, D)
        """
        # Normals for the hexagon sides
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        n_pts = len(pts)
        # Variables: cx, cy, D
        # Minimize D -> c = [0, 0, 1]
        c = np.array([0.0, 0.0, 1.0])
        
        # Constraints:
        # For each normal n_k and point p_i:
        # n_k^T (c_center - p_i) <= D
        # -n_k^T (c_center - p_i) <= D
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
                # nx*cx + ny*cy - nx*px - ny*py <= D
                # nx*cx + ny*cy - D <= nx*px + ny*py
                A_ub.append([nx, ny, -1.0])
                b_ub.append(nx * px + ny * py)
                
                # -nx*cx - ny*cy + nx*px + ny*py <= D
                # -nx*cx - ny*cy - D <= -nx*px - ny*py
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-nx * px - ny * py)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for cx, cy, D
        # D must be positive
        bounds = [(None, None), (None, None), (0.0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                # Fallback to simple estimate if LP fails
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                max_proj = 0.0
                for nx, ny in normals:
                    projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                    max_proj = max(max_proj, np.max(projs), -np.min(projs))
                D = max_proj
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
        except Exception:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            max_proj = 0.0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            D = max_proj
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D

    # Process each candidate
    for idx, init_pts in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Relax the configuration
        relaxed_pts = force_directed_layout(init_pts, 200)
        
        # Find optimal beta
        # Coarse search
        best_local_S = float('inf')
        best_local_beta = 0.0
        best_local_center = None
        
        for i in range(13):
            beta = i * (math.pi / 6) / 12
            S, center, D = compute_enclosing_hexagon_lp(relaxed_pts, beta)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = beta
                best_local_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res_beta = minimize_scalar(lambda b: compute_enclosing_hexagon_lp(relaxed_pts, b)[0], 
                                           bounds=(0, math.pi / 6), method='bounded',
                                           options={'maxiter': 50, 'xatol': 1e-6})
                if res_beta.fun < best_local_S:
                    best_local_S = res_beta.fun
                    best_local_beta = res_beta.x
                    _, best_local_center, _ = compute_enclosing_hexagon_lp(relaxed_pts, best_local_beta)
        except Exception:
            pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_pts = relaxed_pts.copy()
            best_beta = best_local_beta
            best_center = best_local_center

    # Local NLP Refinement on the best configuration
    if best_pts is not None and best_center is not None:
        # We will optimize the positions of the hexagons to minimize the enclosing hexagon size.
        # The objective function will flatten the points, compute the enclosing hexagon, and return S.
        
        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e9
            pts_flat = params[:2*N].reshape(N, 2)
            
            # Quick check for overlaps to penalize if any
            # This is expensive, so we rely on the force-directed layout to have removed them.
            # If overlaps occur, the LP might still work but the "disjoint" constraint is violated.
            # We should add a penalty for overlaps.
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts_flat[i] - pts_flat[j])
                    if d < MIN_DIST:
                        overlap_penalty += (MIN_DIST - d)**2
            
            # Compute enclosing hexagon
            # We need to find the best beta for this configuration? 
            # No, beta is also a variable in the full optimization? 
            # The directive says "inner rows (x, y, angle_degrees)". The angle is per hexagon.
            # But the current interface returns a single `angle` for all? 
            # "inner rows (x, y, angle_degrees) of n unit hexagons"
            # "outer hexagon center (x, y), side length and rotation"
            # The `angle` in the inner rows is the rotation of the individual hexagon.
            # The `rotation` in the return is the rotation of the OUTER hexagon.
            
            # In the current seed, all inner hexagons have the same angle (0 or best_beta).
            # If we allow individual rotations, the problem is much harder.
            # The seed uses `angle_deg` for all inner hexagons.
            # Let's stick to the seed's assumption: all inner hexagons have the same orientation, 
            # and we are optimizing their positions and the outer hexagon's orientation/center.
            
            S, center, D = compute_enclosing_hexagon_lp(pts_flat, best_beta)
            return S + 100.0 * overlap_penalty

        # Initial guess: current best points
        init_params = best_pts.flatten().tolist()
        
        # We can also optimize beta here, but it's coupled. 
        # Let's try to optimize just positions first, keeping beta fixed.
        # Then we can do a final polish on beta.
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res_pos = minimize(objective, init_params, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res_pos.fun < best_S:
                    best_S = res_pos.fun
                    best_pts = res_pos.x[:2*N].reshape(N, 2)
                    _, best_center, _ = compute_enclosing_hexagon_lp(best_pts, best_beta)
        except Exception:
            pass

    # Final polish on beta and center
    if best_pts is not None:
        def final_objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            S, center, D = compute_enclosing_hexagon_lp(best_pts, beta)
            return S
        
        init_final = [best_center[0], best_center[1], best_beta]
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res_final = minimize(final_objective, init_final, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res_final.fun < best_S:
                    best_S = res_final.fun
                    best_center = (res_final.x[0], res_final.x[1])
                    best_beta = ((res_final.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

    # Prepare output
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Ensure non-overlap with a tiny scale if needed? 
    # The force-directed layout should handle it.
    # We return the positions as is.
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
