import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Vertices of a unit regular hexagon (side length 1)
    # Centered at origin, flat-top (vertex at 0 deg? No, flat top means side is horizontal at top/bottom?
    # Standard: vertices at 0, 60, 120...
    # If angle 0 is a vertex, it's pointy-top.
    # If angle 30 is a vertex, it's flat-top?
    # Let's define vertices for side length 1.
    # Distance from center to vertex (circumradius) is 1.
    # Distance from center to side (inradius) is sqrt(3)/2.
    
    def get_hex_vertices(cx, cy, angle_rad):
        # Returns 6x2 array
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack([x, y])

    def sat_penetration(v1, v2):
        """
        Compute penetration depth between two convex polygons v1, v2 (6x2).
        Returns 0 if disjoint or separated, otherwise the minimum translation distance to separate.
        """
        # Get edge normals for v1
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack([e1[:, 1], -e1[:, 0]])
        n1 /= np.linalg.norm(n1, axis=1)[:, np.newaxis]
        
        # Get edge normals for v2
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack([e2[:, 1], -e2[:, 0]])
        n2 /= np.linalg.norm(n2, axis=1)[:, np.newaxis]
        
        normals = np.vstack([n1, n2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        if np.min(overlaps) <= 1e-9:
            return 0.0
            
        return np.min(overlaps)

    def calculate_required_side(positions, inner_angle, outer_angle):
        """
        Calculate the minimum side length S of an outer hexagon (centered at 0,0)
        that contains all inner hexagons at given positions.
        """
        # Outer hexagon normals (for side length 1, inradius is sqrt(3)/2)
        # The support function of a hexagon with side S in direction n is S * sqrt(3)/2
        # So S = (2/sqrt(3)) * max_projection
        
        outer_normals = []
        # Outer hexagon vertices for side 1, angle outer_angle
        # But we just need the normals.
        # Normals are perpendicular to edges.
        # For a regular hexagon, normals are at angles outer_angle + 30 + k*60?
        # If vertices are at outer_angle + k*60, edges are between them.
        # Edge 0: from v0 to v1.
        # Let's just compute vertices and then normals.
        
        outer_verts_unit = get_hex_vertices(0, 0, outer_angle)
        e_o = np.roll(outer_verts_unit, -1, axis=0) - outer_verts_unit
        n_o = np.column_stack([e_o[:, 1], -e_o[:, 0]])
        n_o /= np.linalg.norm(n_o, axis=1)[:, np.newaxis]
        
        max_proj = 0.0
        for i in range(N):
            x, y = positions[i]
            v = get_hex_vertices(x, y, inner_angle)
            p = v @ n_o.T
            max_proj_i = np.max(p)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
                
        # max_proj is the maximum projection onto the unit normals of the unit-hexagon frame.
        # For a hexagon with side S, the projection onto the normal is S * sqrt(3)/2.
        # So S = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)

    def objective(params, inner_angle, outer_angle):
        if time.time() > deadline:
            return 1e10
            
        positions = params.reshape(N, 2)
        
        # Calculate required side length
        s_req = calculate_required_side(positions, inner_angle, outer_angle)
        
        # Calculate overlap penalty
        penalty = 0.0
        verts_list = []
        for i in range(N):
            x, y = positions[i]
            verts_list.append(get_hex_vertices(x, y, inner_angle))
            
        for i in range(N):
            for j in range(i + 1, N):
                pd = sat_penetration(verts_list[i], verts_list[j])
                if pd > 0:
                    penalty += pd * 10.0  # Weight for overlap
        
        return s_req + penalty

    # Initial guess: Honeycomb
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    init_pos = []
    init_pos.append([0.0, 0.0])
    for k in range(6):
        a = np.radians(60 * k)
        init_pos.append([d1 * np.cos(a), d1 * np.sin(a)])
    # 5 outer ones
    # Let's place them at the "corners" of the second ring
    for k in range(5):
        a = np.radians(60 * k + 30) # Offset by 30 to fill gaps? Or 0?
        # In a honeycomb, the next ring touches the first.
        # The distance from center to the center of the "gap" filling hexagons is larger.
        # Let's just use d2 and angles 0, 60, 120, 180, 240
        a = np.radians(60 * k)
        init_pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        
    init_pos = np.array(init_pos)
    x0 = init_pos.flatten()

    best_s = np.inf
    best_params = None
    best_inner_angle = 0.0
    best_outer_angle = 0.0

    # Try different alignments
    # Inner angle 0 (pointy top) or 30 (flat top)
    # Outer angle 0 or 30
    # Due to symmetry, we can fix outer to 0 and vary inner, or vice versa.
    # Let's try a few combinations.
    
    configs = [
        (0.0, 0.0),
        (np.radians(30), 0.0),
        (0.0, np.radians(30)),
        (np.radians(30), np.radians(30))
    ]

    for i_angle, o_angle in configs:
        if time.time() > deadline - 30:
            break
            
        try:
            # Global search with Differential Evolution
            # Bounds: positions roughly within [-5, 5]
            bounds = [(-5, 5) for _ in range(2 * N)]
            
            # Use a smaller population for speed, or rely on the good initial guess
            # DE is stochastic, might be slow.
            # Let's use L-BFGS-B from the initial guess first, then DE if time permits.
            
            res = minimize(
                objective,
                x0,
                args=(i_angle, o_angle),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            s_val = calculate_required_side(res.x.reshape(N, 2), i_angle, o_angle)
            # Check penalty
            penalty_check = objective(res.x, i_angle, o_angle) - s_val
            if penalty_check < 1e-6: # Valid
                if s_val < best_s:
                    best_s = s_val
                    best_params = res.x.copy()
                    best_inner_angle = i_angle
                    best_outer_angle = o_angle
                    
        except Exception:
            pass

    # If time remains, try Differential Evolution from scratch or perturbations
    if best_params is not None and time.time() < deadline - 20:
        # Try DE for a more global search
        # This might be too slow for 24 variables.
        # Instead, try multiple random restarts of L-BFGS-B
        
        for _ in range(10):
            if time.time() > deadline - 10:
                break
                
            # Random perturbation of initial guess
            noise = np.random.normal(0, 0.5, x0.shape)
            x_pert = x0 + noise
            
            for i_angle, o_angle in configs:
                try:
                    res = minimize(
                        objective,
                        x_pert,
                        args=(i_angle, o_angle),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    
                    s_val = calculate_required_side(res.x.reshape(N, 2), i_angle, o_angle)
                    penalty_check = objective(res.x, i_angle, o_angle) - s_val
                    if penalty_check < 1e-6:
                        if s_val < best_s:
                            best_s = s_val
                            best_params = res.x.copy()
                            best_inner_angle = i_angle
                            best_outer_angle = o_angle
                except Exception:
                    pass

    # Fallback
    if best_params is None:
        best_params = x0
        best_s = calculate_required_side(x0.reshape(N, 2), 0, 0)
        best_inner_angle = 0
        best_outer_angle = 0

    final_pos = best_params.reshape(N, 2)
    
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(best_inner_angle)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = best_s
    outer_angle_deg = np.degrees(best_outer_angle)
    
    return inner, outer_center, outer_side, outer_angle_deg
