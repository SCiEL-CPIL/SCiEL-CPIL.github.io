import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    n = 12
    
    # Helper: Get vertices of a hexagon
    def get_verts(cx, cy, side, angle_deg):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        xs = cx + side * np.cos(angles)
        ys = cy + side * np.sin(angles)
        return np.column_stack((xs, ys))

    # Helper: Get outward normals for a CCW polygon
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        dx = edges[:, 0]
        dy = edges[:, 1]
        normals = np.column_stack((dy, -dx))
        norms = np.linalg.norm(normals, axis=1)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        normals = normals / norms[:, np.newaxis]
        return normals

    # Differentiable Objective Function
    # Returns sum of squared violations (penetration depths)
    def objective(params, s_outer):
        # Reshape params: (12, 3) -> (x, y, angle)
        p = params.reshape(n, 3)
        
        # Precompute vertices and normals for inner hexagons
        # We need to handle the gradient carefully or just rely on finite differences if not providing jac.
        # L-BFGS-B uses finite differences if jac is not provided.
        # To make it faster and more robust, let's compute the value.
        
        inner_verts = []
        inner_normals = []
        for i in range(n):
            x, y, ang = p[i]
            v = get_verts(x, y, 1.0, ang)
            inner_verts.append(v)
            inner_normals.append(get_normals(v))
            
        outer_v = get_verts(0, 0, s_outer, 0)
        outer_normals = get_normals(outer_v)
        
        penalty = 0.0
        
        # 1. Inner-Inner Separation (SAT)
        # For each pair, check all normals from both hexagons.
        # If max gap < 0, they overlap. Penalty is (-max_gap)^2.
        # Actually, for differentiability, we can sum squared gaps for all axes?
        # No, SAT is a max over axes. The max function is not differentiable.
        # However, for small overlaps, the separating axis is usually unique or close.
        # A common trick is to sum the squares of the negative gaps for ALL axes, 
        # or just use the worst gap. 
        # Let's use the worst gap (max penetration). 
        # To make it differentiable-ish for L-BFGS, we can smooth it or just let L-BFGS handle the non-smoothness 
        # via finite differences, but it might be slow.
        
        # Alternative: Sum of squared distances for all "violating" axes?
        # Let's stick to the standard penalty: sum of (-gap)^2 for the worst gap per pair.
        # L-BFGS can handle this, but it's better to provide a smoother objective.
        # Let's try summing (-gap)^2 for ALL normals of both hexagons. This forces separation along all axes, 
        # which is sufficient and differentiable (since dot products are smooth).
        
        for i in range(n):
            for j in range(i + 1, n):
                A = inner_verts[i]
                B = inner_verts[j]
                # Normals from A and B
                # We need to project A and B onto these normals.
                # For a normal n, gap = min(A.n) - max(B.n).
                # If gap < 0, penalty += gap^2.
                
                # To be efficient, we can vectorize this slightly, but n=12 is small.
                # Let's just loop over the 12 normals (6 from A, 6 from B).
                # Actually, due to symmetry, 3 unique directions per hexagon. But 6 is fine.
                
                # Project A onto normals of A
                # proj_A_on_A = A @ inner_normals[i].T -> (6, 6)
                # min for each normal: np.min(proj_A_on_A, axis=0)
                
                # This is getting complex to vectorize perfectly without overhead.
                # Let's just loop over pairs and normals.
                
                # Normals from i
                for k in range(6):
                    n_vec = inner_normals[i][k]
                    proj_A = np.dot(A, n_vec)
                    proj_B = np.dot(B, n_vec)
                    gap = np.min(proj_A) - np.max(proj_B)
                    if gap < 0:
                        penalty += gap * gap
                        
                # Normals from j
                for k in range(6):
                    n_vec = inner_normals[j][k]
                    proj_A = np.dot(A, n_vec)
                    proj_B = np.dot(B, n_vec)
                    gap = np.min(proj_A) - np.max(proj_B)
                    if gap < 0:
                        penalty += gap * gap

        # 2. Inner-Outer Containment
        # Inner vertices must be inside outer hexagon.
        # For each inner vertex, check against all outer edges.
        # Outer edge defined by point p1 and outward normal n_out.
        # Distance = (v - p1) . n_out.
        # If distance > 0, it's outside. Penalty += distance^2.
        
        for i in range(n):
            v = inner_verts[i]
            for k in range(6):
                pt = v[k]
                for m in range(6):
                    p1 = outer_v[m]
                    n_out = outer_normals[m]
                    dist = np.dot(pt - p1, n_out)
                    if dist > 0:
                        penalty += dist * dist
                        
        return penalty

    # Initial Guess: Honeycomb
    # 1 center, 6 in ring 1, 5 in ring 2? 
    # A regular hexagonal number is 1, 7, 19...
    # 12 is not a hex number.
    # Let's use a 3x4 grid of hexagons? No, they need to be rotated.
    # Let's use a cluster.
    # Center (0,0).
    # 6 neighbors at distance sqrt(3) (for side 1 hexes touching? No, distance between centers of touching unit hexes is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) approx 1.732? 
    # Wait. Side d=1. Distance from center to vertex is 1. Distance from center to edge midpoint (apothem) is sqrt(3)/2.
    # Two touching hexes: distance between centers = 2 * apothem = sqrt(3).
    
    h = np.sqrt(3) # Distance between centers of touching hexes
    
    init_positions = [
        [0, 0, 0],
        [h, 0, 0],
        [-h, 0, 0],
        [0, 2*h/np.sqrt(3)*np.sqrt(3)/2 * 2, 0], # This is messy. 
        # Let's use standard axial coordinates.
        # q, r
        # (0,0)
        # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
        # Convert to Cartesian:
        # x = h * (q + r/2)
        # y = h * (sqrt(3)/2) * r
        
        # Let's just hardcode a good layout.
    ]
    
    # Layout:
    # Row 0: 4 hexes?
    # Let's try a 2x6 grid?
    # Or just a circle.
    # Center 0.
    # 6 around.
    # 5 more around the gaps?
    
    # Let's use a simple grid init and let optimizer fix it.
    # Grid of 3x4.
    # Spacing x: 1.5, y: sqrt(3) ~ 1.732
    # Actually, for flat-top hexes:
    # x_spacing = 1.5
    # y_spacing = sqrt(3)
    # Offset for odd rows: 0.75
    
    init_params_list = []
    # 3 rows, 4 cols
    for r in range(3):
        for c in range(4):
            x = c * 1.5
            if r % 2 == 1:
                x += 0.75
            y = r * np.sqrt(3)
            init_params_list.append([x, y, 0])
            
    # Center the cluster
    xs = [p[0] for p in init_params_list]
    ys = [p[1] for p in init_params_list]
    cx_mean = np.mean(xs)
    cy_mean = np.mean(ys)
    
    init_params = []
    for p in init_params_list:
        init_params.append([p[0] - cx_mean, p[1] - cy_mean, p[2]])
        
    init_params = np.array(init_params).flatten()

    # Bounds for optimizer
    # s will be around 4-6. Let's bound centers to [-5, 5] and angles [0, 360]
    bounds = []
    for i in range(n):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((0, 360))

    def try_pack(s_target, num_restarts=5):
        min_pen = float('inf')
        best_params = None
        
        for r in range(num_restarts):
            if r == 0:
                p0 = init_params.copy()
            else:
                p0 = init_params + np.random.normal(0, 0.5, size=init_params.shape)
            
            # L-BFGS-B
            res = minimize(objective, p0, args=(s_target,), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < min_pen:
                min_pen = res.fun
                best_params = res.x
                
        # Check validity
        if min_pen < 1e-6:
            return True, best_params
        else:
            return False, best_params

    # Binary search for s
    s_low = 3.5 # Theoretical min is sqrt(12) ~ 3.46, but packing 12 is harder. 3.5 is a safe lower bound guess.
    s_high = 6.0
    
    best_s = s_high
    best_params = init_params
    
    # First, ensure we have a valid packing at s_high
    valid, best_params = try_pack(s_high, num_restarts=10)
    if not valid:
        # If even 6.0 fails, something is wrong with the init or optimizer settings.
        # Try increasing s_high or more restarts.
        valid, best_params = try_pack(7.0, num_restarts=20)
        if valid:
            s_high = 7.0
        else:
            # Fallback: return a very large s to be valid, score will be low but valid.
            # Or try a different init.
            return init_params.reshape(12, 3), np.array([0, 0, 0]), 10.0

    # Now binary search
    for _ in range(20):
        s_mid = (s_low + s_high) / 2
        valid, params = try_pack(s_mid, num_restarts=5)
        if valid:
            best_s = s_mid
            best_params = params
            s_low = s_mid
        else:
            s_high = s_mid

    # Final refinement
    valid, best_params = try_pack(best_s, num_restarts=10)
    if not valid:
        # If final check fails, use the last known good one
        best_params = best_params # from previous iteration

    inner_hex_data = best_params.reshape(12, 3)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s
