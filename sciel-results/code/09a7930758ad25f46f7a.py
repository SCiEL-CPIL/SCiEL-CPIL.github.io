import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (limit is 150s)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """
        Returns vertices of a regular hexagon.
        Side length 'side'. Center (cx, cy).
        Angle is rotation of the hexagon.
        Vertices are at angles 0, 60, ... 300 relative to the x-axis + angle.
        """
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normal_vectors(angle_deg):
        """
        Returns unit normal vectors for the 6 sides of a regular hexagon.
        For a hexagon with vertices at 0, 60, ..., the sides are perpendicular 
        to directions 30, 90, 150, 210, 270, 330.
        """
        angle_rad = np.radians(angle_deg)
        base_angles = np.radians(np.arange(30, 360, 60))
        angles = base_angles + angle_rad
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def outer_side_length(params, outer_angle_deg):
        """
        Calculate minimum side length of outer hexagon (centered at 0,0) 
        to contain all inner hexagons defined by params.
        params: 3N vector [x1, y1, a1, x2, y2, a2, ...]
        """
        # Get normals for the outer hexagon
        normals = get_normal_vectors(outer_angle_deg)
        
        # For each of the 6 directions, find the maximum projection of any vertex
        max_proj = 0.0
        
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            a = params[3*i+2]
            
            verts = hexagon_vertices(cx, cy, 1.0, a)
            
            # Project vertices onto normals
            # normals shape (6, 2), verts shape (6, 2)
            # We want max dot product for each normal direction
            # dot products shape (6, 6) -> normal_i dot vert_j
            dots = np.dot(normals, verts.T)
            
            # Max projection for each normal direction
            max_dots = np.max(dots, axis=1)
            
            # Update global max
            max_proj = np.maximum(max_proj, max_dots)
            
        # The apothem of the outer hexagon must be at least the max projection
        # Apothem = side * sqrt(3) / 2
        # side = apothem / (sqrt(3) / 2) = apothem * 2 / sqrt(3)
        return np.max(max_proj) * (2.0 / np.sqrt(3.0))

    def compute_penalty(params):
        """
        Compute smooth penalty for overlapping inner hexagons.
        Uses softplus approximation of max(0, overlap_depth).
        """
        penalty = 0.0
        # Check all pairs
        for i in range(N):
            for j in range(i + 1, N):
                vi = hexagon_vertices(params[3*i], params[3*i+1], 1.0, params[3*i+2])
                vj = hexagon_vertices(params[3*j], params[3*j+1], 1.0, params[3*j+2])
                
                # SAT check for overlap depth
                # For each edge normal of both polygons, project both polygons
                # If max_i < min_j - eps, they are separated.
                # Overlap depth is min_j - max_i (if positive, they overlap)
                
                axes = []
                for k in range(6):
                    e = vi[(k+1)%6] - vi[k]
                    n = np.array([-e[1], e[0]])
                    axes.append(n / np.linalg.norm(n))
                for k in range(6):
                    e = vj[(k+1)%6] - vj[k]
                    n = np.array([-e[1], e[0]])
                    axes.append(n / np.linalg.norm(n))
                
                min_overlap = float('inf')
                
                for axis in axes:
                    proj_i = np.dot(vi, axis)
                    proj_j = np.dot(vj, axis)
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    # If separated
                    if max_i < min_j:
                        min_overlap = 0.0
                        break
                    if max_j < min_i:
                        min_overlap = 0.0
                        break
                    
                    # Overlap depth in this direction
                    overlap = min(max_i, max_j) - max(min_i, min_j)
                    if overlap < min_overlap:
                        min_overlap = overlap
                        
                # min_overlap is the minimum distance to separate (if >0)
                # We want to penalize this.
                # Softplus: log(1 + exp(x)) approximates max(0, x)
                # We want penalty to be 0 if min_overlap < 0 (separated)
                # Actually, min_overlap > 0 means they overlap by that amount.
                # We want to push them apart.
                # Penalty = softplus(min_overlap)
                penalty += math.log(1 + math.exp(min_overlap))
                
        return penalty

    def objective_and_grad(params, outer_angle_deg):
        """
        Objective: Outer Side Length + Penalty
        Gradient: Analytic gradient of outer side + Gradient of penalty (approx via finite diff or analytic?)
        For speed, we'll use analytic gradient for outer side and finite diff for penalty if needed,
        but L-BFGS-B can use finite diff for the whole thing if we don't provide grad.
        However, providing grad for the smooth outer side part is fast.
        Let's try providing grad for the whole thing using finite differences for penalty is too slow.
        Better: Use analytic gradient for outer side, and rely on the fact that penalty is 0 in valid region?
        No, we need penalty during optimization.
        
        Strategy: 
        1. Compute outer side S.
        2. Compute penalty P.
        3. Total = S + lambda * P.
        
        Gradient of S:
        S = max_k ( max_j ( n_k . v_j ) ) * C
        Let M_k = max_j ( n_k . v_j ).
        S = C * max_k M_k.
        Gradient of S wrt params:
        dS/dp = C * sum_k (1 if M_k == max(M) else 0) * dM_k/dp
        
        dM_k/dp = sum_j (1 if v_j is argmax for M_k else 0) * n_k
        
        So we can compute this analytically.
        
        Gradient of P:
        P = sum_{i<j} softplus(overlap_{ij})
        dP/dp = sum_{i<j} sigmoid(overlap_{ij}) * doverlap_{ij}/dp
        
        doverlap/dp is tricky. It's the gradient of the minimum overlap depth.
        This is non-smooth at boundaries.
        
        Alternative: Use finite difference for the whole objective? 
        With 36 params, FD is 36 evaluations per step.
        If we do 500 steps, that's 18000 evaluations.
        Each evaluation is ~100us? 1.8s. Feasible.
        
        Let's just use finite difference for the whole objective to keep it simple and robust.
        """
        S = outer_side_length(params, outer_angle_deg)
        P = compute_penalty(params)
        total = S + 10.0 * P
        return total

    def objective(params):
        # We need to optimize outer angle too? 
        # The task allows rotating the outer hexagon.
        # The current seed program scans outer angle.
        # We can include outer angle in the parameter vector.
        # params: [x1, y1, a1, ..., xN, yN, aN, theta_out]
        # Total 3N + 1 = 37 params.
        
        theta_out = params[-1]
        inner_params = params[:-1]
        
        S = outer_side_length(inner_params, theta_out)
        P = compute_penalty(inner_params)
        return S + 10.0 * P

    # Initial guess
    # Hexagonal lattice
    init_inner = np.zeros(3 * N)
    
    # Center
    init_inner[0:3] = [0.0, 0.0, 0.0]
    
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        idx = (k + 1) * 3
        init_inner[idx:idx+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        
    # Ring 2 (5 hexagons) - Note: Hex lattice for 12 doesn't have a full ring 2.
    # 1 center, 6 ring 1, 5 ring 2?
    # Distances:
    # Center to Ring 1: 2.0
    # Ring 1 to Ring 2: 
    # In a hex lattice, distance between centers is 2.0.
    # Ring 2 hexagons are at distance 2*sqrt(3) ~ 3.46 from center?
    # Or are they filling the gaps?
    # Let's just place them roughly and let optimizer fix.
    for k in range(5):
        angle = np.radians(60 * k + 30)
        idx = (k + 7) * 3
        init_inner[idx:idx+3] = [3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0]
        
    # Outer angle initial 0
    init_params = np.concatenate([init_inner, [0.0]])
    
    # Bounds
    # Inner x, y in [-10, 10]
    # Inner angle in [0, 60]
    # Outer angle in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 60))   # angle
    bounds.append((0, 60)) # outer angle
    
    # Optimization
    best_params = init_params.copy()
    best_score = objective(best_params)
    
    # Multiple restarts if time permits
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
            
        # Perturb initial guess for restarts
        if restart > 0:
            params0 = best_params + np.random.uniform(-1, 1, size=37)
            # Ensure angles are valid
            for i in range(N):
                params0[3*i+2] = params0[3*i+2] % 60
            params0[-1] = params0[-1] % 60
        else:
            params0 = init_params.copy()
            
        res = minimize(
            objective, 
            params0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
    # Extract results
    inner_params = best_params[:-1]
    theta_out = best_params[-1]
    
    inner = []
    for i in range(N):
        inner.append((inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]))
        
    center = (0.0, 0.0)
    side = outer_side_length(inner_params, theta_out)
    angle = theta_out
    
    # Validation check
    valid = True
    normals = get_normal_vectors(angle)
    apothem = side * np.sqrt(3) / 2
    
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            # Check if point is inside outer hexagon
            for nx, ny in normals:
                dist = vx * nx + vy * ny
                if dist > apothem + 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Expand slightly to ensure validity
        side *= 1.001
        
    return inner, center, side, angle

# EVOLVE-BLOCK-END
