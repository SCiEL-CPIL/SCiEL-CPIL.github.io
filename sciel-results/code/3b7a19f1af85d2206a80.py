"""
Strategy: Centroid-Projected Minkowski Sum with Global Orientation Search.
Arrange 12 unit hexagons in a fixed honeycomb cluster (1 center + 6 ring + 5 ring).
The outer hexagon side length for a given orientation is computed via the Minkowski sum
support function. We minimize over the 6-fold symmetric cluster parameters
(r1, r2, a1, a2, outer_angle) using scipy.optimize with a smooth soft-max surrogate.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Precompute unit hexagon vertices at angle 0
    unit_hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    def get_inner_positions(r1, a1, r2, a2):
        """Generate 12 inner hexagon positions: (x, y, angle_deg)."""
        positions = []
        positions.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = math.radians(60 * k)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            positions.append((x, y, a1))
        for k in range(5):
            angle = math.radians(60 * k + 30)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            positions.append((x, y, a2))
        return positions

    def compute_min_side_fast(r1, a1, r2, a2, outer_angle_deg):
        """Compute minimal outer hexagon side length using vectorized support function."""
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        # Get all inner positions
        positions = get_inner_positions(r1, a1, r2, a2)
        
        # For each inner hexagon and each normal, compute the projection
        # support of unit hex in direction n_rotated = max over vertices of dot(v, n_rot)
        # n_rot = R(-alpha) * n
        
        max_required = 0.0
        for (cx, cy, alpha_deg) in positions:
            alpha_rad = math.radians(alpha_deg)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            # R(-alpha) = [[cos_a, sin_a], [-sin_a, cos_a]]
            # local_normals = normals @ R(-alpha)^T = normals @ [[cos_a, -sin_a],[sin_a, cos_a]]
            local_normals = np.column_stack([
                normals[:, 0] * cos_a + normals[:, 1] * sin_a,
                -normals[:, 0] * sin_a + normals[:, 1] * cos_a
            ])
            # Support: max over unit hex vertices of dot(v, local_n) for each local_n
            # dots shape: (6, 6) - 6 normals x 6 vertices
            dots = local_normals @ unit_hex_verts.T  # (6, 6)
            supports = np.max(dots, axis=1)  # (6,)
            # Projection of center onto each normal
            center_proj = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
            vals = center_proj + supports  # (6,)
            max_required = max(max_required, np.max(vals))
        
        apothem = max_required
        side = apothem / (math.sqrt(3) / 2)
        return side

    def soft_max(values, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        scaled = beta * values
        max_val = np.max(scaled)
        return max_val / beta + math.log(np.sum(np.exp(scaled - max_val))) / beta

    def objective_smooth(params, beta=100.0):
        """Smooth objective using soft-max over all 6 normals."""
        if time.time() > deadline - 3:
            return 1e6
        r1, a1, r2, a2, o_a = params
        if r1 < 0.1 or r2 < 0.1:
            return 1e6
        
        outer_angle_rad = math.radians(o_a)
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        positions = get_inner_positions(r1, a1, r2, a2)
        
        all_vals = []
        for (cx, cy, alpha_deg) in positions:
            alpha_rad = math.radians(alpha_deg)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            local_normals = np.column_stack([
                normals[:, 0] * cos_a + normals[:, 1] * sin_a,
                -normals[:, 0] * sin_a + normals[:, 1] * cos_a
            ])
            dots = local_normals @ unit_hex_verts.T  # (6, 6)
            supports = np.max(dots, axis=1)  # (6,)
            center_proj = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
            vals = center_proj + supports  # (6,)
            all_vals.extend(vals.tolist())
        
        all_vals = np.array(all_vals)
        # Use soft-max to get a smooth approximation of the max
        smooth_max = soft_max(all_vals, beta=beta)
        side = smooth_max / (math.sqrt(3) / 2)
        return side

    def check_disjoint_fast(r1, a1, r2, a2):
        """Check if inner hexagons are pairwise disjoint using SAT (fast version)."""
        positions = get_inner_positions(r1, a1, r2, a2)
        n_pos = len(positions)
        
        # Precompute vertices
        all_verts = []
        for (xi, yi, ai) in positions:
            ai_rad = math.radians(ai)
            v = np.array([
                [xi + math.cos(ai_rad + math.radians(60*k)), yi + math.sin(ai_rad + math.radians(60*k))]
                for k in range(6)
            ])
            all_verts.append(v)
        
        # Precompute edge normals for each hexagon
        all_normals = []
        for v in all_verts:
            normals = []
            for k in range(6):
                p1 = v[k]
                p2 = v[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normals.append(normal / norm)
            all_normals.append(normals)
        
        for i in range(n_pos):
            for j in range(i + 1, n_pos):
                vi = all_verts[i]
                vj = all_verts[j]
                # Use normals from both polygons
                axes = all_normals[i] + all_normals[j]
                
                separated = False
                for axis in axes:
                    proj_i = vi @ axis
                    proj_j = vj @ axis
                    if proj_i.max() < proj_j.min() - 1e-9 or proj_j.max() < proj_i.min() - 1e-9:
                        separated = True
                        break
                
                if not separated:
                    return False
        return True

    def objective_with_constraint(params):
        """Objective with disjointness constraint as penalty."""
        if time.time() > deadline - 3:
            return 1e6
        r1, a1, r2, a2, o_a = params
        if r1 < 0.1 or r2 < 0.1:
            return 1e6
        if not check_disjoint_fast(r1, a1, r2, a2):
            # Penalty proportional to how much they overlap (use a large constant)
            return 50.0 + compute_min_side_fast(r1, a1, r2, a2, o_a)
        return compute_min_side_fast(r1, a1, r2, a2, o_a)

    # Multi-start global search with scipy.optimize
    best_side = float('inf')
    best_params = None
    
    # Initial guesses based on honeycomb geometry
    initial_guesses = [
        [2.0, 0.0, 2*math.sqrt(3), 0.0, 30.0],
        [2.0, 0.0, 2*math.sqrt(3), 30.0, 0.0],
        [2.0, 30.0, 2*math.sqrt(3), 0.0, 0.0],
        [2.0, 0.0, 2*math.sqrt(3), 0.0, 0.0],
        [2.0, 0.0, 2*math.sqrt(3), 30.0, 30.0],
        [1.95, 0.0, 3.45, 0.0, 30.0],
        [2.05, 0.0, 3.5, 0.0, 30.0],
    ]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(15):
        r1 = np.random.uniform(1.85, 2.15)
        a1 = np.random.uniform(0, 60)
        r2 = np.random.uniform(3.3, 3.65)
        a2 = np.random.uniform(0, 60)
        o_a = np.random.uniform(0, 60)
        initial_guesses.append([r1, a1, r2, a2, o_a])
    
    bounds = [(1.5, 2.5), (0, 60), (2.5, 4.5), (0, 60), (0, 60)]
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
        
        # First try smooth objective with BFGS
        try:
            result = minimize(
                objective_smooth,
                x0=np.array(guess, dtype=float),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            if result.success or result.fun < 50:
                params = result.x
                # Verify with exact objective
                exact_val = objective_with_constraint(params)
                if exact_val < best_side and exact_val < 50:
                    best_side = exact_val
                    best_params = params.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
        
        # Then try with constraint penalty for more precise result
        try:
            result2 = minimize(
                objective_with_constraint,
                x0=np.array(guess, dtype=float),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result2.fun < 50:
                params = result2.x
                exact_val = compute_min_side_fast(params[0], params[1], params[2], params[3], params[4])
                if check_disjoint_fast(params[0], params[1], params[2], params[3]):
                    if exact_val < best_side:
                        best_side = exact_val
                        best_params = params.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a known valid configuration
    if best_params is None:
        best_params = np.array([2.0, 0.0, 2*math.sqrt(3), 0.0, 30.0])
        best_side = compute_min_side_fast(*best_params)
    
    r1, a1, r2, a2, o_a = best_params
    a1 = a1 % 360
    a2 = a2 % 360
    o_a = o_a % 360
    
    # Construct final inner positions
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = math.radians(60 * k)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        inner.append((x, y, a1))
    for k in range(5):
        angle = math.radians(60 * k + 30)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        inner.append((x, y, a2))
    
    oc = (0.0, 0.0)
    oa = o_a
    os = compute_min_side_fast(r1, a1, r2, a2, o_a)
    
    return inner, oc, os, oa
