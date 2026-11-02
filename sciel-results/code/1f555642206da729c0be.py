import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
DEADLINE = time.time() + 140.0

# Precompute base vertices for a unit hexagon (flat-top, vertex at angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 330
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_VERTS = np.column_stack([np.cos(BASE_ANGLES), np.sin(BASE_ANGLES)])

def get_hex_verts(positions, angles):
    """
    positions: (N, 2)
    angles: (N,) in radians
    Returns: (N, 6, 2)
    """
    # angles shape (N, 1, 1) for broadcasting
    theta = angles[:, np.newaxis, np.newaxis]
    c = np.cos(theta)
    s = np.sin(theta)
    # R = [[c, -s], [s, c]]
    # verts = R @ BASE_VERTS.T + pos
    # Let's do it manually for speed/clarity or use einsum
    # v_i = c * B_i - s * B_{i+1}? No, standard rotation.
    # x' = c*x - s*y
    # y' = s*x + c*y
    bx = BASE_VERTS[:, 0]
    by = BASE_VERTS[:, 1]
    
    # (N, 1) * (1, 6)
    rx = c * bx[None, :] - s * by[None, :]
    ry = s * bx[None, :] + c * by[None, :]
    
    # Add positions
    rx += positions[:, 0:1]
    ry += positions[:, 1:2]
    
    return np.stack([rx, ry], axis=-1)

def smooth_min(a, b, k=10.0):
    """
    Approximates min(a, b) smoothly.
    k controls the sharpness. Higher k -> sharper.
    """
    return np.minimum(a, b) - np.logaddexp(-k * a, -k * b) / k

def compute_objective(params, k_overlap=20.0, k_contain=20.0):
    """
    params: (3N, 1) -> [x1, y1, theta1, x2, y2, theta2, ...]
    Returns: (side_length, penalty)
    """
    pos = params[0::3]
    ang = params[1::3]
    # Note: params are [x, y, theta] interleaved? 
    # Let's stick to [x1, y1, theta1, x2, y2, theta2...]
    # pos = params[0::3] -> x1, x2...
    # pos_y = params[1::3] -> y1, y2...
    # ang = params[2::3] -> theta1, theta2...
    pos_y = params[1::3]
    ang = params[2::3]
    
    positions = np.column_stack([pos, pos_y])
    
    verts = get_hex_verts(positions, ang)
    all_verts = verts.reshape(-1, 2)
    
    # 1. Containment Penalty
    # We want all points inside a regular hexagon of side S centered at origin.
    # Inequalities: L_i(v) <= S * apothem = S * sqrt(3)/2
    # L_i are linear forms with unit normals.
    # Normals for flat-top hexagon (vertex at 0):
    # The sides are perpendicular to normals at 30, 90, 150, 210, 270, 330.
    # Normal 30 deg: (sqrt3/2, 1/2)
    # Normal 90 deg: (0, 1)
    # ...
    
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    L1 = x * (SQRT3/2) + y * 0.5
    L2 = y
    L3 = -x * (SQRT3/2) + y * 0.5
    L4 = -x * (SQRT3/2) - y * 0.5
    L5 = -y
    L6 = x * (SQRT3/2) - y * 0.5
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    max_L = np.max(all_L, axis=0) # (N*6,)
    
    # We want max_L <= S * SQRT3/2
    # Equivalent to S >= max_L / (SQRT3/2)
    # Let's define the objective as minimizing S, subject to S >= max_L / (SQRT3/2).
    # In penalty method: we can just set S = max_L / (SQRT3/2) and add penalty if it's too small?
    # Actually, the task asks to minimize the outer side length.
    # If we just return S = max_L / (SQRT3/2), the optimizer will try to make S small.
    # But S is determined by the positions. So we minimize max_L.
    # Wait, the "side length" returned is the metric.
    # If we minimize max_L, we minimize the required side length.
    # So objective 1 = max_L * 2 / SQRT3.
    
    side_len = np.max(max_L) * 2.0 / SQRT3
    
    # 2. Overlap Penalty
    # For each pair of hexagons, we need them to be disjoint.
    # SAT condition: There exists a separating axis.
    # For two convex polygons, they are disjoint if there is an axis (normal to an edge of either)
    # such that their projections do not overlap.
    # Projection overlap on axis n: 
    # max(proj1) < min(proj2) OR max(proj2) < min(proj1)
    # Let d = max(proj1) - min(proj2). If d < 0, separated.
    # We want d < 0 for some axis.
    # Penalty = sum over pairs of [ max(0, min_axis_overlap) ]
    # Where min_axis_overlap = min( max(proj1)-min(proj2), max(proj2)-min(proj1) )
    # If min_axis_overlap < 0, they are separated.
    # If > 0, they overlap by that amount.
    
    penalty = 0.0
    # Precompute edge normals for each hexagon?
    # Or just compute on the fly.
    # There are 12 hexagons. 66 pairs.
    # For each pair, 12 axes (6 from each).
    
    # To speed up, we can vectorize part of it, but 66 pairs is small.
    # However, inside the objective function called hundreds of times, it must be fast.
    
    # Let's optimize the overlap check.
    # For pair (i, j):
    # Axes: normals of edges of i and j.
    # Since hexagons are regular, the normals are fixed relative to their rotation.
    
    for i in range(N):
        for j in range(i + 1, N):
            vi = verts[i]
            vj = verts[j]
            
            # Get normals for vi
            # Edge vectors
            edges_i = np.roll(vi, -1, axis=0) - vi
            normals_i = np.column_stack([-edges_i[:, 1], edges_i[:, 0]])
            normals_i /= (np.linalg.norm(normals_i, axis=1, keepdims=True) + 1e-9)
            
            edges_j = np.roll(vj, -1, axis=0) - vj
            normals_j = np.column_stack([-edges_j[:, 1], edges_j[:, 0]])
            normals_j /= (np.linalg.norm(normals_j, axis=1, keepdims=True) + 1e-9)
            
            all_normals = np.vstack([normals_i, normals_j])
            
            # Project vi and vj onto all_normals
            # proj_i = vi @ all_normals.T
            proj_i = vi @ all_normals.T
            proj_j = vj @ all_normals.T
            
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            
            # Overlap on each axis
            # overlap_k = min(max_i - min_j, max_j - min_i)
            # If overlap_k < 0, separated on this axis.
            # We want the MINIMUM overlap over all axes to be < 0.
            # Let m = min_k( overlap_k )
            # If m < 0, valid.
            # Penalty should be max(0, m).
            
            overlap1 = max_i - min_j
            overlap2 = max_j - min_i
            overlaps = np.minimum(overlap1, overlap2)
            
            m = np.min(overlaps)
            
            if m > 0:
                # Use smooth max to allow gradients
                # penalty += smooth_max(m, 0, k)
                penalty += m # Linear penalty for simplicity and speed? 
                # Actually, L-BFGS-B handles non-differentiable points poorly if we use hard max.
                # Let's use a soft plus: x_+ = (x + sqrt(x^2 + eps))/2
                # Or just smooth_max.
                # penalty += smooth_max(m, 0.0, k_overlap)
                # Let's stick to simple linear for now, it's often robust enough if k is high.
                # Wait, if m is the min overlap, and it's positive, we want to reduce it.
                # The gradient of min is tricky.
                # Alternative: Sum of squares of overlaps? No, that penalizes separation too.
                # We only care if they overlap.
                # Let's use: penalty += m if m > 0 else 0.
                # To make it differentiable, use: 0.5 * (m + sqrt(m^2 + 1e-6))
                penalty += 0.5 * (m + np.sqrt(m*m + 1e-6))
    
    return side_len + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_params = None
    
    # Initial guesses
    # 1. Honeycomb
    # 2. Random
    
    def generate_honeycomb():
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        pos = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pos.append([x, y])
        pos = np.array(pos)
        # Center it
        pos -= np.mean(pos, axis=0)
        # Params: x, y, theta
        params = np.zeros(3 * N)
        params[0::3] = pos[:, 0]
        params[1::3] = pos[:, 1]
        params[2::3] = 0.0
        return params

    def generate_random():
        # Random positions in a box, random angles
        pos = np.random.uniform(-2, 2, (N, 2))
        ang = np.random.uniform(0, 2 * np.pi, N)
        params = np.zeros(3 * N)
        params[0::3] = pos[:, 0]
        params[1::3] = pos[:, 1]
        params[2::3] = ang
        return params

    initializations = [generate_honeycomb()]
    # Add a few random ones
    for _ in range(5):
        initializations.append(generate_random())
    
    for init_params in initializations:
        if time.time() > deadline - 20:
            break
            
        # L-BFGS-B
        # We need to handle the case where the initial guess is very bad.
        # The penalty might be huge.
        
        def obj_func(p):
            return compute_objective(p)
        
        try:
            res = minimize(obj_func, init_params, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_side:
                # Verify validity with strict SAT
                pos = res.x[0::3]
                pos_y = res.x[1::3]
                ang = res.x[2::3]
                positions = np.column_stack([pos, pos_y])
                
                # Check validity
                valid = True
                verts = get_hex_verts(positions, ang)
                for i in range(N):
                    for j in range(i + 1, N):
                        vi = verts[i]
                        vj = verts[j]
                        
                        edges_i = np.roll(vi, -1, axis=0) - vi
                        normals_i = np.column_stack([-edges_i[:, 1], edges_i[:, 0]])
                        normals_i /= (np.linalg.norm(normals_i, axis=1, keepdims=True) + 1e-9)
                        
                        edges_j = np.roll(vj, -1, axis=0) - vj
                        normals_j = np.column_stack([-edges_j[:, 1], edges_j[:, 0]])
                        normals_j /= (np.linalg.norm(normals_j, axis=1, keepdims=True) + 1e-9)
                        
                        all_normals = np.vstack([normals_i, normals_j])
                        proj_i = vi @ all_normals.T
                        proj_j = vj @ all_normals.T
                        
                        max_i = np.max(proj_i, axis=0)
                        min_i = np.min(proj_i, axis=0)
                        max_j = np.max(proj_j, axis=0)
                        min_j = np.min(proj_j, axis=0)
                        
                        overlap1 = max_i - min_j
                        overlap2 = max_j - min_i
                        overlaps = np.minimum(overlap1, overlap2)
                        
                        if np.min(overlaps) > 1e-6: # Overlap
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Calculate exact side
                    all_verts = verts.reshape(-1, 2)
                    x = all_verts[:, 0]
                    y = all_verts[:, 1]
                    L1 = x * (SQRT3/2) + y * 0.5
                    L2 = y
                    L3 = -x * (SQRT3/2) + y * 0.5
                    L4 = -x * (SQRT3/2) - y * 0.5
                    L5 = -y
                    L6 = x * (SQRT3/2) - y * 0.5
                    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
                    side = np.max(all_L) * 2.0 / SQRT3
                    
                    if side < best_side:
                        best_side = side
                        best_params = res.x
                        
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_params is None:
        # Use the honeycomb one, scaled up if necessary
        init = generate_honeycomb()
        # Just return the honeycomb, it's likely valid and close to optimal
        # Or scale it to ensure validity
        pos = init[0::3]
        pos_y = init[1::3]
        ang = init[2::3]
        positions = np.column_stack([pos, pos_y])
        
        # Check validity
        verts = get_hex_verts(positions, ang)
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                vi = verts[i]
                vj = verts[j]
                edges_i = np.roll(vi, -1, axis=0) - vi
                normals_i = np.column_stack([-edges_i[:, 1], edges_i[:, 0]])
                normals_i /= (np.linalg.norm(normals_i, axis=1, keepdims=True) + 1e-9)
                edges_j = np.roll(vj, -1, axis=0) - vj
                normals_j = np.column_stack([-edges_j[:, 1], edges_j[:, 0]])
                normals_j /= (np.linalg.norm(normals_j, axis=1, keepdims=True) + 1e-9)
                all_normals = np.vstack([normals_i, normals_j])
                proj_i = vi @ all_normals.T
                proj_j = vj @ all_normals.T
                max_i = np.max(proj_i, axis=0)
                min_i = np.min(proj_i, axis=0)
                max_j = np.max(proj_j, axis=0)
                min_j = np.min(proj_j, axis=0)
                overlap1 = max_i - min_j
                overlap2 = max_j - min_i
                overlaps = np.minimum(overlap1, overlap2)
                if np.min(overlaps) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Scale up positions
            scale = 1.0
            while scale < 5.0:
                positions = positions * scale
                verts = get_hex_verts(positions, ang)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        vi = verts[i]
                        vj = verts[j]
                        edges_i = np.roll(vi, -1, axis=0) - vi
                        normals_i = np.column_stack([-edges_i[:, 1], edges_i[:, 0]])
                        normals_i /= (np.linalg.norm(normals_i, axis=1, keepdims=True) + 1e-9)
                        edges_j = np.roll(vj, -1, axis=0) - vj
                        normals_j = np.column_stack([-edges_j[:, 1], edges_j[:, 0]])
                        normals_j /= (np.linalg.norm(normals_j, axis=1, keepdims=True) + 1e-9)
                        all_normals = np.vstack([normals_i, normals_j])
                        proj_i = vi @ all_normals.T
                        proj_j = vj @ all_normals.T
                        max_i = np.max(proj_i, axis=0)
                        min_i = np.min(proj_i, axis=0)
                        max_j = np.max(proj_j, axis=0)
                        min_j = np.min(proj_j, axis=0)
                        overlap1 = max_i - min_j
                        overlap2 = max_j - min_i
                        overlaps = np.minimum(overlap1, overlap2)
                        if np.min(overlaps) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                scale += 0.1
            
            pos = positions[:, 0]
            pos_y = positions[:, 1]
        
        best_params = np.zeros(3 * N)
        best_params[0::3] = pos
        best_params[1::3] = pos_y
        best_params[2::3] = ang
        
        # Calculate side
        all_verts = get_hex_verts(np.column_stack([pos, pos_y]), ang).reshape(-1, 2)
        x = all_verts[:, 0]
        y = all_verts[:, 1]
        L1 = x * (SQRT3/2) + y * 0.5
        L2 = y
        L3 = -x * (SQRT3/2) + y * 0.5
        L4 = -x * (SQRT3/2) - y * 0.5
        L5 = -y
        L6 = x * (SQRT3/2) - y * 0.5
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        best_side = np.max(all_L) * 2.0 / SQRT3

    # Extract results
    pos = best_params[0::3]
    pos_y = best_params[1::3]
    ang = best_params[2::3]
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i]), float(pos_y[i]), float(np.degrees(ang[i]))])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
