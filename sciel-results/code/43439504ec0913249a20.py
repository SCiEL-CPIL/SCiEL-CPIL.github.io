import numpy as np
from scipy.optimize import minimize

def hexagon_vertices(cx, cy, side, angle_deg):
    """Calculate vertices of a regular hexagon."""
    angles = np.deg2rad(angle_deg + np.arange(6) * 60)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def get_edge_normals(vertices):
    """Get outward edge normals for a convex polygon given in CCW order."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    dx = edges[:, 0]
    dy = edges[:, 1]
    normals = np.column_stack((dy, -dx))
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    return normals

def compute_penetrations(params, s_outer):
    """
    Compute penetration depths for all pairs.
    Returns:
        inner_penetrations: (66,) array of penetration depths for inner-inner pairs
        outer_penetrations: (72,) array of penetration depths for inner-outer vertex-edge pairs
    """
    n_inner = 12
    inner_params = params.reshape(n_inner, 3)
    
    # Generate vertices and normals for all inner hexagons
    inner_verts = []
    inner_normals = []
    for i in range(n_inner):
        x, y, ang = inner_params[i]
        v = hexagon_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
        inner_normals.append(get_edge_normals(v))
    
    # Outer hexagon
    outer_v = hexagon_vertices(0, 0, s_outer, 0)
    outer_normals = get_edge_normals(outer_v)
    
    # Inner-inner penetrations (66 pairs)
    inner_pen = []
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            A = inner_verts[i]
            B = inner_verts[j]
            normals = np.vstack((inner_normals[i], inner_normals[j]))
            proj_A = np.min(A @ normals.T, axis=0)
            proj_B = np.max(B @ normals.T, axis=0)
            gaps = proj_A - proj_B
            max_gap = np.max(gaps)
            # Penetration is -max_gap if overlapping, else 0
            pen = max(0, -max_gap)
            inner_pen.append(pen)
    
    # Inner-outer penetrations (12 hexagons * 6 vertices * 6 outer edges = 432, but we use 72 = 12*6)
    # Actually, for each inner hexagon, we need to check if any vertex is outside
    # We'll compute the maximum penetration per inner hexagon
    outer_pen = []
    for i in range(n_inner):
        v = inner_verts[i]
        max_pen = 0.0
        for k in range(6):
            pt = v[k]
            for m in range(6):
                p1 = outer_v[m]
                n = outer_normals[m]
                dist = np.dot(pt - p1, n)
                if dist > 0:
                    max_pen = max(max_pen, dist)
        outer_pen.append(max_pen)
    
    return np.array(inner_pen), np.array(outer_pen)

def pressure_based_energy(params, s_outer, inner_pressures, outer_pressures):
    """
    Compute total potential energy using pressure-based formulation.
    
    Energy = sum over inner-inner pairs: 0.5 * p_ij * d_ij^2
           + sum over inner hexagons: p_i * d_i^2
    
    where d is penetration depth and p is pressure.
    """
    inner_pen, outer_pen = compute_penetrations(params, s_outer)
    
    # Inner-inner energy
    E_inner = 0.5 * np.sum(inner_pressures * inner_pen ** 2)
    
    # Outer energy
    E_outer = np.sum(outer_pressures * outer_pen ** 2)
    
    return E_inner + E_outer

def update_pressures(params, s_outer, inner_pressures, outer_pressures, learning_rate=0.1):
    """
    Update pressures based on current penetrations.
    Pressure increases where there is penetration, decreases where there isn't.
    """
    inner_pen, outer_pen = compute_penetrations(params, s_outer)
    
    # Update inner-inner pressures
    new_inner_p = np.maximum(0, inner_pressures + learning_rate * inner_pen)
    
    # Update outer pressures
    new_outer_p = np.maximum(0, outer_pressures + learning_rate * outer_pen)
    
    return new_inner_p, new_outer_p

def minimize_energy_for_s(s_target, init_params, max_iters=200, tol=1e-8):
    """
    Minimize the pressure-based energy for a given s.
    Returns (valid, best_params, min_energy)
    """
    n_inner = 12
    n_pairs = 66
    
    # Initialize pressures
    inner_pressures = np.ones(n_pairs) * 0.1
    outer_pressures = np.ones(n_inner) * 0.1
    
    best_params = init_params.copy()
    best_energy = float('inf')
    
    for it in range(max_iters):
        # Compute current energy
        energy = pressure_based_energy(best_params, s_target, inner_pressures, outer_pressures)
        
        if energy < best_energy:
            best_energy = energy
        
        # Check convergence
        inner_pen, outer_pen = compute_penetrations(best_params, s_target)
        max_pen = max(np.max(inner_pen), np.max(outer_pen))
        
        if max_pen < tol:
            return True, best_params, energy
        
        # Use L-BFGS-B to minimize energy with current pressures
        res = minimize(
            pressure_based_energy,
            best_params,
            args=(s_target, inner_pressures, outer_pressures),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
        
        # Update pressures
        inner_pressures, outer_pressures = update_pressures(
            best_params, s_target, inner_pressures, outer_pressures, learning_rate=0.05
        )
    
    # Final check
    inner_pen, outer_pen = compute_penetrations(best_params, s_target)
    max_pen = max(np.max(inner_pen), np.max(outer_pen))
    valid = max_pen < 1e-6
    
    return valid, best_params, best_energy

def hexagon_packing_12():
    n = 12
    
    # Initial guess: Honeycomb pattern
    init_params = np.array([
        [0, 0, 0],
        [1.5, 0, 0],
        [-1.5, 0, 0],
        [0, 2.598, 0],
        [0, -2.598, 0],
        [1.5, 2.598, 0],
        [-1.5, 2.598, 0],
        [3.0, 2.598, 0],
        [-3.0, 2.598, 0],
        [3.0, -2.598, 0],
        [-3.0, -2.598, 0],
        [0, -5.196, 0]
    ]).flatten()
    
    # Binary search for s
    s_low = np.sqrt(12) * 1.01
    s_high = 8.0
    
    best_s = s_high
    best_params = init_params.copy()
    
    # Try to find valid packing for various s values
    for _ in range(12):
        s_mid = (s_low + s_high) / 2
        
        # Try multiple restarts with jitter
        valid = False
        for restart in range(3):
            if restart == 0:
                p0 = init_params.copy()
            else:
                p0 = init_params + np.random.normal(0, 0.05, size=init_params.shape)
            
            valid, params, _ = minimize_energy_for_s(s_mid, p0, max_iters=100)
            if valid:
                best_s = s_mid
                best_params = params
                s_low = s_mid
                break
        
        if not valid:
            s_high = s_mid
    
    # Final optimization with best_s
    for restart in range(5):
        if restart == 0:
            p0 = best_params.copy()
        else:
            p0 = best_params + np.random.normal(0, 0.02, size=best_params.shape)
        
        valid, params, _ = minimize_energy_for_s(best_s, p0, max_iters=150)
        if valid:
            best_params = params
            break
    
    # Fallback if no valid packing found
    if not valid:
        # Use a known good configuration with larger s
        best_s = 5.0
        init_params = np.array([
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
            [0, -4, 0]
        ]).flatten()
        valid, best_params, _ = minimize_energy_for_s(best_s, init_params, max_iters=200)
        if not valid:
            best_s = 8.0
            best_params = init_params
    
    # Construct return values
    inner_hex_data = best_params.reshape(12, 3)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s
