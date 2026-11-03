import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            # proj[k, i] = ox[k]*nx[i] + oy[k]*ny[i]
            proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Coarse scan first to find a good region
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune with minimize_scalar
        try:
            lo = max(0.0, best_phi - 2.0)
            hi = min(60.0, best_phi + 2.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-7, 'maxiter': 50})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Honeycomb initial configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Strategy: Use spectral graph partitioning to get a good initial layout,
    # then refine with simulated annealing.
    
    # Spectral Graph Partitioning Approach:
    # 1. Build a graph where nodes are hexagons
    # 2. Edge weights are inversely proportional to the Minkowski sum distance
    # 3. Use Fiedler vector to get a 2D embedding
    # 4. This gives a continuous layout that minimizes max support function violation
    
    def spectral_initial_layout():
        """
        Use spectral graph partitioning to get an initial layout.
        The idea is to model the packing as a graph problem where
        edge weights represent how "close" two hexagons should be.
        """
        # Start with the honeycomb positions
        positions = base_positions.copy()
        angles = base_angles.copy()
        
        # Build adjacency matrix based on proximity in honeycomb
        # In the honeycomb, hexagons that are neighbors should be close
        # We want to minimize the maximum distance to the container
        
        # Compute pairwise distances
        n = N
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_matrix[i, j] = math.sqrt(dx*dx + dy*dy)
        
        # Build weight matrix: inverse of distance (with regularization)
        # Closer hexagons get higher weights (they should stay close)
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Use a soft inverse distance
                    W[i, j] = 1.0 / (dist_matrix[i, j] + 0.1)
        
        # Compute degree matrix
        D = np.diag(W.sum(axis=1))
        
        # Compute Laplacian L = D - W
        L = D - W
        
        # Find Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
        # Use power iteration on (I - L/lambda_max) to find the Fiedler vector
        # First, estimate lambda_max
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        # Power iteration for largest eigenvalue of L
        for _ in range(100):
            v_new = L @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break
            v_new = v_new / norm
            if np.linalg.norm(v_new - v) < 1e-8:
                break
            v = v_new
        
        lambda_max = v @ (L @ v)
        
        # Now find Fiedler vector: eigenvector of L with second smallest eigenvalue
        # Use shifted inverse iteration or just compute all eigenvalues
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # The smallest eigenvalue is 0 (with eigenvector of all ones)
            # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            fiedler = eigenvectors[:, 1]
        except Exception:
            fiedler = np.random.randn(n)
            fiedler = fiedler / np.linalg.norm(fiedler)
        
        # Use the Fiedler vector to define a 1D embedding
        # For 2D, we can use the first two non-trivial eigenvectors
        try:
            # Get the second and third eigenvectors
            embed_1 = eigenvectors[:, 1]
            embed_2 = eigenvectors[:, 2]
        except Exception:
            embed_1 = fiedler
            embed_2 = np.random.randn(n)
            embed_2 = embed_2 - np.mean(embed_2)
            embed_2 = embed_2 / np.linalg.norm(embed_2)
        
        # Scale the embedding to match the expected scale
        # The honeycomb has a certain extent, we want to preserve that
        extent_1 = np.max(embed_1) - np.min(embed_1)
        extent_2 = np.max(embed_2) - np.min(embed_2)
        
        # Target extent based on honeycomb
        honeycomb_extent = np.max(base_positions[:, 0]) - np.min(base_positions[:, 0])
        scale = honeycomb_extent / max(extent_1, extent_2, 0.1)
        
        # Create new positions from the spectral embedding
        new_positions = np.zeros((n, 2))
        for i in range(n):
            new_positions[i, 0] = embed_1[i] * scale
            new_positions[i, 1] = embed_2[i] * scale
        
        # Center the positions
        center_x = np.mean(new_positions[:, 0])
        center_y = np.mean(new_positions[:, 1])
        new_positions[:, 0] -= center_x
        new_positions[:, 1] -= center_y
        
        # Check if this is valid
        angles_rad = np.zeros(n)
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                h1 = (new_positions[i, 0], new_positions[i, 1], angles_rad[i])
                h2 = (new_positions[j, 0], new_positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side, phi, cx, cy = compute_min_outer_side(new_positions, np.zeros(n))
            return new_positions, np.zeros(n), side, phi, cx, cy
        
        # If invalid, fall back to honeycomb
        side, phi, cx, cy = compute_min_outer_side(base_positions, base_angles)
        return base_positions.copy(), base_angles.copy(), side, phi, cx, cy
    
    # Try spectral initialization
    spec_positions, spec_angles, spec_side, spec_phi, spec_cx, spec_cy = spectral_initial_layout()
    
    if spec_side < best_side:
        best_side = spec_side
        best_positions = spec_positions.copy()
        best_angles = spec_angles.copy()
        best_phi = spec_phi
        best_cx = spec_cx
        best_cy = spec_cy
    else:
        spec_positions = best_positions.copy()
        spec_angles = best_angles.copy()
    
    # Run SA from the best initial configuration
    sa_time_budget = DEADLINE * 0.65
    
    current_positions = spec_positions.copy()
    current_angles = spec_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.15
    T_min = 0.00005
    cooling = 0.9999
    step_size = 0.15
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
    eval_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 2.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002
            continue
            
        eval_count += 1
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        
        # Adaptive step size
        if total_count % 200 == 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.05:
                step_size *= 0.7
                if step_size < 0.002: step_size = 0.002
            elif acc_rate > 0.35:
                step_size *= 1.25
                if step_size > 0.5: step_size = 0.5
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.002: step_size = 0.002

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side_cd, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side_cd
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.03
            min_step = 0.0005
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                            
                            if new_side < best_side_cd - 1e-12:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                        
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.75
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.5, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. Nelder-Mead Polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def polish_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                    
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6
                
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-9})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
