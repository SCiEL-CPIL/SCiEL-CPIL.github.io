import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 570
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    best_positions = None
    best_angle = None
    
    # Triangular lattice basis vectors for unit hexagons (side 1)
    # Standard triangular lattice: nearest neighbor distance = sqrt(3) for hexagons of side 1
    h = np.sqrt(3) / 2
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    # But we'll use a more general parameterization
    # Integer coefficients for 12 hexagons in a hexagonal arrangement
    # Center + 6 around + 5 more (total 12)
    # Using a hexagonal number arrangement:
    # Layer 0: (0,0)
    # Layer 1: 6 hexagons
    # Layer 2: 5 more (to get 12)
    
    # Standard hexagonal packing of 12 hexagons:
    # Center at origin, then 6 around it, then 5 more in the next ring
    lattice_coeffs = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [0, -1],
        [1, -1],
        [2, 0],
        [1, 1],
        [0, 2],
        [-2, 1],
        [-1, -1]
    ], dtype=float)
    
    def lattice_positions(lattice_params):
        """
        lattice_params: [a1x, a1y, a2x, a2y, rotation_deg]
        Returns positions of 12 hexagons on a triangular lattice.
        """
        a1 = np.array([lattice_params[0], lattice_params[1]])
        a2 = np.array([lattice_params[2], lattice_params[3]])
        rot = np.deg2rad(lattice_params[4])
        
        # Rotation matrix
        R = np.array([[np.cos(rot), -np.sin(rot)],
                      [np.sin(rot), np.cos(rot)]])
        
        # Compute positions
        positions = np.zeros((12, 2))
        for i in range(12):
            pos = lattice_coeffs[i, 0] * a1 + lattice_coeffs[i, 1] * a2
            positions[i] = R @ pos
        return positions
    
    def lattice_objective(lattice_params):
        positions = lattice_positions(lattice_params)
        angle_deg = lattice_params[4]
        s = _min_outer_side(positions, angle_deg)
        
        # Add penalty for overlapping hexagons
        penalty = 0.0
        unit_hex_verts = _hex_vertices(0, 0, 1, 0)
        for i in range(12):
            for j in range(i+1, 12):
                vi = positions[i] + unit_hex_verts
                vj = positions[j] + unit_hex_verts
                
                # Check separation using SAT
                separated = False
                for k in range(6):
                    v1 = vi[k]
                    v2 = vi[(k+1)%6]
                    e = v2 - v1
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12: continue
                    n = n / norm
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                    if gap > 1e-6:
                        separated = True
                        break
                if not separated:
                    for k in range(6):
                        v1 = vj[k]
                        v2 = vj[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap > 1e-6:
                            separated = True
                            break
                if not separated:
                    # Penalize overlap
                    # Estimate minimum separation (negative if overlapping)
                    min_sep = float('inf')
                    for k in range(6):
                        v1 = vi[k]
                        v2 = vi[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap < min_sep:
                            min_sep = gap
                    for k in range(6):
                        v1 = vj[k]
                        v2 = vj[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap < min_sep:
                            min_sep = gap
                    if min_sep < 0:
                        penalty += (-min_sep) ** 2 * 100
        
        return s + penalty
    
    def free_objective(x):
        positions = x[:24].reshape(12, 2)
        angle_deg = x[24]
        s = _min_outer_side(positions, angle_deg)
        
        # Add penalty for overlapping hexagons
        penalty = 0.0
        unit_hex_verts = _hex_vertices(0, 0, 1, 0)
        for i in range(12):
            for j in range(i+1, 12):
                vi = positions[i] + unit_hex_verts
                vj = positions[j] + unit_hex_verts
                
                # Check separation using SAT
                separated = False
                for k in range(6):
                    v1 = vi[k]
                    v2 = vi[(k+1)%6]
                    e = v2 - v1
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12: continue
                    n = n / norm
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                    if gap > 1e-6:
                        separated = True
                        break
                if not separated:
                    for k in range(6):
                        v1 = vj[k]
                        v2 = vj[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap > 1e-6:
                            separated = True
                            break
                if not separated:
                    # Estimate minimum separation (negative if overlapping)
                    min_sep = float('inf')
                    for k in range(6):
                        v1 = vi[k]
                        v2 = vi[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap < min_sep:
                            min_sep = gap
                    for k in range(6):
                        v1 = vj[k]
                        v2 = vj[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_i = np.dot(vi, n)
                        proj_j = np.dot(vj, n)
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap < min_sep:
                            min_sep = gap
                    if min_sep < 0:
                        penalty += (-min_sep) ** 2 * 100
        
        return s + penalty
    
    def update_best(positions, angle_deg):
        nonlocal best_score, best_inner, best_outer, best_s, best_positions, best_angle
        s = _min_outer_side(positions, angle_deg)
        
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = 0.0
        outer_candidate = np.array([0, 0, angle_deg])
        
        if _check_validity(inner_candidate, outer_candidate, s):
            score_candidate = 1.0 / s
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate.copy()
                best_outer = outer_candidate.copy()
                best_s = s
                best_positions = positions.copy()
                best_angle = angle_deg
    
    # Phase 1: Lattice-constrained optimization
    # Initial lattice parameters: standard triangular lattice
    # a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5), rotation = 0
    a1_init = np.array([np.sqrt(3), 0])
    a2_init = np.array([np.sqrt(3)/2, 1.5])
    
    lattice_starts = [
        [a1_init[0], a1_init[1], a2_init[0], a2_init[1], 0.0],
        [a1_init[0], a1_init[1], a2_init[0], a2_init[1], 30.0],
        [a1_init[0] * 0.95, a1_init[1], a2_init[0] * 0.95, a2_init[1], 0.0],
        [a1_init[0] * 1.05, a1_init[1], a2_init[0] * 1.05, a2_init[1], 0.0],
        [a1_init[0], a1_init[1] * 1.1, a2_init[0], a2_init[1] * 1.1, 0.0],
        [a1_init[0] * 0.9, a1_init[1], a2_init[0] * 0.9, a2_init[1], 15.0],
    ]
    
    for start in lattice_starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                lattice_objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < 100:  # Reasonable solution
                positions = lattice_positions(res.x)
                angle_deg = res.x[4]
                update_best(positions, angle_deg)
        except Exception:
            continue
    
    # Phase 2: Free-position refinement from best lattice solution
    if best_positions is not None:
        x0 = np.concatenate([best_positions.flatten(), [best_angle]])
        
        # Multiple refinement starts
        np.random.seed(42)
        for r in range(5):
            if time.time() > deadline:
                break
            
            if r == 0:
                x_start = x0
            else:
                noise_pos = np.random.randn(12, 2) * 0.1
                noise_angle = np.random.uniform(-2, 2)
                x_start = np.concatenate([(best_positions + noise_pos).flatten(), [best_angle + noise_angle]])
            
            try:
                res = minimize(
                    free_objective,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if res.fun < 100:  # Reasonable solution
                    positions = res.x[:24].reshape(12, 2)
                    angle_deg = res.x[24]
                    update_best(positions, angle_deg)
            except Exception:
                continue
    
    # Phase 3: Additional free-position starts from different initial configurations
    if time.time() < deadline - 30:
        # Start from slightly perturbed standard hexagonal packing
        h = np.sqrt(3)/2
        init_pos = np.array([
            [0, 0],
            [2*h, 0],
            [h, 1.5],
            [-h, 1.5],
            [-2*h, 0],
            [-h, -1.5],
            [h, -1.5],
            [3*h, 1.5],
            [0, 3],
            [-3*h, 1.5],
            [-3*h, -1.5],
            [3*h, -1.5]
        ])
        
        for r in range(3):
            if time.time() > deadline:
                break
            
            if r == 0:
                x_start = np.concatenate([init_pos.flatten(), [0.0]])
            else:
                noise_pos = np.random.randn(12, 2) * 0.2
                noise_angle = np.random.uniform(-5, 5)
                x_start = np.concatenate([(init_pos + noise_pos).flatten(), [noise_angle]])
            
            try:
                res = minimize(
                    free_objective,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if res.fun < 100:
                    positions = res.x[:24].reshape(12, 2)
                    angle_deg = res.x[24]
                    update_best(positions, angle_deg)
            except Exception:
                continue
    
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback: standard hexagonal packing
        h = np.sqrt(3)/2
        inner_fallback = np.array([
            [0, 0, 0],
            [2*h, 0, 0],
            [h, 1.5, 0],
            [-h, 1.5, 0],
            [-2*h, 0, 0],
            [-h, -1.5, 0],
            [h, -1.5, 0],
            [3*h, 1.5, 0],
            [0, 3, 0],
            [-3*h, 1.5, 0],
            [-3*h, -1.5, 0],
            [3*h, -1.5, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
