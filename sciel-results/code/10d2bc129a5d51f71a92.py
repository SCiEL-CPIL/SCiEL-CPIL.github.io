# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import itertools

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(cx, cy, d, angle_deg):
    """Compute outward edge normals of a regular hexagon."""
    verts = _hex_vertices(cx, cy, d, angle_deg)
    normals = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k+1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n = n / n_norm
            normals.append(n)
    return np.array(normals)

def _separation_check(inner, tol=1e-6):
    """Check separation between all pairs of inner hexagons using SAT."""
    for i in range(12):
        v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        n1 = _hex_edge_normals(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, 12):
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            n2 = _hex_edge_normals(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            all_normals = np.vstack([n1, n2])
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                gap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                if gap > -tol:
                    separated = True
                    break
            if not separated:
                return False
    return True

def _is_inside_outer(inner, outer_verts):
    """Check all inner vertices are inside outer hexagon."""
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    return False
    return True

def _validate(inner, outer, s, tol=1e-6):
    """Full validation."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check unit edges
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k+1)%6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Separation
    if not _separation_check(inner, tol):
        return False
    
    # Inside outer
    if not _is_inside_outer(inner, outer_verts):
        return False
    
    return True

def _compute_min_outer(inner, outer_angle_deg, outer_center):
    """Compute minimum outer side length to contain all inner hexagons."""
    oang = np.deg2rad(outer_angle_deg)
    all_verts = np.vstack([_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(12)])
    
    n_angles = oang + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    dots = all_verts @ normals.T - outer_center @ normals.T
    max_dots = dots.max(axis=0)
    s_min = (2.0 / math.sqrt(3.0)) * max_dots.max()
    return s_min

def _lattice_positions(m, n):
    """Convert triangular lattice coords (m,n) to Cartesian with unit spacing."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, math.sqrt(3)/2])
    return m * a1 + n * a2

def _build_inner_from_lattice(lattice_coords, scale, rotation_deg, center_offset=None):
    """Build inner array from lattice coordinates with given scale and rotation."""
    d = scale  # center-to-center distance
    
    centers = []
    for (m, n) in lattice_coords:
        pos = d * _lattice_positions(m, n)
        centers.append(pos)
    centers = np.array(centers)
    
    # Center the cluster
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    
    # Rotate
    rot = np.deg2rad(rotation_deg)
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    centers = centers @ R.T
    
    if center_offset is not None:
        centers = centers + center_offset
    
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = centers[i, 0]
        inner[i, 1] = centers[i, 1]
        inner[i, 2] = rotation_deg
    
    return inner

def _objective_lattice(params, lattice_coords):
    """
    Objective function for lattice-constrained optimization.
    params: [scale, rotation_deg]
    Returns s (outer side length) or large value if invalid.
    """
    scale = params[0]
    rotation_deg = params[1]
    
    if scale < 0:
        return 1000.0
    
    inner = _build_inner_from_lattice(lattice_coords, scale, rotation_deg)
    
    # Check separation
    if not _separation_check(inner, tol=1e-6):
        return 1000.0
    
    # Compute minimum enclosing hexagon
    best_s = float('inf')
    best_oa = 0.0
    for oa in np.linspace(0, 60, 31):
        s = _compute_min_outer(inner, oa, np.array([0.0, 0.0]))
        if s < best_s:
            best_s = s
            best_oa = oa
    
    outer = np.array([0.0, 0.0, best_oa])
    
    if not _validate(inner, outer, best_s):
        return 1000.0
    
    return best_s

def _free_refinement(inner, outer, s, max_iters=200, step_size=0.01):
    """
    Free-position refinement: perturb inner hexagon positions and angles
    to reduce outer side length while maintaining validity.
    """
    best_inner = inner.copy()
    best_outer = outer.copy()
    best_s = s
    
    for iteration in range(max_iters):
        improved = False
        # Try random perturbations
        for _ in range(50):
            trial_inner = best_inner.copy()
            
            # Perturb a random subset of hexagons
            num_perturbed = np.random.randint(1, 4)
            indices = np.random.choice(12, size=num_perturbed, replace=False)
            
            for idx in indices:
                # Small position perturbation
                dx = np.random.normal(0, step_size)
                dy = np.random.normal(0, step_size)
                trial_inner[idx, 0] += dx
                trial_inner[idx, 1] += dy
                
                # Small angle perturbation
                dangle = np.random.normal(0, step_size * 5)
                trial_inner[idx, 2] += dangle
            
            # Check separation
            if not _separation_check(trial_inner, tol=1e-6):
                continue
            
            # Compute new outer
            ocx, ocy, oang = best_outer
            new_s = _compute_min_outer(trial_inner, oang, np.array([ocx, ocy]))
            
            # Also try optimizing outer angle
            best_new_s = new_s
            best_new_oa = oang
            for oa in np.linspace(oang - 10, oang + 10, 11):
                s_test = _compute_min_outer(trial_inner, oa, np.array([ocx, ocy]))
                if s_test < best_new_s:
                    best_new_s = s_test
                    best_new_oa = oa
            
            if best_new_s < best_s - 1e-8:
                new_outer = np.array([ocx, ocy, best_new_oa])
                if _validate(trial_inner, new_outer, best_new_s):
                    best_inner = trial_inner
                    best_outer = new_outer
                    best_s = best_new_s
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_inner, best_outer, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 610.0
    
    best_state = {'inner': None, 'outer': None, 's': float('inf')}
    
    def _update_best(inner, outer, s):
        if s < best_state['s'] and _validate(inner, outer, s):
            best_state['inner'] = inner.copy()
            best_state['outer'] = outer.copy()
            best_state['s'] = s
    
    # Define candidate lattice configurations for 12 hexagons
    # All use triangular lattice coordinates
    lattice_configs = [
        # Hexagonal cluster: 1 + 6 + 5
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
         (2,0), (1,1), (-1,2), (-2,0), (0,-2)],
        # Alternative ring 2 selection
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
         (2,0), (1,1), (-1,2), (-2,0), (-1,-1)],
        # Another alternative
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
         (2,0), (1,1), (-1,2), (-2,0), (1,-2)],
        # More spread out
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
         (2,0), (2,-1), (-2,1), (-2,0), (0,-2)],
    ]
    
    # Phase 1: Lattice-constrained optimization
    for config in lattice_configs:
        if time.time() > deadline - 120:
            break
        
        # Initial guess: scale = 2*sqrt(3) for touching hexagons
        initial_scale = 2 * math.sqrt(3)
        
        try:
            from scipy.optimize import minimize
            
            # Optimize over [scale, rotation]
            initial_params = np.array([initial_scale, 0.0])
            
            res = minimize(
                _objective_lattice,
                initial_params,
                args=(config,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < 1000:
                scale, rotation = res.x
                inner = _build_inner_from_lattice(config, scale, rotation)
                
                # Compute optimal outer
                best_s = float('inf')
                best_oa = 0.0
                for oa in np.linspace(0, 60, 61):
                    s = _compute_min_outer(inner, oa, np.array([0.0, 0.0]))
                    if s < best_s:
                        best_s = s
                        best_oa = oa
                
                outer = np.array([0.0, 0.0, best_oa])
                _update_best(inner, outer, best_s)
                
        except Exception as e:
            pass
    
    # Phase 2: Free-position refinement
    if best_state['inner'] is not None and time.time() < deadline - 60:
        inner, outer, s = _free_refinement(
            best_state['inner'], best_state['outer'], best_state['s'],
            max_iters=100, step_size=0.05
        )
        _update_best(inner, outer, s)
    
    # Phase 3: Try more lattice configurations with different initial scales
    if time.time() < deadline - 30:
        for config in lattice_configs[:2]:
            for init_scale in [2*math.sqrt(3) - 0.1, 2*math.sqrt(3), 2*math.sqrt(3) + 0.1]:
                if time.time() > deadline - 15:
                    break
                try:
                    from scipy.optimize import minimize
                    initial_params = np.array([init_scale, np.random.uniform(0, 60)])
                    res = minimize(
                        _objective_lattice,
                        initial_params,
                        args=(config,),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if res.fun < 1000:
                        scale, rotation = res.x
                        inner = _build_inner_from_lattice(config, scale, rotation)
                        best_s = float('inf')
                        best_oa = 0.0
                        for oa in np.linspace(0, 60, 61):
                            s = _compute_min_outer(inner, oa, np.array([0.0, 0.0]))
                            if s < best_s:
                                best_s = s
                                best_oa = oa
                        outer = np.array([0.0, 0.0, best_oa])
                        _update_best(inner, outer, best_s)
                except Exception as e:
                    pass
    
    # Fallback
    if best_state['inner'] is None:
        # Use the first lattice config with default parameters
        config = lattice_configs[0]
        inner = _build_inner_from_lattice(config, 2*math.sqrt(3), 0.0)
        best_s = float('inf')
        best_oa = 0.0
        for oa in np.linspace(0, 60, 61):
            s = _compute_min_outer(inner, oa, np.array([0.0, 0.0]))
            if s < best_s:
                best_s = s
                best_oa = oa
        outer = np.array([0.0, 0.0, best_oa])
        return inner, outer, best_s
    
    return best_state['inner'], best_state['outer'], best_state['s']

# EVOLVE-BLOCK-END
