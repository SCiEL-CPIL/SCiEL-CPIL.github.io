# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

# Precompute unit hexagon vertices (angle 0)
_UNIT_COS = np.cos(np.deg2rad(60 * np.arange(6)))
_UNIT_SIN = np.sin(np.deg2rad(60 * np.arange(6)))
_UNIT_VERTS = np.column_stack([_UNIT_COS, _UNIT_SIN])  # (6,2)

def _compute_min_side_fast(inner_positions, outer_center, outer_angle_deg):
    """
    Fast computation of minimum outer hexagon side length.
    inner_positions: (12, 2) array of centers
    outer_center: (2,) array
    outer_angle_deg: float
    Returns s (scalar)
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    
    # All 72 vertices relative to outer center
    # inner_positions (12,2) + _UNIT_VERTS (6,2) -> (12,6,2)
    all_verts = inner_positions[:, None, :] + _UNIT_VERTS[None, :, :] 
    all_verts = all_verts.reshape(72, 2) - outer_center[None, :]
    
    # Project onto each normal direction
    projs = all_verts @ normals.T  # (72, 6)
    max_projs = np.max(projs, axis=0)  # (6,)
    max_inradius = np.max(max_projs)
    if max_inradius < 0:
        max_inradius = 0.0
    s = 2.0 * max_inradius / np.sqrt(3.0)
    return s

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    """Separating axis test: return True if hexagons are separated (no overlap)."""
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    """Check if vertex v is inside or on the boundary of outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Return True if valid."""
    if s < np.sqrt(12) - 1e-8:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        inner_verts_list.append(verts)
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _make_hex_lattice(n_rings, spacing):
    """Create a hexagonal lattice of positions."""
    positions = [[0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        for i in range(6 * ring):
            angle = np.deg2rad(60 * (i // ring + 1) + (i % ring) * (60.0 / ring))
            positions.append([ring * spacing * np.cos(angle),
                              ring * spacing * np.sin(angle)])
    positions = np.array(positions[:12])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _make_grid(rows, cols, dx, dy):
    """Create a rectangular grid of positions."""
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2.0) * dx
            y = (r - (rows - 1) / 2.0) * dy
            if r % 2 == 1:
                x += dx / 2
            positions.append([x, y])
    positions = np.array(positions[:12])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _objective(params, outer_angle_deg):
    """Objective: minimize the outer hexagon side length."""
    inner_pos = params[:24].reshape(12, 2)
    oc = params[24:26]
    return _compute_min_side_fast(inner_pos, oc, outer_angle_deg)

def _nelder_mead_fast(init_params, outer_angle_deg, maxiter=5000, deadline=None):
    """
    Nelder-Mead optimization with fast objective.
    """
    def obj(p):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _objective(p, outer_angle_deg)
    
    # Use adaptive Nelder-Mead for better convergence
    result = minimize(obj, init_params, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True, 'initial_simplex': 0.1})
    return result.x, result.fun

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    # 630s limit, keep 30s safety margin
    deadline = start_time + 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Hexagonal lattice with different spacings
    for spacing in [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
        initial_configs.append(_make_hex_lattice(2, spacing))
    
    # Grid configurations
    for dx, dy in [(2.0, sqrt3*0.8), (2.2, sqrt3*0.9), (2.4, sqrt3), (2.6, sqrt3*1.1), (2.8, sqrt3*1.2)]:
        initial_configs.append(_make_grid(3, 4, dx, dy))
        initial_configs.append(_make_grid(4, 3, dx, dy))
    
    # Random perturbations
    rng = np.random.default_rng(42)
    for seed in range(10):
        rng_local = np.random.default_rng(seed)
        base = _make_hex_lattice(2, 2.5)
        base += rng_local.normal(0, 0.5, size=(12, 2))
        initial_configs.append(base)
    
    # Try different outer angles
    # The symmetry of the problem means angles 0-60 are equivalent, but due to numerical 
    # optimization and potential asymmetries in the inner packing, we test a few.
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    best_s = np.inf
    best_params = None
    best_angle = 0
    
    # We will perform multiple Nelder-Mead runs.
    # Since the objective is fast, we can do many runs.
    # Strategy: Iterate through angles and initial configs.
    
    # To ensure we don't spend too much time on one bad start, we limit maxiter per run
    # and rely on the loop to try more starts.
    
    total_configs = len(outer_angles) * len(initial_configs)
    # Estimate time per run: ~10-50ms for fast NM with 5000 iters (usually converges faster)
    # Let's aim for ~100 runs max if possible, but constrained by time.
    
    run_count = 0
    for angle_deg in outer_angles:
        if time.time() > deadline - 60:
            break
            
        for init_pos in initial_configs:
            if time.time() > deadline - 60:
                break
            
            init_params = np.concatenate([init_pos.flatten(), np.array([0.0, 0.0])])
            
            # Run Nelder-Mead
            # Use a moderate maxiter. If it converges, it stops early.
            # If it hits maxiter, it might be stuck, so we move on to the next start.
            try:
                nm_params, nm_val = _nelder_mead_fast(init_params, angle_deg, 
                                                       maxiter=3000, deadline=deadline - 60)
                run_count += 1
                if nm_val < best_s:
                    best_s = nm_val
                    best_params = nm_params
                    best_angle = angle_deg
            except Exception:
                pass

    # If we have time left, refine the best solution found so far with a longer run
    if best_params is not None and time.time() < deadline - 30:
        try:
            nm_params, nm_val = _nelder_mead_fast(best_params, best_angle,
                                                   maxiter=10000, deadline=deadline - 30)
            if nm_val < best_s:
                best_s = nm_val
                best_params = nm_params
        except Exception:
            pass
            
    # Final check: If best_params is still None (shouldn't happen), use a fallback
    if best_params is None:
        init_pos = _make_hex_lattice(2, 2.5)
        best_params = np.concatenate([init_pos.flatten(), np.array([0.0, 0.0])])
        best_angle = 0
        best_s = _objective(best_params, best_angle)

    # Extract best solution
    inner_positions = best_params[:24].reshape(12, 2)
    outer_center = best_params[24:26]
    
    # Build return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], float(best_angle)])
    s_final = best_s
    
    # Validate
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        # Fallback to a known valid configuration if optimization failed validation
        # This is a safety net. The optimization should produce valid results.
        inner_hex_data = np.zeros((12, 3))
        positions_fb = _make_hex_lattice(2, 2.5)
        for i in range(12):
            inner_hex_data[i, 0] = positions_fb[i, 0]
            inner_hex_data[i, 1] = positions_fb[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side_fast(positions_fb, np.array([0.0, 0.0]), 0.0) + 0.3
        
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
