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
    all_verts = inner_positions[:, None, :] + _UNIT_VERTS[None, :, :]  # (12, 6, 2)
    all_verts = all_verts.reshape(72, 2) - outer_center[None, :]  # (72, 2)
    
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

def _make_honeycomb_343(spacing):
    """
    Create a 3-4-3 honeycomb topology: 3 hexagons in row 0, 4 in row 1, 3 in row 2.
    This mimics a compact hexagonal arrangement.
    """
    positions = []
    sqrt3 = np.sqrt(3)
    # Row 0: 3 hexagons
    for i in range(3):
        x = (i - 1) * spacing
        y = 0
        positions.append([x, y])
    # Row 1: 4 hexagons, offset by spacing/2 in x, shifted by spacing*sqrt(3)/2 in y
    for i in range(4):
        x = (i - 1.5) * spacing
        y = spacing * sqrt3 / 2
        positions.append([x, y])
    # Row 2: 3 hexagons
    for i in range(3):
        x = (i - 1) * spacing
        y = spacing * sqrt3
        positions.append([x, y])
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _make_honeycomb_434(spacing):
    """
    Create a 4-3-4 honeycomb topology: 4 hexagons in row 0, 3 in row 1, 4 in row 2.
    """
    positions = []
    sqrt3 = np.sqrt(3)
    # Row 0: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * spacing
        y = 0
        positions.append([x, y])
    # Row 1: 3 hexagons, offset
    for i in range(3):
        x = (i - 1) * spacing
        y = spacing * sqrt3 / 2
        positions.append([x, y])
    # Row 2: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * spacing
        y = spacing * sqrt3
        positions.append([x, y])
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _objective(params, outer_angle_deg):
    """Objective: minimize the outer hexagon side length."""
    inner_pos = params[:24].reshape(12, 2)
    oc = params[24:26]
    return _compute_min_side_fast(inner_pos, oc, outer_angle_deg)

def _objective_with_penalty(params, outer_angle_deg, penalty_weight=50.0):
    """Objective with penalty for overlapping hexagons."""
    inner_pos = params[:24].reshape(12, 2)
    oc = params[24:26]
    s = _compute_min_side_fast(inner_pos, oc, outer_angle_deg)
    
    # Check for overlaps - add penalty based on minimum distance
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(inner_pos[i] - inner_pos[j])
            # For unit hexagons, the minimum center distance for non-overlap depends on relative angle.
            # With angle 0 for all, the minimum distance is 2.0 (touching sides) or sqrt(3) (touching vertices).
            # Actually, for two axis-aligned unit hexagons, the minimum distance to avoid overlap is 2.0 in the normal direction.
            # Let's use a conservative bound. The inradius is sqrt(3)/2. So distance >= 2*inradius = sqrt(3) is not enough.
            # The minimum distance between centers for non-overlapping unit hexagons (same orientation) is 2.0.
            min_dist = 2.0
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return s + penalty_weight * penalty

def _nelder_mead(init_params, outer_angle_deg, maxiter=5000, deadline=None):
    """Nelder-Mead optimization."""
    def obj(p):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _objective(p, outer_angle_deg)
    
    result = minimize(obj, init_params, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
    return result.x, result.fun

def _nelder_mead_penalty(init_params, outer_angle_deg, maxiter=5000, deadline=None):
    """Nelder-Mead optimization with penalty."""
    def obj(p):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _objective_with_penalty(p, outer_angle_deg)
    
    result = minimize(obj, init_params, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11,
                               'adaptive': True})
    return result.x, result.fun

def _refine_with_angles(params, outer_angle_deg, deadline):
    """
    Refine a solution by trying small variations in the outer angle.
    Returns best params and angle.
    """
    best_params = params.copy()
    best_val = _objective(params, outer_angle_deg)
    best_angle = outer_angle_deg
    
    # Try small angle perturbations
    for da in [-5, -3, -2, -1, 1, 2, 3, 5]:
        if time.time() > deadline:
            break
        new_angle = (outer_angle_deg + da) % 180
        p, v = _nelder_mead(params, new_angle, maxiter=2000, deadline=deadline)
        if v < best_val:
            best_val = v
            best_params = p
            best_angle = new_angle
    
    return best_params, best_angle, best_val

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 30s safety margin before 630s limit
    
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
    
    # Honeycomb 3-4-3 and 4-3-4 topologies
    for spacing in [2.0, 2.2, 2.4, 2.6]:
        initial_configs.append(_make_honeycomb_343(spacing))
        initial_configs.append(_make_honeycomb_434(spacing))
    
    # Random perturbations
    rng = np.random.default_rng(42)
    for seed in range(10):
        rng_local = np.random.default_rng(seed)
        base = _make_hex_lattice(2, 2.5)
        base += rng_local.normal(0, 0.5, size=(12, 2))
        initial_configs.append(base)
    
    # Try different outer angles
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    best_s = np.inf
    best_params = None
    best_angle = 0
    
    # Phase 1: Nelder-Mead from multiple starts with penalty (to avoid overlaps)
    for angle_deg in outer_angles:
        if time.time() > deadline - 180:
            break
        
        for init_pos in initial_configs:
            if time.time() > deadline - 180:
                break
            
            init_params = np.concatenate([init_pos.flatten(), np.array([0.0, 0.0])])
            
            # Nelder-Mead with penalty
            nm_params, nm_val = _nelder_mead_penalty(init_params, angle_deg,
                                                     maxiter=3000, deadline=deadline - 180)
            
            # Check if the solution is valid (no overlaps)
            inner_pos = nm_params[:24].reshape(12, 2)
            oc = nm_params[24:26]
            s_val = _compute_min_side_fast(inner_pos, oc, angle_deg)
            
            # Verify no overlaps
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = np.linalg.norm(inner_pos[i] - inner_pos[j])
                    if dist < 1.99:  # conservative check
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and s_val < best_s:
                best_s = s_val
                best_params = nm_params
                best_angle = angle_deg
    
    # Phase 2: Refine the best solution with plain Nelder-Mead (no penalty)
    if best_params is not None and time.time() < deadline - 60:
        nm_params, nm_val = _nelder_mead(best_params, best_angle,
                                         maxiter=10000, deadline=deadline - 60)
        if nm_val < best_s:
            best_s = nm_val
            best_params = nm_params
    
    # Phase 3: Try angle refinement
    if best_params is not None and time.time() < deadline - 30:
        refined_params, refined_angle, refined_val = _refine_with_angles(
            best_params, best_angle, deadline - 30)
        if refined_val < best_s:
            best_s = refined_val
            best_params = refined_params
            best_angle = refined_angle
    
    # Phase 4: Final fine-tuning with Nelder-Mead on the best
    if best_params is not None and time.time() < deadline - 10:
        nm_params, nm_val = _nelder_mead(best_params, best_angle,
                                         maxiter=20000, deadline=deadline - 10)
        if nm_val < best_s:
            best_s = nm_val
            best_params = nm_params
    
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
        # Fallback to a known valid configuration
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
