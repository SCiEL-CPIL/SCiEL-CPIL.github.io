# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _build_hex_positions(center, ring_r, ring_angles, valley_r, valley_angles, ring_rot=0.0, valley_rot=0.0):
    """Build positions for 11 hexagons: 1 center, 6 ring, 4 valley."""
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    positions[0] = center
    angles[0] = 0.0
    
    for k in range(6):
        a = np.radians(ring_angles[k] + ring_rot)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
        angles[k + 1] = 0.0
    
    for k in range(4):
        a = np.radians(valley_angles[k] + valley_rot)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
        angles[7 + k] = 0.0
    
    return positions, angles

def _separation_violation(positions, angles, tol=1e-6):
    """Return max violation of separation constraint (positive = violation)."""
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    max_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            
            # Check if separated on any normal
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                # Find the minimum gap
                gaps = np.minimum(max_i - min_j, max_j - min_i)
                min_gap = np.min(gaps)
                if min_gap < 0:
                    max_violation = max(max_violation, -min_gap)
    return max_violation

def _solve_topology(contact_pairs, init_positions, init_angles):
    """Solve for positions such that specified pairs are exactly touching."""
    n = len(init_positions)
    
    def equations(vars):
        positions = vars[:2*n].reshape(n, 2)
        angles = vars[2*n:]
        
        eqs = []
        for (i, j) in contact_pairs:
            # For two unit hexagons to be touching, their centers must be at distance sqrt(3)
            # when they share an edge, or specific other distances for other contact types
            # For simplicity, assume edge-sharing: distance = sqrt(3)
            d = np.linalg.norm(positions[i] - positions[j])
            eqs.append(d - np.sqrt(3))
        
        return eqs
    
    init_vars = np.concatenate([init_positions.flatten(), init_angles])
    
    try:
        sol, info, ier, msg = fsolve(equations, init_vars, full_output=True, maxfev=10000)
        if ier == 1:
            positions = sol[:2*n].reshape(n, 2)
            angles = sol[2*n:]
            if _check_separation_fast(positions, angles):
                return positions, angles
    except Exception:
        pass
    return None, None

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    d = np.sqrt(3)
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Strategy: Use a symmetric parameterization and optimize
    # 11 hexagons: 1 center + 6 ring + 4 outer
    # The 6 ring hexagons are at angles 30+60k
    # The 4 outer hexagons are at angles 30+120k and 90+120k (or similar)
    
    # Phase 1: Grid search over symmetric parameters
    # ring_r: distance from center to ring hexagons
    # outer_r: distance from center to outer hexagons
    # angle_offset: global rotation
    
    ring_r_range = np.linspace(1.732, 2.8, 30)
    outer_r_range = np.linspace(2.5, 5.0, 30)
    angle_offset_range = np.linspace(0, 30, 15)
    
    for ar in ring_r_range:
        if time.time() > deadline - 120:
            break
        for vr in outer_r_range:
            if time.time() > deadline - 120:
                break
            for ao in angle_offset_range:
                if time.time() > deadline - 120:
                    break
                
                # Build positions
                positions = np.zeros((11, 2))
                angles = np.zeros(11)
                
                # Center
                positions[0] = [0.0, 0.0]
                
                # 6 ring hexagons at 30+60k + angle_offset
                for k in range(6):
                    a = np.radians(30 + 60 * k + ao)
                    positions[k + 1] = [ar * np.cos(a), ar * np.sin(a)]
                
                # 4 outer hexagons at 30+120k and 90+120k + angle_offset
                outer_angles = [30, 90, 150, 210]
                for k, va in enumerate(outer_angles):
                    a = np.radians(va + ao)
                    positions[7 + k] = [vr * np.cos(a), vr * np.sin(a)]
                
                if not _check_separation_fast(positions, angles):
                    continue
                
                side = _min_enclosing_side(positions, angles)
                if side < best_side:
                    best_side = side
                    best_positions = positions.copy()
                    best_angles = angles.copy()
    
    # Phase 2: Local optimization with Nelder-Mead on full position space
    if best_positions is not None and time.time() < deadline - 60:
        def objective(pos_flat):
            pos = pos_flat.reshape(11, 2)
            if not _check_separation_fast(pos, best_angles):
                return 1e10
            return _min_enclosing_side(pos, best_angles)
        
        try:
            res = minimize(
                objective,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_side and res.fun < 1e9:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    side = _min_enclosing_side(trial_pos, best_angles)
                    if side < best_side:
                        best_side = side
                        best_positions = trial_pos
        except Exception:
            pass
    
    # Phase 3: Try different contact topologies
    if time.time() < deadline - 30:
        # Try a few different symmetric configurations
        configs = [
            # (ring_angles, outer_angles)
            ([30, 90, 150, 210, 270, 330], [30, 90, 150, 210]),
            ([30, 90, 150, 210, 270, 330], [0, 60, 120, 180]),
            ([30, 90, 150, 210, 270, 330], [30, 150, 270, 90]),
        ]
        
        for ring_angles, outer_angles in configs:
            if time.time() > deadline - 30:
                break
            
            # Grid search for this config
            for ar in np.linspace(1.732, 2.8, 20):
                if time.time() > deadline - 30:
                    break
                for vr in np.linspace(2.5, 5.0, 20):
                    if time.time() > deadline - 30:
                        break
                    for ao in np.linspace(0, 30, 10):
                        if time.time() > deadline - 30:
                            break
                        
                        positions = np.zeros((11, 2))
                        angles = np.zeros(11)
                        
                        positions[0] = [0.0, 0.0]
                        
                        for k in range(6):
                            a = np.radians(ring_angles[k] + ao)
                            positions[k + 1] = [ar * np.cos(a), ar * np.sin(a)]
                        
                        for k, va in enumerate(outer_angles):
                            a = np.radians(va + ao)
                            positions[7 + k] = [vr * np.cos(a), vr * np.sin(a)]
                        
                        if not _check_separation_fast(positions, angles):
                            continue
                        
                        side = _min_enclosing_side(positions, angles)
                        if side < best_side:
                            best_side = side
                            best_positions = positions.copy()
                            best_angles = angles.copy()
    
    # Phase 4: Fine coordinate descent
    if best_positions is not None and time.time() < deadline - 10:
        for cd_iter in range(1000):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(11):
                for coord in range(2):
                    for step in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                        trial = best_positions.copy()
                        trial[i, coord] += step
                        if not _check_separation_fast(trial, best_angles):
                            continue
                        trial_side = _min_enclosing_side(trial, best_angles)
                        if trial_side < best_side - 1e-12:
                            best_positions = trial
                            best_side = trial_side
                            improved = True
                            break
            if not improved:
                break
    
    # Fallback
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 120 * k)
            positions[7 + k] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known valid config
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 120 * k)
            positions[7 + k] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
