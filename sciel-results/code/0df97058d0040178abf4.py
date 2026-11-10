# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _hex_overlap_penalty(pos_i, ang_i, pos_j, ang_j, stiffness=1e4):
    """Compute repulsive force magnitude if two hexagons overlap.
    Uses minimum distance between vertices/edges as proxy."""
    vi = _hex_vertices(pos_i[0], pos_i[1], 1.0, ang_i)
    vj = _hex_vertices(pos_j[0], pos_j[1], 1.0, ang_j)
    
    # Check all vertex-edge distances for a quick overlap indicator
    # If any vertex of i is inside j or vice versa, they overlap
    min_dist = np.inf
    for k in range(6):
        for m in range(6):
            d = np.linalg.norm(vi[k] - vj[m])
            if d < min_dist:
                min_dist = d
    
    # For touching hexagons, min_dist should be around 0 (shared edge midpoint)
    # For separated, min_dist > 0
    # This is a rough proxy - use it for force direction
    if min_dist < 1e-3:
        # Compute separation direction
        sep = pos_i - pos_j
        dist = np.linalg.norm(sep)
        if dist < 1e-10:
            sep = np.array([1.0, 0.0])
            dist = 1.0
        sep = sep / dist
        # Stiffness scaled by how much they "overlap" (penalty for being too close)
        # Unit hexagons have inradius sqrt(3)/2, so centers should be >= sqrt(3) apart for non-overlap
        # But for rotated hexagons it's complex. Use a soft penalty for distance < 2*inradius
        target_dist = 2 * np.sqrt(3) / 2 * 1.5  # conservative
        if dist < target_dist:
            force_mag = stiffness * (target_dist - dist)
            return force_mag * sep
    return np.zeros(2)

def _physics_settle(initial_positions, initial_angles, max_iters=5000, dt=0.001, 
                     spring_k=50.0, damping=0.95, stiffness=1e4, deadline=None):
    """Rigid-body dynamics: hexagons repel each other and are pulled toward center."""
    n = 11
    positions = initial_positions.astype(np.float64).copy()
    angles = initial_angles.astype(np.float64).copy()
    velocities = np.zeros((n, 2))
    ang_velocities = np.zeros(n)
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(positions, angles)
    best_valid = _check_separation_fast(positions, angles)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros((n, 2))
        torques = np.zeros(n)
        
        # Spring force toward center
        for i in range(n):
            forces[i] = -spring_k * positions[i]
        
        # Pairwise repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                sep = positions[i] - positions[j]
                dist = np.linalg.norm(sep)
                if dist < 1e-10:
                    continue
                sep_unit = sep / dist
                
                # Check for overlap using vertex distances
                vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                
                # Find minimum distance between any vertex of i and any edge of j, and vice versa
                # Simplified: check if any vertex is "too close"
                min_vd = np.inf
                for k in range(6):
                    for m in range(6):
                        d = np.linalg.norm(vi[k] - vj[m])
                        if d < min_vd:
                            min_vd = d
                
                # Hexagons with side 1 have width 2, so centers at distance 2 with aligned edges just touch
                # For safety, repel when center distance < 2 * apothem = sqrt(3) ~ 1.732 for same orientation
                # But with rotation, effective width varies. Use a conservative threshold.
                threshold = 1.8  # slightly more than sqrt(3)
                
                if dist < threshold:
                    # Strong repulsion
                    overlap = threshold - dist
                    force_mag = stiffness * overlap
                    forces[i] += force_mag * sep_unit
                    forces[j] -= force_mag * sep_unit
                    
                    # Torque to align edges (encourage flat-edge contact)
                    # Compute angle difference
                    ang_diff = angles[i] - angles[j]
                    # Reduce to [-60, 60]
                    ang_diff = ((ang_diff + 180) % 120) - 60
                    torque = 10.0 * np.sin(np.radians(ang_diff))
                    torques[i] += torque
                    torques[j] -= torque
        
        # Update velocities with damping
        velocities = velocities * damping + forces * dt
        ang_velocities = ang_velocities * damping + torques * dt
        
        # Update positions
        positions = positions + velocities * dt
        angles = (angles + ang_velocities * dt) % 360
        
        # Check validity and track best
        if _check_separation_fast(positions, angles):
            side = _min_enclosing_side(positions, angles)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()
                best_valid = True
        else:
            best_valid = False
    
    return best_positions, best_angles, best_side, best_valid

def _build_dense_cluster(radius_scale=1.0):
    """Build initial dense hexagonal cluster configuration."""
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    # Center hexagon
    positions[0] = [0.0, 0.0]
    angles[0] = 0.0
    
    # 6 surrounding hexagons in a ring
    for k in range(6):
        a = np.radians(60 * k)
        r = 2 * radius_scale
        positions[k + 1] = [r * np.cos(a), r * np.sin(a)]
        angles[k + 1] = 0.0
    
    # 4 additional hexagons in outer positions
    outer_angles = [30, 90, 150, 210]
    for k, a_deg in enumerate(outer_angles):
        a = np.radians(a_deg)
        r = 3.2 * radius_scale
        positions[7 + k] = [r * np.cos(a), r * np.sin(a)]
        angles[7 + k] = 0.0
    
    return positions, angles

def _refine_with_nm(positions, angles, deadline):
    """Final Nelder-Mead refinement of the best physics solution."""
    from scipy.optimize import minimize
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(positions, angles)
    
    def obj(pos_flat):
        pos = pos_flat.reshape(11, 2)
        if not _check_separation_fast(pos, best_angles):
            return 1e10
        return _min_enclosing_side(pos, best_angles)
    
    try:
        res = minimize(
            obj,
            best_positions.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
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
    
    return best_positions, best_angles, best_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 610.0  # 20s safety margin
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Phase 1: Physics-based settling with multiple random initializations
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline - 120:
            break
        
        # Vary the initial configuration
        radius_scale = 1.0 + 0.1 * trial
        positions_init, angles_init = _build_dense_cluster(radius_scale)
        
        # Add small random perturbations for different trials
        if trial > 0:
            positions_init += np.random.randn(11, 2) * 0.1
            angles_init += np.random.randn(11) * 5.0
            angles_init = angles_init % 360
        
        # Run physics simulation
        pos_settled, ang_settled, side_settled, valid = _physics_settle(
            positions_init, angles_init,
            max_iters=3000, dt=0.0005,
            spring_k=30.0, damping=0.93,
            stiffness=5e3,
            deadline=deadline - 60
        )
        
        if valid and side_settled < best_side:
            best_side = side_settled
            best_positions = pos_settled
            best_angles = ang_settled
    
    # Fallback: use a known-good configuration if physics didn't find anything
    if best_positions is None or not _check_separation_fast(best_positions, best_angles):
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 30:
        best_positions, best_angles, best_side = _refine_with_nm(
            best_positions, best_angles, deadline
        )
    
    # Phase 3: Global rotation optimization
    if time.time() < deadline - 15:
        from scipy.optimize import minimize
        
        def obj_rot(theta):
            c = np.cos(np.radians(theta))
            s = np.sin(np.radians(theta))
            rot_mat = np.array([[c, -s], [s, c]])
            pos_rot = best_positions @ rot_mat.T
            angles_rot = (best_angles + theta) % 360
            if not _check_separation_fast(pos_rot, angles_rot):
                return 1e10
            return _min_enclosing_side(pos_rot, angles_rot)
        
        best_rot = 0.0
        best_rot_side = best_side
        for t in np.linspace(0, 60, 121):
            s_t = obj_rot(t)
            if s_t < best_rot_side:
                best_rot_side = s_t
                best_rot = t
        
        try:
            res_rot = minimize(obj_rot, [best_rot], method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-11})
            if res_rot.fun < best_rot_side:
                best_rot = res_rot.x[0]
                best_rot_side = res_rot.fun
        except Exception:
            pass
        
        if best_rot_side < best_side:
            best_side = best_rot_side
            c = np.cos(np.radians(best_rot))
            s = np.sin(np.radians(best_rot))
            rot_mat = np.array([[c, -s], [s, c]])
            best_positions = best_positions @ rot_mat.T
            best_angles = (best_angles + best_rot) % 360
    
    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
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
