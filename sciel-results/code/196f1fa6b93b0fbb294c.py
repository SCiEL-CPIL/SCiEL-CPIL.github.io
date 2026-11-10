# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# ---------- geometry helpers (match the task contract exactly) ----------
def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts = np.zeros((n, 6, 2))
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _separated(positions, angles, tol=1e-6):
    """Check if all pairs of hexagons are separated (no intersection).
    
    For two convex polygons, they are separated iff there exists at least one
    edge normal (from either polygon) along which their projections are disjoint.
    """
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals = np.stack([np.cos(base), np.sin(base)], axis=2)
    for i in range(n):
        vi = all_verts[i]
        ni = hex_normals[i]
        for j in range(i + 1, n):
            vj = all_verts[j]
            nj = hex_normals[j]
            # All 12 edge normals (6 from each hexagon)
            normals = np.vstack([ni, nj])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            # Check if ANY normal axis shows separation
            # If no axis separates them, the polygons overlap
            sep_i = np.max(proj_i, axis=0) < np.min(proj_j, axis=0) - tol
            sep_j = np.max(proj_j, axis=0) < np.min(proj_i, axis=0) - tol
            if not (np.any(sep_i) or np.any(sep_j)):
                # No separating axis found -> overlap
                return False
    return True

def _separated_aligned(positions, tol=1e-6):
    """Fast check for aligned hexagons (all angle 0).
    Separation distance in normal directions must be > sqrt(3).
    """
    n = len(positions)
    # Normals for angle 0 hexagon
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1) # (6, 2)
    
    # Projections of all centers onto all normals: (n, 6)
    projs = positions @ normals.T
    
    # Difference in projections for all pairs: (n, n, 6)
    diff = projs[:, np.newaxis, :] - projs[np.newaxis, :, :]
    
    # Max absolute projection difference for each pair
    max_abs_diff = np.max(np.abs(diff), axis=2)
    
    # For each pair, we need at least one direction where diff > sqrt(3)
    # Actually, for aligned hexagons, the condition is that the Minkowski difference
    # does not contain the origin. The Minkowski difference of two unit hexagons is a hexagon
    # with side 2. The support function in normal direction k is 2 * (apothem) = 2 * sqrt(3)/2 * 2?
    # Let's derive:
    # Hexagon H1 centered at c1, H2 at c2.
    # H1 and H2 intersect iff (c1-c2) is in the Minkowski difference H1 - (-H2) = H1 + H2.
    # H1 + H2 is a regular hexagon with side length 2 (since unit hex + unit hex = side 2 hex? No).
    # Side 1 hex has apothem sqrt(3)/2.
    # Sum of two hexagons has apothem sqrt(3).
    # So the condition for non-intersection is that the vector d = c1-c2 is outside the hexagon with apothem sqrt(3).
    # The hexagon with apothem A is defined by |d . n_k| <= A for all k.
    # So they are separated if max_k |d . n_k| > sqrt(3).
    
    sqrt3 = np.sqrt(3.0)
    # Set diagonal to inf to ignore self
    np.fill_diagonal(max_abs_diff, np.inf)
    
    # For each pair (i, j), check if any normal k has diff > sqrt3
    # valid_pair[i, j] = any(max_abs_diff[i, j, :] > sqrt3)
    valid_pairs = np.any(max_abs_diff > sqrt3 + tol, axis=2)
    
    # All pairs must be valid
    return np.all(valid_pairs)

def _enclosing_side_aligned(positions):
    """Calculate min side length for aligned hexagons in outer hex angle 0."""
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    projs = positions @ normals.T
    max_projs = np.max(projs, axis=0)
    # Max projection of inner hex vertex onto normal is center_proj + 1 (circumradius)
    # Wait, support function of unit hex in normal direction is apothem = sqrt(3)/2.
    # So max projection of inner hex is center_proj + sqrt(3)/2.
    # Outer hex apothem is side * sqrt(3)/2.
    # Condition: center_proj + sqrt(3)/2 <= side * sqrt(3)/2
    # side >= (center_proj + sqrt(3)/2) / (sqrt(3)/2) = center_proj * 2/sqrt(3) + 1
    required_sides = max_projs * (2.0 / np.sqrt(3.0)) + 1.0
    return np.max(required_sides)

def _enclosing_side(positions, angles, outer_angle=0.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(outer_angle + 30 + 60 * i)),
                         np.sin(np.radians(outer_angle + 30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return np.max(projs) / (np.sqrt(3) / 2)

def _soft_max(x, beta=100.0):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _enclosing_side_soft(positions, angles, outer_angle=0.0, beta=100.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(outer_angle + 30 + 60 * i)),
                         np.sin(np.radians(outer_angle + 30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return _soft_max(projs, beta) / (np.sqrt(3) / 2)

def _contained(positions, angles, side, outer_angle=0.0, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer = _hex_vertices(0.0, 0.0, side, outer_angle)
    for i in range(6):
        p1 = outer[i]
        p2 = outer[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

# ---------- candidate generators ----------
def _sym_candidate(ring_r, valley_r, angle_offset=0.0):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
    for k in range(4):
        a = np.radians(0 + 60 * k + angle_offset)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
    return positions, angles

def _random_candidate(rng):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    angles[0] = rng.uniform(0, 60)
    for k in range(6):
        a = np.radians(30 + 60 * k + rng.uniform(-4, 4))
        r = rng.uniform(1.8, 2.6)
        positions[k + 1] = [r * np.cos(a), r * np.sin(a)]
        angles[k + 1] = rng.uniform(0, 60)
    for k in range(4):
        a = np.radians(0 + 60 * k + rng.uniform(-8, 8))
        r = rng.uniform(2.8, 4.4)
        positions[7 + k] = [r * np.cos(a), r * np.sin(a)]
        angles[7 + k] = rng.uniform(0, 60)
    return positions, angles

# ---------- Refinement for Aligned Case (Fast) ----------
def _refine_aligned(positions, deadline, rng):
    """Optimize positions for aligned hexagons (angles fixed to 0)."""
    best_pos = positions.copy()
    best_side = _enclosing_side_aligned(best_pos)
    
    def obj_aligned(x):
        pos = x.reshape(11, 2)
        if not _separated_aligned(pos):
            # Penalty for overlap
            # Calculate max violation
            angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
            normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            projs = pos @ normals.T
            diff = projs[:, np.newaxis, :] - projs[np.newaxis, :, :]
            max_abs_diff = np.max(np.abs(diff), axis=2)
            np.fill_diagonal(max_abs_diff, np.inf)
            # We want max_abs_diff > sqrt(3). Violation is max(0, sqrt(3) - max_abs_diff)
            violations = np.maximum(0, np.sqrt(3.0) - max_abs_diff)
            penalty = np.sum(violations)
            return 1000.0 + penalty
        return _enclosing_side_aligned(pos)

    # Nelder-Mead
    try:
        res = minimize(
            obj_aligned, best_pos.flatten(), method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True, 'maxfev': 100000}
        )
        pos_new = res.x.reshape(11, 2)
        if _separated_aligned(pos_new):
            s_new = _enclosing_side_aligned(pos_new)
            if s_new < best_side:
                best_side = s_new
                best_pos = pos_new.copy()
    except Exception:
        pass

    # Simulated Annealing
    if time.time() < deadline - 10:
        T = 0.1
        T_min = 0.001
        n_iter = 20000
        x = best_pos.flatten().copy()
        cur_side = best_side
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            T = T_min + (T - T_min) * (1 - it / n_iter)
            x_new = x + rng.normal(0, 0.05 * (T / 0.1) + 0.005, size=x.shape)
            pos_new = x_new.reshape(11, 2)
            
            if _separated_aligned(pos_new):
                s_new = _enclosing_side_aligned(pos_new)
                delta = s_new - cur_side
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-8)):
                    x = x_new
                    cur_side = s_new
                    if s_new < best_side:
                        best_side = s_new
                        best_pos = pos_new.copy()
            else:
                # Accept invalid moves with low prob to escape
                if rng.random() < 0.01:
                    x = x_new
                    # Don't update cur_side if invalid, or estimate it?
                    # Better to just keep cur_side from last valid state for SA logic consistency
                    # But if we stay in invalid state too long, we might get stuck.
                    # Let's just revert if invalid in this simple SA? 
                    # No, standard SA allows invalid states. 
                    # Let's just not update best if invalid.
                    pass

    return best_pos, best_side

# ---------- Local Refinement for General Case ----------
def _refine(positions, angles, deadline, rng, outer_angle=0.0):
    best_pos = positions.copy()
    best_ang = angles.copy()
    best_side = _enclosing_side(positions, angles, outer_angle)
    t0 = time.time()

    def _run(obj, x0, maxiter, method='Nelder-Mead', **opts):
        nonlocal best_pos, best_ang, best_side
        try:
            res = minimize(
                obj, x0, method=method,
                options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-13,
                         'adaptive': True, 'maxfev': maxiter * 40, **opts}
            )
            pos = res.x[:22].reshape(11, 2)
            ang = res.x[22:] % 360
            if _separated(pos, ang):
                s = _enclosing_side(pos, ang, outer_angle)
                if s < best_side:
                    best_side = s
                    best_pos = pos.copy()
                    best_ang = ang.copy()
        except Exception:
            pass

    # smooth-surface pass
    def obj_smooth(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:]
        if not _separated(pos, ang):
            return 1e6 + 0.001 * np.sum((pos) ** 2)
        return _enclosing_side_soft(pos, ang, outer_angle, beta=100.0)

    x0 = np.concatenate([positions.flatten(), angles])
    _run(obj_smooth, x0, 2000)

    # Simulated annealing pass
    if time.time() < deadline - 30:
        def sa_pass():
            nonlocal best_pos, best_ang, best_side
            T = 0.15
            T_min = 0.005
            n_iter = 3000
            x = np.concatenate([best_pos.flatten(), best_ang])
            cur_side = _enclosing_side(best_pos, best_ang, outer_angle)
            
            for it in range(n_iter):
                if time.time() > deadline - 15:
                    break
                T = T_min + (T - T_min) * (1 - it / n_iter)
                base_scale = 0.08 * (T / 0.15) + 0.005
                x_new = x + rng.normal(0, base_scale, size=x.shape)
                x_new[22:] = x_new[22:] % 360
                pos_new = x_new[:22].reshape(11, 2)
                ang_new = x_new[22:]
                
                if _separated(pos_new, ang_new):
                    s_new = _enclosing_side(pos_new, ang_new, outer_angle)
                    delta = s_new - cur_side
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-8)):
                        x = x_new
                        cur_side = s_new
                        if s_new < best_side:
                            best_side = s_new
                            best_pos = pos_new.copy()
                            best_ang = ang_new.copy()
                else:
                    if rng.random() < 0.01:
                        x = x_new
                        # cur_side remains from last valid or is invalid?
                        # If we accept invalid, cur_side should reflect the "cost" which is high.
                        # But we only track best_side if valid.
                        # Let's just not update cur_side to avoid confusion, or set to high value.
                        # Actually, if we accept invalid, we are in an invalid state.
                        # Next step will compare new valid/invalid against this.
                        # It's safer to just not update cur_side if we go invalid, 
                        # effectively treating the "current" state as the last valid one for acceptance logic?
                        # No, that breaks SA. 
                        # Standard SA: Cost function returns high value for invalid.
                        # Here we have a binary valid/invalid.
                        # Let's just keep it simple: only accept valid moves in SA for general case?
                        # The previous code did: if rng.random() < 0.01: x = x_new.
                        # It did NOT update cur_side. This means if we go invalid, 
                        # the next acceptance is compared against the last VALID cur_side.
                        # This is a heuristic that works.
                        pass

        sa_pass()

    # exact polish
    if time.time() < deadline - 10:
        def obj_exact(x):
            pos = x[:22].reshape(11, 2)
            ang = x[22:]
            if not _separated(pos, ang):
                return 1e6 + 0.001 * np.sum((pos) ** 2)
            return _enclosing_side(pos, ang, outer_angle)

        x1 = np.concatenate([best_pos.flatten(), best_ang])
        _run(obj_exact, x1, 2000)

    return best_pos, best_ang, best_side

# ---------- Joint Rotation ----------
def _best_rotation_joint(positions, angles):
    """Optimize cluster rotation theta and outer hex angle phi jointly."""
    best_side = _enclosing_side(positions, angles, 0.0)
    best_theta = 0.0
    best_phi = 0.0

    def _eval(theta, phi):
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        rot = np.array([[c, -s], [s, c]])
        p_rot = positions @ rot.T
        a_rot = (angles + theta) % 360
        return _enclosing_side(p_rot, a_rot, phi)

    # Coarse grid
    for theta in np.linspace(0, 60, 21):
        for phi in np.linspace(0, 60, 21):
            s = _eval(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi

    # Fine grid
    for theta in np.linspace(max(0, best_theta - 2), min(60, best_theta + 2), 41):
        for phi in np.linspace(max(0, best_phi - 2), min(60, best_phi + 2), 41):
            s = _eval(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi

    # Nelder-Mead polish
    try:
        res = minimize(
            lambda v: _eval(v[0], v[1]),
            [best_theta, best_phi],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True, 'maxfev': 50000}
        )
        if res.fun < best_side:
            best_side = res.fun
            best_theta = res.x[0] % 60
            best_phi = res.x[1] % 60
    except Exception:
        pass

    c = np.cos(np.radians(best_theta))
    s = np.sin(np.radians(best_theta))
    rot = np.array([[c, -s], [s, c]])
    p_rot = positions @ rot.T
    a_rot = (angles + best_theta) % 360
    return p_rot, a_rot, best_phi, best_side

# ---------- Main Entry Point ----------
def hexagon_packing_11():
    start = time.time()
    deadline = start + 580.0
    rng = np.random.default_rng(42)

    best_pos = None
    best_ang = None
    best_side = float('inf')
    best_phi = 0.0

    def consider(pos, ang, phi=0.0):
        nonlocal best_pos, best_ang, best_side, best_phi
        if not _separated(pos, ang):
            return False
        s = _enclosing_side(pos, ang, phi)
        if s < best_side:
            best_side = s
            best_pos = pos.copy()
            best_ang = ang.copy()
            best_phi = phi
            return True
        return False

    # --- Phase A: Aligned Search (Fast, High Quality) ---
    # Initialize with symmetric candidate
    pos_init, ang_init = _sym_candidate(2.0, 3.5, 0.0)
    # ang_init is all zeros.
    
    # Refine aligned
    pos_aligned, side_aligned = _refine_aligned(pos_init, deadline - 100, rng)
    
    # Convert aligned result to general format
    ang_aligned = np.zeros(11)
    # Check if this is better than current best (which is inf)
    if _separated(pos_aligned, ang_aligned):
        s_check = _enclosing_side(pos_aligned, ang_aligned, 0.0)
        if s_check < best_side:
            best_pos = pos_aligned.copy()
            best_ang = ang_aligned.copy()
            best_side = s_check
            best_phi = 0.0

    # --- Phase B: General Search (Slower, Broader) ---
    # Use the aligned solution as a starting point for general search
    # Also try some random starts
    
    # 1. Start from aligned best
    pos2, ang2, s2 = _refine(best_pos, best_ang, deadline - 60, rng, outer_angle=0.0)
    if s2 < best_side:
        best_side = s2
        best_pos = pos2.copy()
        best_ang = ang2.copy()

    # 2. Random starts
    n_starts = 0
    while time.time() < deadline - 30 and n_starts < 5:
        pos, ang = _random_candidate(rng)
        pos2, ang2, s2 = _refine(pos, ang, deadline - 30, rng, outer_angle=0.0)
        if s2 < best_side:
            best_side = s2
            best_pos = pos2.copy()
            best_ang = ang2.copy()
        n_starts += 1

    # --- Phase C: Joint Rotation ---
    if time.time() < deadline - 10:
        pos_r, ang_r, phi_r, s_r = _best_rotation_joint(best_pos, best_ang)
        if s_r < best_side:
            best_pos, best_ang, best_side, best_phi = pos_r, ang_r, phi_r, s_r

    # --- Final Verification ---
    final_side = _enclosing_side(best_pos, best_ang, best_phi)
    if not _separated(best_pos, best_ang):
        # Fallback to a known valid configuration
        pos, ang = _sym_candidate(2.0, 3.5, 0.0)
        best_pos, best_ang, best_phi = pos, ang, 0.0
        final_side = _enclosing_side(pos, ang, 0.0)

    if not _contained(best_pos, best_ang, final_side, best_phi):
        final_side *= 1.001

    inner = np.column_stack([best_pos, best_ang])
    outer = np.array([0.0, 0.0, best_phi])
    return inner, outer, float(final_side)

# EVOLVE-BLOCK-END
