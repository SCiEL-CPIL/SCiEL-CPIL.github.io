# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_pair_separation(hex1, hex2, tol=1e-6):
    """
    Check if two unit hexagons are separated (non-intersecting).
    Returns True if they are separated on at least one of the 12 edge normals.
    """
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])

    # Get all 12 edge normals (6 from each hexagon)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
                normals.append(n)

    # Check each normal
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)

        # Check if intervals are apart by more than tol
        if max1 < min2 - tol or max2 < min1 - tol:
            return True

    return False


def _check_inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon.
    For each inner vertex, check cross product with each outer edge.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)

    for hex_data in inner_hexes:
        inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])

        for iv in inner_verts:
            # Shift by outer center
            iv_shifted = iv - outer_center

            for i in range(6):
                p1 = outer_verts[i] - outer_center
                p2 = outer_verts[(i + 1) % 6] - outer_center

                # Cross product: (p2 - p1) x (p - p1)
                edge = p2 - p1
                vec = iv_shifted - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]

                if cross < -tol:
                    return False

    return True


def _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle):
    """
    Find the minimum outer side length such that all inner hexagons fit inside.
    """
    min_side = 0.0

    for i in range(6):
        angle = np.radians(60 * i + 30)
        n = np.array([np.cos(angle), np.sin(angle)])

        max_proj = 0.0
        for hex_data in inner_hexes:
            inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
            inner_verts_shifted = inner_verts - outer_center
            projs = inner_verts_shifted @ n
            max_proj = max(max_proj, np.max(projs))

        required_side = 2.0 * max_proj / np.sqrt(3.0)
        min_side = max(min_side, required_side)

    return min_side


def _soft_max_enclosing_side(inner_hexes, outer_center, outer_angle, beta=50.0):
    """
    Smooth approximation of the minimum enclosing side using log-sum-exp.
    """
    max_projs = np.zeros(6)

    for i in range(6):
        angle = np.radians(60 * i + 30)
        n = np.array([np.cos(angle), np.sin(angle)])

        max_proj = 0.0
        for hex_data in inner_hexes:
            inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
            inner_verts_shifted = inner_verts - outer_center
            projs = inner_verts_shifted @ n
            max_proj = max(max_proj, np.max(projs))

        max_projs[i] = max_proj

    scaled = beta * max_projs
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum_exp / beta

    return 2.0 * smooth_max / np.sqrt(3.0)


def _objective_smooth(params, outer_center, outer_angle, beta=50.0):
    """
    Objective function with smooth enclosing side and penalty for overlaps.
    """
    inner_hexes = params.reshape(11, 3)

    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                penalty += 10.0

    side = _soft_max_enclosing_side(inner_hexes, outer_center, outer_angle, beta)

    return side + penalty


def _objective_exact(params, outer_center, outer_angle):
    """
    Exact objective for final polish.
    """
    inner_hexes = params.reshape(11, 3)

    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                return 1e6

    side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
    return side


def _initial_guesses():
    """Generate multiple initial guesses for the packing."""
    sqrt3 = np.sqrt(3)

    guesses = []

    # Guess 1: Standard hexagonal arrangement (center + 6 + 4 corners)
    g1 = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [-3, 0, 0],
        [1.5, 2.598, 0],
        [-1.5, 2.598, 0],
        [1.5, -2.598, 0],
        [-1.5, -2.598, 0],
        [4.5, 2.598, 0],
        [-4.5, 2.598, 0],
        [4.5, -2.598, 0],
        [-4.5, -2.598, 0],
    ]).flatten()
    guesses.append(g1)

    # Guess 2: Slightly compressed arrangement
    g2 = g1.copy()
    for i in range(11):
        x, y, a = g2[3*i], g2[3*i+1], g2[3*i+2]
        r = np.sqrt(x*x + y*y)
        if r > 0.1:
            scale = 0.95
            g2[3*i] = x * scale
            g2[3*i+1] = y * scale
    guesses.append(g2)

    # Guess 3: Different angular arrangement
    g3 = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [-3, 0, 0],
        [1.5, 2.598, 0],
        [-1.5, 2.598, 0],
        [1.5, -2.598, 0],
        [-1.5, -2.598, 0],
        [3.0, 1.5, 0],
        [-3.0, 1.5, 0],
        [3.0, -1.5, 0],
        [-3.0, -1.5, 0],
    ]).flatten()
    guesses.append(g3)

    # Guess 4: Rotated outer arrangement
    g4 = g1.copy()
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(11):
        x, y = g4[3*i], g4[3*i+1]
        g4[3*i] = x * c - y * s
        g4[3*i+1] = x * s + y * c
    guesses.append(g4)

    # Graph-based guesses: central + ring1(6) + ring2(4) with tangency constraints
    # For unit hexagons, adjacent centers are sqrt(3) apart
    # But for a more compact packing, we can use different arrangements

    # Guess 5: Tighter hexagonal lattice
    apothem = np.sqrt(3) / 2
    center_dist = 2 * apothem  # sqrt(3)

    # Hexagonal lattice positions
    positions = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (2, 0),
        (0, 2),
    ]

    g5 = np.zeros(33)
    for i, (q, r) in enumerate(positions):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        g5[3*i] = x
        g5[3*i+1] = y
        g5[3*i+2] = 0
    guesses.append(g5)

    # Guess 6: Alternative lattice
    positions6 = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (0, 2),
        (0, -2),
    ]

    g6 = np.zeros(33)
    for i, (q, r) in enumerate(positions6):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        g6[3*i] = x
        g6[3*i+1] = y
        g6[3*i+2] = 0
    guesses.append(g6)

    # Guess 7: Scaled and rotated versions
    for scale in [0.9, 1.1]:
        g7 = g1.copy()
        for i in range(11):
            g7[3*i] *= scale
            g7[3*i+1] *= scale
        guesses.append(g7)

    return guesses


def _verify_solution(inner_hexes, outer_center, outer_side, outer_angle):
    """Verify that a solution is valid."""
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                return False

    if not _check_inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle):
        return False

    if outer_side < np.sqrt(11) - 1e-9:
        return False

    if not np.all(np.isfinite(inner_hexes)):
        return False

    return True


def _rigid_body_dynamics(params, outer_center, outer_angle, dt=0.001, iterations=2000, time_limit=30):
    """
    Rigid-body dynamics simulation with repulsive forces to find dense packing.
    """
    start_time = time.time()
    positions = params.reshape(11, 3).copy()
    velocities = np.zeros((11, 2))
    
    best_positions = positions.copy()
    best_side = _min_outer_side_for_inner(positions, outer_center, outer_angle)
    best_valid = _verify_solution(positions, outer_center, best_side, outer_angle)
    
    stiffness = 50.0
    damping = 0.95
    center_pull = 0.1
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros((11, 2))
        
        # Repulsive forces between all pairs
        for i in range(11):
            for j in range(i + 1, 11):
                di = positions[i, :2]
                dj = positions[j, :2]
                diff = dj - di
                dist = np.linalg.norm(diff)
                
                if dist < 1e-10:
                    # Random perturbation to avoid singularity
                    diff = np.random.randn(2) * 1e-3
                    dist = np.linalg.norm(diff)
                
                # Check if hexagons are overlapping
                hex_i = positions[i]
                hex_j = positions[j]
                if not _check_pair_separation(hex_i, hex_j):
                    # Strong repulsive force
                    force_mag = stiffness * (3.0 - dist)
                    force_dir = diff / dist
                    forces[i] -= force_mag * force_dir
                    forces[j] += force_mag * force_dir
        
        # Center pull force
        for i in range(11):
            forces[i] -= center_pull * positions[i, :2]
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        positions[:, :2] += velocities * dt
        
        # Check if this configuration is better
        side = _min_outer_side_for_inner(positions, outer_center, outer_angle)
        if _verify_solution(positions, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_valid = True
        
        # Reduce stiffness over time for settling
        stiffness *= 0.999
        center_pull *= 0.9995
    
    return best_positions.flatten(), best_side


def _local_search(params, outer_center, outer_angle, step_size=0.01, max_iter=200, time_limit=30):
    """
    Local search using small random perturbations.
    """
    start_time = time.time()
    best_params = params.copy()
    best_side = _min_outer_side_for_inner(params.reshape(11, 3), outer_center, outer_angle)
    best_valid = _verify_solution(params.reshape(11, 3), outer_center, best_side, outer_angle)

    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        # Random perturbation
        new_params = params + np.random.normal(0, step_size, size=params.shape)

        inner_hexes = new_params.reshape(11, 3)
        side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)

        if _verify_solution(inner_hexes, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_params = new_params.copy()
                best_valid = True
            params = new_params.copy()
        else:
            # Try smaller step
            params = 0.5 * params + 0.5 * best_params

    return best_params, best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600

    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0

    # Get initial guesses
    guesses = _initial_guesses()

    best_params = None
    best_side = float('inf')

    # Phase 1: Rigid-body dynamics from multiple starting points
    for gi, guess in enumerate(guesses[:4]):
        if time.time() > deadline - 120:
            break

        params0 = guess.copy()
        try:
            dyn_params, dyn_side = _rigid_body_dynamics(
                params0, outer_center, outer_angle,
                dt=0.001, iterations=1500, time_limit=25
            )
            
            if dyn_side < best_side:
                inner_hexes = dyn_params.reshape(11, 3)
                side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                if _verify_solution(inner_hexes, outer_center, side, outer_angle):
                    best_side = side
                    best_params = dyn_params.copy()
        except Exception:
            continue

    # Phase 2: Smooth optimization from multiple starting points
    for gi, guess in enumerate(guesses):
        if time.time() > deadline - 90:
            break

        params0 = guess.copy()

        try:
            result = minimize(
                _objective_smooth,
                params0,
                args=(outer_center, outer_angle, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )

            if result.fun < best_side:
                inner_hexes = result.x.reshape(11, 3)

                side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                if _verify_solution(inner_hexes, outer_center, side, outer_angle):
                    best_side = side
                    best_params = result.x.copy()

        except Exception:
            continue

    # Phase 3: Exact polish from the best smooth solution
    if best_params is not None and time.time() < deadline - 60:
        try:
            result = minimize(
                _objective_exact,
                best_params,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7}
            )

            if result.fun < best_side:
                inner_hexes = result.x.reshape(11, 3)
                side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                if _verify_solution(inner_hexes, outer_center, side, outer_angle):
                    best_side = side
                    best_params = result.x.copy()
        except Exception:
            pass

    # Phase 4: Local search refinement
    if best_params is not None and time.time() < deadline - 30:
        try:
            best_params, best_side = _local_search(
                best_params, outer_center, outer_angle,
                step_size=0.005, max_iter=300, time_limit=25
            )
        except Exception:
            pass

    # Phase 5: Final fine-tuning with very tight tolerances
    if best_params is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                _objective_exact,
                best_params,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )

            if result.fun < best_side:
                inner_hexes = result.x.reshape(11, 3)
                side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                if _verify_solution(inner_hexes, outer_center, side, outer_angle):
                    best_side = side
                    best_params = result.x.copy()
        except Exception:
            pass

    # If no valid solution found, use the best guess
    if best_params is None:
        best_params = guesses[0].copy()
        best_side = _min_outer_side_for_inner(best_params.reshape(11, 3), outer_center, outer_angle)

    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_side)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
