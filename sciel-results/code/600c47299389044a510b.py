# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals_fast(a_deg):
    """Return 6 outward unit edge normals for a unit hexagon with angle a_deg (no vertices needed)."""
    a = np.deg2rad(a_deg)
    # Edge normals for a regular hexagon are at angles a + 30 + 60*i degrees
    normal_angles = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])


def _hex_projections(cx, cy, a_deg):
    """
    For a unit hexagon at (cx,cy) with angle a_deg, return the 6 vertex positions
    and the min/max projections along each of the 6 edge normals.
    Returns: vertices (6,2), proj_mins (6,), proj_maxs (6,), normals (6,2)
    """
    verts = _hex_vertices(cx, cy, 1.0, a_deg)
    normals = _edge_normals_fast(a_deg)
    # Project vertices onto each normal
    projs = verts @ normals.T  # (6, 6): vertex i projected onto normal j
    proj_mins = projs.min(axis=0)
    proj_maxs = projs.max(axis=0)
    return verts, proj_mins, proj_maxs, normals


def _separated_fast(p1, pm1, px1, n1, p2, pm2, px2, n2):
    """
    Check if two hexagons are separated using precomputed projections.
    p1,pm1,px1,n1: for hex 1; p2,pm2,px2,n2: for hex 2.
    Returns True if separated by > 1e-6 on at least one of 12 normals.
    """
    # Combine normals (12, 2)
    all_normals = np.vstack([n1, n2])
    # Project vertices of each hex onto all 12 normals
    proj1 = p1 @ all_normals.T  # (6, 12)
    proj2 = p2 @ all_normals.T  # (6, 12)
    # For each of 12 normals, check separation
    max1 = proj1.max(axis=0)  # (12,)
    min1 = proj1.min(axis=0)
    max2 = proj2.max(axis=0)
    min2 = proj2.min(axis=0)
    # Separated if max1 < min2 - eps OR max2 < min1 - eps
    sep = (max1 < min2 - 1e-6) | (max2 < min1 - 1e-6)
    return bool(np.any(sep))


def _all_separated_fast(hex_data):
    """Check all pairs are separated. hex_data is list of (verts, pmin, pmax, normals)."""
    n = len(hex_data)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated_fast(hex_data[i][0], hex_data[i][1], hex_data[i][2], hex_data[i][3],
                                   hex_data[j][0], hex_data[j][1], hex_data[j][2], hex_data[j][3]):
                return False
    return True


def _min_outer_side_fast(all_vertices):
    """
    Find minimum outer hex side length and angle.
    all_vertices: array of shape (N, 2) with all vertices of all inner hexagons.
    Returns (S, angle_deg).
    """
    # For a regular hexagon centered at origin with side S and angle a,
    # the constraint is that for each edge, all points satisfy the half-plane.
    # Edge i goes from vertex i to vertex i+1. The inward normal condition:
    # cross(edge_vec, p - p1) >= 0 for all p.
    # 
    # For a regular hexagon with vertices at angles a + 60*i:
    # Vertex i: S * (cos(a+60i), sin(a+60i))
    # Edge from vertex i to vertex i+1: direction = 60 degrees from radius
    # The supporting line for edge i has normal pointing inward at angle a + 60i + 30
    # Actually, let's think more carefully.
    # 
    # For a regular hexagon centered at origin with "radius" R (= side length S for regular hex):
    # Vertices at angles theta_i = a + 60*i, position = S*(cos(theta_i), sin(theta_i))
    # Edge i connects vertex i to vertex i+1.
    # The midpoint of edge i is at distance S*cos(30) = S*sqrt(3)/2 from center, at angle a+60i+30.
    # The outward normal of edge i points at angle a+60i+30.
    # The half-plane constraint: dot(p, outward_normal_i) <= S*sqrt(3)/2
    # i.e., dot(p, n_i) <= S * sqrt(3)/2 where n_i = (cos(a+60i+30), sin(a+60i+30))
    #
    # So S >= max_i( dot(p, n_i) / (sqrt(3)/2) ) for all p and all i.
    # S >= max_i( 2/sqrt(3) * max_p( dot(p, n_i) ) )
    
    # We need to find angle a that minimizes S(a) = max_{i=0..5} (2/sqrt(3) * max_p dot(p, n_i(a)))
    # where n_i(a) = (cos(a + 30 + 60i), sin(a + 30 + 60i))
    
    # For a given angle a, compute required S:
    def _required_S(a_deg):
        a = np.deg2rad(a_deg)
        normal_angles = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        # For each normal, find max projection of all vertices
        projs = all_vertices @ normals.T  # (N, 6)
        max_projs = projs.max(axis=0)  # (6,)
        return np.max(max_projs) * 2.0 / np.sqrt(3.0)
    
    # Minimize over angle a in [0, 60)
    # The function is periodic with period 60 degrees
    # Use golden section or simple scan + refine
    
    # Coarse scan
    best_S = np.inf
    best_a = 0.0
    for a_deg in np.arange(0, 60, 1.0):
        S = _required_S(a_deg)
        if S < best_S:
            best_S = S
            best_a = a_deg
    
    # Fine scan around best
    for a_deg in np.arange(best_a - 2, best_a + 2, 0.1):
        a_deg = a_deg % 60
        S = _required_S(a_deg)
        if S < best_S:
            best_S = S
            best_a = a_deg
    
    # Even finer
    for a_deg in np.arange(best_a - 0.2, best_a + 0.2, 0.01):
        a_deg = a_deg % 60
        S = _required_S(a_deg)
        if S < best_S:
            best_S = S
            best_a = a_deg
    
    # Verify with the cross product check
    def _verify(S, a_deg):
        outer_verts = _hex_vertices(0, 0, S, a_deg)
        for p in all_vertices:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
        return True
    
    # If verification fails, increase S slightly
    if not _verify(best_S, best_a):
        # Binary search for the correct S
        S_low = best_S
        S_high = best_S * 1.5
        for _ in range(40):
            S_mid = (S_low + S_high) / 2.0
            if _verify(S_mid, best_a):
                S_high = S_mid
            else:
                S_low = S_mid
        best_S = S_high
    
    return best_S, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    t_start = time.time()
    deadline = 600.0

    def _elapsed():
        return time.time() - t_start

    def _time_left():
        return deadline - _elapsed()

    # Multiple seed arrangements
    sqrt3 = np.sqrt(3)
    
    # Seed 1: Original 1+6+4 grid
    seed1 = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
    ])
    
    # Seed 2: 1+6+4 with 30-degree offset for outer ring
    # Center at origin, ring of 6 at distance 2.5, outer 4 at distance ~4.33
    seed2 = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-2.5, 4.33],
        [2.5, 4.33],
        [-2.5, -4.33],
        [2.5, -4.33],
    ])
    
    # Seed 3: Compact hexagonal lattice - 3 rows
    # Row 1 (y=0): 4 hexagons
    # Row 2 (y=sqrt(3)): 3 hexagons offset by 1.25
    # Row 3 (y=2*sqrt(3)): 4 hexagons
    # Actually for unit hexagons in hex lattice, spacing is 2.5 in x, sqrt(3)*1.5 in y
    # Let me use a proper hex grid
    # In a hex grid with unit hexagons, adjacent centers are 2.5 apart
    # 11 hexagons: try 3-4-4 or 4-3-4 arrangement
    
    # 4-3-4 arrangement (rows of 4, 3, 4)
    # Row 0 (y=0): x = -3.75, -1.25, 1.25, 3.75
    # Row 1 (y=2.165): x = -2.5, 0, 2.5
    # Row 2 (y=4.33): x = -3.75, -1.25, 1.25, 3.75
    seed3 = np.array([
        [-3.75, 0], [-1.25, 0], [1.25, 0], [3.75, 0],
        [-2.5, 2.165], [0, 2.165], [2.5, 2.165],
        [-3.75, 4.33], [-1.25, 4.33], [1.25, 4.33], [3.75, 4.33],
    ])
    
    # Seed 4: 1+6+4 with tighter outer
    seed4 = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.5, 2.0],
        [3.5, 2.0],
        [-3.5, -2.0],
        [3.5, -2.0],
    ])
    
    seeds = [seed1, seed2, seed3, seed4]
    seed_angles_list = [np.zeros(11) for _ in range(len(seeds))]

    def _build_all_hex_data(positions, angles):
        """Build all hex data: vertices, proj_mins, proj_maxs, normals for each hex."""
        data = []
        for i in range(11):
            verts, pmin, pmax, norms = _hex_projections(positions[i, 0], positions[i, 1], angles[i])
            data.append((verts, pmin, pmax, norms))
        return data

    def _get_all_vertices(positions, angles):
        """Get all 66 vertices as (66, 2) array."""
        all_v = []
        for i in range(11):
            all_v.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        return np.vstack(all_v)

    def _objective(params):
        """Objective: minimize outer side length, with penalty for separation violations."""
        if _time_left() < 30:
            return 1e10

        positions = params[:22].reshape(11, 2)
        angles = params[22:]

        if not np.all(np.isfinite(params)):
            return 1e10

        hex_data = _build_all_hex_data(positions, angles)

        if not _all_separated_fast(hex_data):
            return 1e10

        all_verts = _get_all_vertices(positions, angles)
        S, _ = _min_outer_side_fast(all_verts)
        if not np.isfinite(S):
            return 1e10
        return S

    def _refine(params, method='Nelder-Mead', maxiter=500):
        """Refine parameters using the specified optimization method."""
        if _time_left() < 30:
            return params, _objective(params)
        try:
            if method == 'Nelder-Mead':
                res = minimize(
                    _objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
            elif method == 'Powell':
                res = minimize(
                    _objective,
                    params,
                    method='Powell',
                    options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7}
                )
            else:
                res = minimize(
                    _objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                if res.fun < _objective(params):
                    return res.x, res.fun
            if res.fun < _objective(params):
                return res.x, res.fun
        except Exception:
            pass
        return params, _objective(params)

    # Evaluate all seeds and pick the best
    best_params = None
    best_obj = np.inf
    
    for seed_pos, seed_ang in zip(seeds, seed_angles_list):
        x0 = np.zeros(33)
        x0[:22] = seed_pos.flatten()
        x0[22:] = seed_ang
        obj = _objective(x0)
        if obj < best_obj:
            best_obj = obj
            best_params = x0.copy()

    # Multi-start optimization
    rng = np.random.default_rng(42)
    bounds = [(-6, 6)] * 22 + [(-30, 30)] * 11

    # Phase 1: Differential evolution for global search
    if _time_left() > 250:
        try:
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=40,
                tol=1e-6,
                popsize=12,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 2: Local refinements from best
    if _time_left() > 80:
        p, obj = _refine(best_params.copy(), method='Powell', maxiter=400)
        if obj < best_obj:
            best_obj = obj
            best_params = p

        p, obj = _refine(best_params.copy(), method='Nelder-Mead', maxiter=400)
        if obj < best_obj:
            best_obj = obj
            best_params = p

    # Phase 3: Multi-start perturbations
    if _time_left() > 60:
        for pert_scale in [0.5, 0.3, 0.15]:
            if _time_left() < 40:
                break
            n_trials = max(1, int((_time_left() - 40) / 30))
            for trial in range(n_trials):
                if _time_left() < 40:
                    break
                perturbed = best_params + rng.normal(0, pert_scale, size=33)
                perturbed[22:] = np.clip(perturbed[22:], -30, 30)
                perturbed[:22] = np.clip(perturbed[:22], -6, 6)

                p, obj = _refine(perturbed.copy(), method='Powell', maxiter=200)
                if obj < best_obj:
                    best_obj = obj
                    best_params = p

                if _time_left() > 50:
                    p, obj = _refine(perturbed.copy(), method='Nelder-Mead', maxiter=200)
                    if obj < best_obj:
                        best_obj = obj
                        best_params = p

    # Phase 4: Try from other seeds too
    if _time_left() > 60:
        for seed_pos, seed_ang in zip(seeds, seed_angles_list):
            if _time_left() < 40:
                break
            x0 = np.zeros(33)
            x0[:22] = seed_pos.flatten()
            x0[22:] = seed_ang
            p, obj = _refine(x0.copy(), method='Powell', maxiter=300)
            if obj < best_obj:
                best_obj = obj
                best_params = p

    # Final refinement
    if _time_left() > 20:
        p, obj = _refine(best_params.copy(), method='Nelder-Mead', maxiter=300)
        if obj < best_obj:
            best_obj = obj
            best_params = p

    # Compute final result
    positions = best_params[:22].reshape(11, 2)
    angles = best_params[22:]

    all_verts = _get_all_vertices(positions, angles)
    final_S, final_angle = _min_outer_side_fast(all_verts)

    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    outer_hex_side_length = float(final_S)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
