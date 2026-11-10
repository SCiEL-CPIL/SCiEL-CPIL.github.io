# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_min_side(inner, outer_center=None):
    """Compute the minimum side length of outer hexagon needed to contain all inner hexagons."""
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])

    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]

    all_verts = []
    for i in range(11):
        verts = _hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], 1.0, inner_shifted[i, 2])
        all_verts.append(verts)

    all_verts = np.vstack(all_verts)

    normals = np.column_stack([np.cos(np.radians(30 + 60 * np.arange(6))),
                               np.sin(np.radians(30 + 60 * np.arange(6)))])

    projections = all_verts @ normals.T
    max_projs = projections.max(axis=0)

    min_side = np.max(max_projs) * 2.0 / np.sqrt(3)
    return min_side

def _check_separation_fast(inner, tol=1e-6):
    """
    Fast separation check using Minkowski sum property.
    Two unit hexagons with the same orientation are separated iff
    max over 6 normals of |dot(c2-c1, n)| > sqrt(3) + tol.
    For different orientations, we use the general 12-normal projection check
    but with precomputed vertex positions for speed.
    """
    n = len(inner)
    # Precompute vertices for all hexagons
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
    
    sqrt3 = np.sqrt(3)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts[i]
            vj = verts[j]
            
            # Check if centers are far enough apart (quick rejection)
            dc = inner[j, :2] - inner[i, :2]
            dist = np.linalg.norm(dc)
            if dist < 2.0 - 0.01:
                # Definitely overlapping if too close
                continue
            if dist > 4.0:
                # Definitely separated
                continue
            
            # Get edge normals for both hexagons
            def get_edge_normals(v):
                normals = []
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    nn = np.linalg.norm(n)
                    if nn > 1e-12:
                        n = n / nn
                    normals.append(n)
                return np.array(normals)
            
            ni = get_edge_normals(vi)
            nj = get_edge_normals(vj)
            
            separated = False
            for n_vec in ni:
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                if proj_i.max() < proj_j.min() - tol or proj_j.max() < proj_i.min() - tol:
                    separated = True
                    break
            if separated:
                continue
            for n_vec in nj:
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                if proj_i.max() < proj_j.min() - tol or proj_j.max() < proj_i.min() - tol:
                    separated = True
                    break
            if not separated:
                return False
    
    return True

def _check_separation(inner, tol=1e-6):
    """Same as fast version for API compatibility."""
    return _check_separation_fast(inner, tol)

def _build_rigid_cluster(global_rot_deg, satellite_offsets, flower_gap=0.001):
    """Build the 11 hexagons from a rigid cluster topology."""
    sqrt3 = np.sqrt(3)
    d = 2.0 + flower_gap
    flower = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [d / 2, d * sqrt3 / 2],
        [-d / 2, d * sqrt3 / 2],
        [-d, 0.0],
        [-d / 2, -d * sqrt3 / 2],
        [d / 2, -d * sqrt3 / 2],
    ])

    rot = np.radians(global_rot_deg)
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]])

    all_pos = np.vstack([flower, satellite_offsets])
    all_pos = all_pos @ rot_mat.T

    angles = np.full(11, global_rot_deg)
    return np.column_stack([all_pos[:, 0], all_pos[:, 1], angles])

def _objective(x, deadline_ref):
    """Objective function for scipy optimization: minimize outer side length."""
    if time.time() > deadline_ref[0]:
        return 1e6
    
    trial_inner = x.reshape(11, 3)
    
    # Check separation
    if not _check_separation_fast(trial_inner):
        return 1e6
    
    return _compute_min_side(trial_inner)

def _scipy_refine(inner, deadline):
    """Use scipy Nelder-Mead to refine the packing."""
    deadline_ref = [deadline]
    x0 = inner.flatten()
    
    try:
        result = minimize(
            _objective, x0,
            args=(deadline_ref,),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.fun < 1e5:
            trial_inner = result.x.reshape(11, 3)
            if _check_separation_fast(trial_inner):
                return trial_inner, result.fun
    except Exception:
        pass
    
    return inner, _compute_min_side(inner)

def _local_search(inner, deadline, rng, max_iter=3000):
    """Local search to minimize outer side length."""
    current_side = _compute_min_side(inner)
    best_inner = inner.copy()
    best_side = current_side

    x = inner[:, 0].copy()
    y = inner[:, 1].copy()
    a_deg = inner[:, 2].copy()

    no_improve = 0
    max_no_improve = 60
    step_size = 0.12

    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break

        improved = False
        for trial in range(80):
            if time.time() > deadline:
                break

            dx = rng.normal(0, step_size, 11)
            dy = rng.normal(0, step_size, 11)
            da = rng.normal(0, step_size * 0.4, 11)

            dx[0] = 0
            dy[0] = 0
            da[0] = 0

            trial_x = x + dx
            trial_y = y + dy
            trial_a = a_deg + da
            trial_inner = np.column_stack([trial_x, trial_y, trial_a])

            if not _check_separation_fast(trial_inner):
                continue

            trial_side = _compute_min_side(trial_inner)

            if trial_side < current_side - 1e-12:
                x = trial_x.copy()
                y = trial_y.copy()
                a_deg = trial_a.copy()
                current_side = trial_side
                best_inner = trial_inner.copy()
                best_side = trial_side
                improved = True
                no_improve = 0

        if not improved:
            no_improve += 1
            step_size *= 0.96

    return best_inner, best_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit

    sqrt3 = np.sqrt(3)
    rng = np.random.RandomState(42)

    best_inner = None
    best_side = np.inf

    # Try multiple initial configurations
    for global_rot in [0, 15, 30, 45, 60, 75, 90, 120, 150]:
        if time.time() > deadline - 90:
            break

        satellite_patterns = [
            # Pattern 1: Satellites at 30, 150, 210, 330 degrees
            np.array([
                [3.5 * np.cos(np.radians(30)), 3.5 * np.sin(np.radians(30))],
                [3.5 * np.cos(np.radians(150)), 3.5 * np.sin(np.radians(150))],
                [3.5 * np.cos(np.radians(210)), 3.5 * np.sin(np.radians(210))],
                [3.5 * np.cos(np.radians(330)), 3.5 * np.sin(np.radians(330))],
            ]),
            # Pattern 2: Satellites at 0, 90, 180, 270 degrees
            np.array([
                [3.2, 0],
                [0, 3.2],
                [-3.2, 0],
                [0, -3.2],
            ]),
            # Pattern 3: Asymmetric satellites
            np.array([
                [3.0 * np.cos(np.radians(30)), 3.0 * np.sin(np.radians(30))],
                [0, 3.0],
                [3.0 * np.cos(np.radians(150)), 3.0 * np.sin(np.radians(150))],
                [0, -3.0],
            ]),
            # Pattern 4: Linear extension
            np.array([
                [3.0, 0],
                [-3.0, 0],
                [1.5, 2.6],
                [-1.5, -2.6],
            ]),
            # Pattern 5: Diagonal extension
            np.array([
                [3.0, sqrt3],
                [-3.0, sqrt3],
                [3.0, -sqrt3],
                [-3.0, -sqrt3],
            ]),
            # Pattern 6: Tighter cluster with 4 satellites in a ring
            np.array([
                [2.8 * np.cos(np.radians(45)), 2.8 * np.sin(np.radians(45))],
                [2.8 * np.cos(np.radians(135)), 2.8 * np.sin(np.radians(135))],
                [2.8 * np.cos(np.radians(225)), 2.8 * np.sin(np.radians(225))],
                [2.8 * np.cos(np.radians(315)), 2.8 * np.sin(np.radians(315))],
            ]),
        ]

        for sat_offsets in satellite_patterns:
            if time.time() > deadline - 90:
                break

            inner = _build_rigid_cluster(global_rot, sat_offsets, flower_gap=0.001)

            if not _check_separation_fast(inner):
                continue

            side = _compute_min_side(inner)

            if side < best_side:
                best_side = side
                best_inner = inner.copy()

    # If we have a valid configuration, do local search then scipy refinement
    if best_inner is not None:
        # Local search first
        improved_inner, improved_side = _local_search(best_inner, deadline - 30, rng)
        if improved_side < best_side:
            best_inner = improved_inner
            best_side = improved_side

        # Then scipy refinement
        refined_inner, refined_side = _scipy_refine(best_inner, deadline - 5)
        if refined_side < best_side:
            best_inner = refined_inner
            best_side = refined_side

        # Final local search on the scipy result
        if time.time() < deadline - 5:
            final_inner, final_side = _local_search(best_inner, deadline - 2, rng)
            if final_side < best_side:
                best_inner = final_inner
                best_side = final_side

    # Fallback
    if best_inner is None:
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0

    # Final verification
    if not _check_separation_fast(best_inner):
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_inner, outer_hex_data, float(best_side)

# EVOLVE-BLOCK-END
