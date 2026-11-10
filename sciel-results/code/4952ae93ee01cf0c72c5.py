# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _unit_hex_verts():
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _min_s_vectorized(centers, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (centered at origin, given angle)
    that contains all 12 unit hexagons at given centers.
    Returns s.
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)

    uv = _unit_hex_verts()  # (6, 2)
    # For each inner hex i and each normal n: max projection of vertices
    # proj[i, n, v] = (centers[i] + uv[v]) dot normals[n]
    # = centers[i] dot normals[n] + uv[v] dot normals[n]
    # centers @ normals.T -> (12, 6)
    # uv @ normals.T -> (6, 6)
    center_proj = centers @ normals.T  # (12, 6)
    uv_proj = uv @ normals.T  # (6, 6)
    # max over vertices for each hex, each normal
    max_proj = center_proj + np.max(uv_proj, axis=0)  # (12, 6)
    # overall max
    max_support = np.max(max_proj)
    s = 2.0 * max_support / np.sqrt(3.0)
    return s


def _objective_full(params):
    """
    params: 24 values = [cx0, cy0, cx1, cy1, ..., cx11, cy11, outer_angle_deg]
    Actually let's use 25: 24 centers + 1 outer angle.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return 1e10
    centers = params[:24].reshape(12, 2)
    outer_angle_deg = params[24]
    s = _min_s_vectorized(centers, outer_angle_deg)
    return s


def _objective_full_constrained(params, sep_margin=1e-4):
    """
    Objective with soft penalty for separation violations.
    params: 25 values
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return 1e10
    centers = params[:24].reshape(12, 2)
    outer_angle_deg = params[24]
    s = _min_s_vectorized(centers, outer_angle_deg)

    # Quick separation check: for unit hexagons at angle 0, two hexagons overlap
    # iff center distance < 2*sqrt(3) in the hex lattice sense.
    # More precisely: for two unit hexagons at angle 0, the minimum distance
    # between centers for non-overlap depends on direction.
    # The hexagonal tiling has spacing 2*sqrt(3) in 6 directions.
    # For arbitrary positions, use: if center distance < 2*sqrt(3) - epsilon,
    # they might overlap. But this is conservative.
    # Better: compute actual SAT for close pairs only.
    n = 12
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0 * np.sqrt(3.0) - 0.1:
                # Potential overlap, add penalty
                s += 10.0 * (2.0 * np.sqrt(3.0) - d)
    return s


def _check_separation_fast(centers, tol=1e-6):
    """
    Fast separation check for unit hexagons all at angle 0.
    Two unit hexagons at angle 0 are separated iff for all 6 edge normals,
    the projection intervals don't overlap.
    For hexagons at angle 0, the edge normals are at 30, 90, 150, 210, 270, 330 degrees.
    The support in direction n for a unit hex at center c is c.dot(n) + support_unit(n).
    Two hexes are separated if for some normal n:
      c1.dot(n) + sup(n) < c2.dot(n) - sup(n)  OR  c2.dot(n) + sup(n) < c1.dot(n) - sup(n)
    i.e., |c1 - c2|.dot(n) > 2 * sup(n) for some n.
    """
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    uv = _unit_hex_verts()
    # Support of unit hex in each normal direction
    sup = np.max(uv @ normals.T, axis=0)  # (6,)
    # For pair (i,j): separated if max_k |(ci-cj).dot(nk) - 2*sup[k]| ... 
    # Actually: separated if exists k such that (ci-cj).dot(nk) > 2*sup[k] OR (cj-ci).dot(nk) > 2*sup[k]
    # i.e., |(ci-cj).dot(nk)| > 2*sup[k] for some k
    n = 12
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dots = np.abs(diff @ normals.T)  # (6,)
            if np.any(dots > 2.0 * sup + tol):
                continue
            else:
                return False
    return True


def _make_hex_lattice_initial():
    """
    Create a hexagonal lattice arrangement of 12 unit hexagons.
    Use axial coordinates.
    """
    sqrt3 = np.sqrt(3.0)
    # Hex lattice: spacing 2*sqrt(3)
    # Use a compact cluster
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1),
    ]
    centers = []
    for q, r in axial:
        x = 2.0 * sqrt3 * (q + r / 2.0)
        y = 2.0 * sqrt3 * (r * np.sqrt(3.0) / 2.0)
        centers.append([x, y])
    centers = np.array(centers)
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    return centers


def _make_spread_initial():
    """
    A more spread out initial configuration for DE.
    """
    sqrt3 = np.sqrt(3.0)
    centers = np.array([
        [0.0, 0.0],
        [2.0 * sqrt3, 0.0],
        [-2.0 * sqrt3, 0.0],
        [sqrt3, 3.0],
        [-sqrt3, 3.0],
        [sqrt3, -3.0],
        [-sqrt3, -3.0],
        [3.0 * sqrt3, 1.5],
        [-3.0 * sqrt3, 1.5],
        [3.0 * sqrt3, -1.5],
        [-3.0 * sqrt3, -1.5],
        [0.0, -4.0],
    ])
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    return centers


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    best_s = np.inf
    best_params = None

    # Initial configurations
    init1 = _make_hex_lattice_initial()
    init2 = _make_spread_initial()

    # Phase 1: Differential Evolution for global search
    # 25 parameters: 12*2 centers + 1 outer angle
    # Bounds: centers in [-10, 10], angle in [0, 60]
    bounds = [(-10.0, 10.0)] * 24 + [(0.0, 60.0)]

    def _de_objective(params):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            return 1e10
        centers = params[:24].reshape(12, 2)
        outer_angle_deg = params[24]
        s = _min_s_vectorized(centers, outer_angle_deg)
        # Add penalty if separation is violated
        # Quick check: if any pair is too close
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0 * np.sqrt(3.0) - 0.05:
                    s += 5.0 * (2.0 * np.sqrt(3.0) - d)
        return s

    try:
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN - 120:
            de_result = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 0.8),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
            )
            if de_result is not None and de_result.fun < best_s:
                best_s = de_result.fun
                best_params = de_result.x.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead local refinement from best DE result
    if best_params is not None:
        try:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                nm_result = minimize(
                    _objective_full,
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 20000,
                        'maxfev': 50000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if nm_result is not None and nm_result.fun < best_s:
                    best_s = nm_result.fun
                    best_params = nm_result.x.copy()
        except Exception:
            pass

    # Phase 3: Try multiple local refinements from different initial points
    for init_centers in [init1, init2]:
        try:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            # Flatten: 12 centers + 1 angle
            for angle in [0.0, 30.0]:
                params_init = np.concatenate([init_centers.flatten(), [angle]])
                nm_res = minimize(
                    _objective_full,
                    params_init,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 10000,
                        'maxfev': 30000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if nm_res is not None and nm_res.fun < best_s:
                    best_s = nm_res.fun
                    best_params = nm_res.x.copy()
        except Exception:
            pass

    # Phase 4: If no valid result, use fallback
    if best_params is None:
        # Use the hex lattice as fallback
        centers = init1
        # Find best angle
        best_angle_s = np.inf
        best_angle = 0.0
        for angle in np.linspace(0, 60, 61):
            s = _min_s_vectorized(centers, angle)
            if s < best_angle_s:
                best_angle_s = s
                best_angle = angle
        best_s = best_angle_s
        best_params = np.concatenate([centers.flatten(), [best_angle]])

    # Extract final solution
    centers = best_params[:24].reshape(12, 2)
    outer_angle_deg = best_params[24]

    # Verify separation
    if not _check_separation_fast(centers):
        # Try to fix by spreading
        # Use a known-valid configuration
        sqrt3 = np.sqrt(3.0)
        centers = np.array([
            [0.0, 0.0],
            [2.0 * sqrt3, 0.0],
            [-2.0 * sqrt3, 0.0],
            [sqrt3, 3.0],
            [-sqrt3, 3.0],
            [sqrt3, -3.0],
            [-sqrt3, -3.0],
            [3.0 * sqrt3, 1.5],
            [-3.0 * sqrt3, 1.5],
            [3.0 * sqrt3, -1.5],
            [-3.0 * sqrt3, -1.5],
            [0.0, -4.0],
        ])
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        best_angle_s = np.inf
        best_angle = 0.0
        for angle in np.linspace(0, 60, 61):
            s = _min_s_vectorized(centers, angle)
            if s < best_angle_s:
                best_angle_s = s
                best_angle = angle
        best_s = best_angle_s
        outer_angle_deg = best_angle

    # Final computation of s
    s_final = _min_s_vectorized(centers, outer_angle_deg)

    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([0.0, 0.0, outer_angle_deg])
    outer_hex_side_length = s_final

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
