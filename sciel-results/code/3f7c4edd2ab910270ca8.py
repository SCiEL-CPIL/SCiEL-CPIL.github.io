"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = 3 ** 0.5
TRI_AREA = SQRT3 / 4  # area of the equilateral triangle with side 1


def _barycentric_to_cartesian(bary):
    """Convert barycentric coords (u, v) where point = u*A + v*B + (1-u-v)*C
    with A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    So point = (u + 0.5*v, sqrt(3)/2 * v)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = (SQRT3 / 2) * v
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert cartesian (x,y) to barycentric (u,v) where w=1-u-v"""
    x = pts[:, 0]
    y = pts[:, 1]
    v = 2 * y / SQRT3
    u = x - 0.5 * v
    return np.column_stack([u, v])


def _all_triangle_areas(pts):
    """Compute areas of all C(11,3)=165 triangles formed by 11 points.
    Returns array of shape (165,) with areas (not normalized by triangle area).
    """
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = pts[i]
                xj, yj = pts[j]
                xk, yk = pts[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(area)
    return np.array(areas)


def _min_triangle_area(pts):
    return np.min(_all_triangle_areas(pts))


def _objective_smooth(params, beta=50.0):
    """Smooth surrogate: -logsumexp(beta * areas) / beta approximates -min(areas).
    We want to maximize min area, so minimize -min area.
    Using soft-min: softmin(areas) = -(1/beta)*logsumexp(beta*areas)
    So we minimize softmin(areas) = -(1/beta)*logsumexp(beta*areas)
    """
    n = N
    # params is 2*n: barycentric coords (u, v) for each point
    bary = params.reshape(n, 2)
    # Clip to ensure feasibility (u>=0, v>=0, u+v<=1)
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    # Enforce u + v <= 1
    s = u + v
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        u[mask] *= scale
        v[mask] *= scale
    bary_clipped = np.column_stack([u, v])
    pts = _barycentric_to_cartesian(bary_clipped)
    areas = _all_triangle_areas(pts)
    # Soft-min
    max_a = np.max(areas)
    logsumexp = max_a + np.log(np.sum(np.exp(beta * (areas - max_a))))
    softmin = -(logsumexp / beta)
    return softmin


def _objective_exact(params):
    """Exact objective: negative of min triangle area (for minimization)."""
    n = N
    bary = params.reshape(n, 2)
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        u[mask] *= scale
        v[mask] *= scale
    bary_clipped = np.column_stack([u, v])
    pts = _barycentric_to_cartesian(bary_clipped)
    return -_min_triangle_area(pts)


def _random_initial_bary(seed):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary = rng.random((N, 2))
    # Normalize so u + v <= 1
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        bary[mask, 0] *= scale
        bary[mask, 1] *= scale
    return bary.ravel()


def _refine_with_nelder_mead(params0, max_iters=2000, deadline=None):
    """Run Nelder-Mead on the exact objective."""
    result = minimize(
        _objective_exact,
        params0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iters,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True,
            'initial_simplex': None,
        }
    )
    if deadline is not None and time.time() > deadline:
        return params0  # return initial if we've exceeded deadline
    return result.x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_params = None
    best_score = -np.inf

    # Try multiple random restarts with Nelder-Mead
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 30:  # leave 30s for final polish
            break

        seed = 42 + i * 137
        params0 = _random_initial_bary(seed)

        # First, use smooth objective to get close
        smooth_result = minimize(
            _objective_smooth,
            params0,
            args=(30.0,),
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            }
        )

        if time.time() > deadline - 20:
            break

        # Then refine with exact objective
        refined = _refine_with_nelder_mead(smooth_result.x, max_iters=1000, deadline=deadline)

        # Evaluate
        bary = refined.reshape(N, 2)
        u = np.maximum(0, bary[:, 0])
        v = np.maximum(0, bary[:, 1])
        s = u + v
        mask = s > 1
        if np.any(mask):
            scale = 1.0 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        score = _min_triangle_area(pts) / TRI_AREA  # normalize by triangle area

        if score > best_score:
            best_score = score
            best_params = refined.copy()

    if best_params is None:
        # Fallback: simple random
        rng = np.random.RandomState(0)
        bary = _random_initial_bary(0)
        u = np.maximum(0, bary[0::2])
        v = np.maximum(0, bary[1::2])
        s = u + v
        mask = s > 1
        if np.any(mask):
            scale = 1.0 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        return pts

    # Final high-precision polish
    if time.time() < deadline - 5:
        polished = _refine_with_nelder_mead(
            best_params,
            max_iters=5000,
            deadline=deadline
        )
        bary = polished.reshape(N, 2)
        u = np.maximum(0, bary[:, 0])
        v = np.maximum(0, bary[:, 1])
        s = u + v
        mask = s > 1
        if np.any(mask):
            scale = 1.0 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        final_score = _min_triangle_area(pts) / TRI_AREA
        if final_score >= best_score:
            best_params = polished

    # Extract final points
    bary = best_params.reshape(N, 2)
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        u[mask] *= scale
        v[mask] *= scale
    pts = _barycentric_to_cartesian(np.column_stack([u, v]))

    # Verify all points are inside the triangle
    for p in pts:
        if p[0] < -1e-9 or p[0] > 1 + 1e-9 or p[1] < -1e-9 or p[1] > SQRT3 / 2 + 1e-9:
            pass  # Should not happen with clipping

    return pts
