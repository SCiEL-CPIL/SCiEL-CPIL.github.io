"""Rewritten program: L-BFGS-B on sphere with multi-restart for 50-point spherical covering."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n


def _covering_radius(pts):
    """Compute covering radius in degrees using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_ang = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        a, b, c = verts[0], verts[1], verts[2]

        # Circumcenter of spherical triangle
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)

        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)

        if np.dot(x, outward) < 0:
            x = -x

        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot_val))

        if ang > max_ang:
            max_ang = ang

    return max_ang


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _tangent_direction(pts, idx, random_dir):
    """Get a normalized tangent direction at pts[idx]."""
    tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
    tn = np.linalg.norm(tangent)
    if tn < 1e-15:
        return np.zeros(3)
    return tangent / tn


def _soft_covering_radius(pts, beta=50.0):
    """Smooth surrogate for covering radius using log-sum-exp over a fine grid."""
    n_grid = 2000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    dots = grid @ pts.T
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))

    shifted = angles - np.max(angles)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta + np.max(angles)

    return np.degrees(log_sum)


def _local_optimize_hull(pts, time_limit=10.0):
    """Local optimization using hull-based facet refinement."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)

    step_size = 0.03
    stagnation = 0

    while time.time() - start_time < time_limit:
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        worst_facet = None
        worst_ang = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            v1 = a - b
            v2 = a - c
            x = np.cross(v1, v2)
            x = _normalize(x)
            centroid = (a + b + c) / 3.0
            outward = _normalize(centroid)
            if np.dot(x, outward) < 0:
                x = -x
            dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
            ang = np.degrees(np.arccos(dot_val))
            if ang > worst_ang:
                worst_ang = ang
                worst_facet = simplex

        if worst_facet is None:
            break

        verts = pts[worst_facet].copy()
        a, b, c = verts[0], verts[1], verts[2]
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        if np.dot(x, outward) < 0:
            x = -x

        old_pts = pts.copy()

        for idx in worst_facet:
            pts[idx] = _normalize(pts[idx] + step_size * 0.5 * x)

        new_verts = pts[worst_facet]
        na, nb, nc = new_verts[0], new_verts[1], new_verts[2]
        nv1 = na - nb
        nv2 = na - nc
        nx = np.cross(nv1, nv2)
        nx = _normalize(nx)
        ncentroid = (na + nb + nc) / 3.0
        noutward = _normalize(ncentroid)
        if np.dot(nx, noutward) < 0:
            nx = -nx
        ndot_val = np.clip(np.dot(nx, na), -1.0, 1.0)
        new_worst_ang = np.degrees(np.arccos(ndot_val))

        if new_worst_ang < worst_ang - 1e-6:
            cr = _covering_radius(pts)
            if cr < best_cr - 1e-10:
                best_cr = cr
                best_pts = pts.copy()
                stagnation = 0
                step_size = max(0.002, step_size * 0.97)
            else:
                pts = old_pts.copy()
                stagnation += 1
                step_size = max(0.002, step_size * 0.8)
        else:
            pts = old_pts.copy()
            stagnation += 1
            step_size = max(0.002, step_size * 0.8)

            if stagnation > 20:
                for idx in range(N):
                    random_dir = np.random.randn(3)
                    tangent = _tangent_direction(pts, idx, random_dir)
                    pts[idx] = _normalize(pts[idx] + step_size * 0.1 * tangent)
                cr = _covering_radius(pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()
                    stagnation = 0
                else:
                    pts = best_pts.copy()
                    if step_size < 0.003:
                        break

    return best_pts, best_cr


def _gradient_optimize(pts, time_limit=15.0):
    """Optimize using gradient of soft covering radius."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)

    lr = 0.01
    beta = 30.0
    patience = 0
    max_patience = 50

    while time.time() - start_time < time_limit:
        n_grid = 1500
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

        dots = grid @ pts.T
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))

        shifted = angles - np.max(angles)
        weights = np.exp(beta * shifted)
        weights /= weights.sum()

        soft_max_val = np.max(angles) + np.log(weights.sum()) / beta

        grad = np.zeros_like(pts)

        nearest = np.argmax(dots, axis=1)

        for gi in range(n_grid):
            j = nearest[gi]
            dot_val = dots[gi, j]
            dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)

            d_arccos = -1.0 / np.sqrt(1.0 - dot_val * dot_val)

            grad_vec = grid[gi]
            tangent = grad_vec - np.dot(grad_vec, pts[j]) * pts[j]

            grad[j] += weights[gi] * d_arccos * tangent

        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            patience += 1
        else:
            grad /= grad_norm
            new_pts = pts.copy()
            for j in range(N):
                new_pts[j] = _normalize(new_pts[j] - lr * grad[j])

            cr = _covering_radius(new_pts)
            if cr < best_cr - 1e-10:
                best_cr = cr
                best_pts = new_pts.copy()
                pts = new_pts.copy()
                patience = 0
                lr = min(0.05, lr * 1.05)
            else:
                patience += 1
                lr = max(0.001, lr * 0.9)

        if patience > max_patience:
            break

    return best_pts, best_cr


def _lbfgs_sphere(pts, time_limit=20.0):
    """Optimize using L-BFGS-B on the sphere with a smooth objective."""
    start_time = time.time()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)

    # Use a fine grid for the smooth objective
    n_grid = 3000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    beta = 80.0

    def objective(x):
        """Smooth covering radius from flattened parameters."""
        pts_x = x.reshape(N, 3)
        # Project to sphere
        norms = np.linalg.norm(pts_x, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts_x = pts_x / norms

        dots = grid @ pts_x.T
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))

        shifted = angles - np.max(angles)
        soft_max = np.max(angles) + np.log(np.sum(np.exp(beta * shifted))) / beta

        return np.degrees(soft_max)

    x0 = pts.flatten()

    # Run L-BFGS-B with time limit
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 2000, 'maxls': 50}
    )

    if result.success or result.fun < best_cr:
        new_pts = result.x.reshape(N, 3)
        new_pts = np.array([_normalize(p) for p in new_pts])
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr:
            best_pts = new_pts
            best_cr = new_cr

    return best_pts, best_cr


def _multi_restart_local(pts, time_limit=30.0):
    """Multiple restarts with perturbation and local optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)

    restart_count = 0
    while time.time() - start_time < time_limit:
        restart_count += 1
        # Perturb the best solution
        strength = 0.02 + 0.05 * np.random.random()
        perturbed = best_pts.copy()
        for idx in range(N):
            random_dir = np.random.randn(3)
            tangent = _tangent_direction(perturbed, idx, random_dir)
            perturbed[idx] = _normalize(perturbed[idx] + strength * tangent)

        # Local optimize the perturbed version
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            break
        time_limit_sub = min(8.0, remaining)
        new_pts, new_cr = _local_optimize_hull(perturbed, time_limit=time_limit_sub)

        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()

    return best_pts, best_cr


def construct():
    t_start = time.time()
    np.random.seed(42)

    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)

    # Phase 1: Hull-based local optimization
    remaining = DEADLINE - (time.time() - t_start) - 15
    if remaining > 5:
        best_pts, best_cr = _local_optimize_hull(best_pts, time_limit=min(30.0, remaining))

    # Phase 2: L-BFGS-B optimization
    remaining = DEADLINE - (time.time() - t_start) - 15
    if remaining > 5:
        best_pts, best_cr = _lbfgs_sphere(best_pts, time_limit=min(20.0, remaining))

    # Phase 3: Multiple restarts with perturbation
    remaining = DEADLINE - (time.time() - t_start) - 10
    if remaining > 10:
        best_pts, best_cr = _multi_restart_local(best_pts, time_limit=remaining)

    # Phase 4: Final L-BFGS-B polish
    remaining = DEADLINE - (time.time() - t_start) - 5
    if remaining > 5:
        best_pts, best_cr = _lbfgs_sphere(best_pts, time_limit=min(15.0, remaining))

    # Phase 5: Final hull-based polish with very small steps
    remaining = DEADLINE - (time.time() - t_start) - 3
    if remaining > 3:
        final_pts, final_cr = _local_optimize_hull(best_pts, time_limit=remaining)
        if final_cr < best_cr:
            best_pts = final_pts.copy()
            best_cr = final_cr

    # Final check: make sure points are on the unit sphere and valid
    best_pts = np.array([_normalize(p) for p in best_pts])

    # Verify no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] = _normalize(best_pts[j] + 1e-4 * np.random.randn(3))

    return best_pts
