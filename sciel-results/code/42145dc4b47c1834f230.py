"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import math

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import math

TIME_LIMIT = 130.0  # safety margin before 150s kill


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on unit sphere using Fibonacci spiral with optional angular offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _covering_radius_chord(points, test_points):
    """Compute covering radius as max chord distance (proportional to angular distance)."""
    tree = cKDTree(points)
    dists, _ = tree.query(test_points)
    return np.max(dists)


def _calc_covering_radius(pts):
    """Calculate the covering radius from the convex hull facets (in radians)."""
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
        max_dist = 0.0
        for s in simplices:
            v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            if np.dot(n, v1) < 0:
                n = -n
            dot_val = np.dot(n, v1)
            if abs(dot_val) < 1e-10:
                continue
            k = 1.0 / (2.0 * dot_val)
            c = k * n
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            if dist > max_dist:
                max_dist = dist
        return max_dist
    except Exception:
        return np.inf


def _spherical_circumcenter_of_triangle(p1, p2, p3):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    Returns the circumcenter as a unit vector and the circumradius (angular distance in radians).
    If the triangle is degenerate, returns None.
    """
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    p3 = p3 / np.linalg.norm(p3)

    d1 = p1 - p2
    d2 = p2 - p3
    c_candidate = np.cross(d1, d2)
    c_norm = np.linalg.norm(c_candidate)

    if c_norm < 1e-12:
        return None, None

    c_candidate = c_candidate / c_norm

    cos_r1 = np.clip(np.dot(c_candidate, p1), -1.0, 1.0)
    cos_r2 = np.clip(np.dot(c_candidate, p2), -1.0, 1.0)
    cos_r3 = np.clip(np.dot(c_candidate, p3), -1.0, 1.0)

    r1 = np.arccos(cos_r1)
    r2 = np.arccos(cos_r2)
    r3 = np.arccos(cos_r3)

    max_diff = max(abs(r1 - r2), abs(r2 - r3), abs(r1 - r3))
    if max_diff > 0.1:
        c_candidate = -c_candidate
        cos_r1 = np.clip(np.dot(c_candidate, p1), -1.0, 1.0)
        cos_r2 = np.clip(np.dot(c_candidate, p2), -1.0, 1.0)
        cos_r3 = np.clip(np.dot(c_candidate, p3), -1.0, 1.0)
        r1 = np.arccos(cos_r1)
        r2 = np.arccos(cos_r2)
        r3 = np.arccos(cos_r3)
        max_diff = max(abs(r1 - r2), abs(r2 - r3), abs(r1 - r3))
        if max_diff > 0.1:
            return None, None

    circumradius = max(r1, r2, r3)
    return c_candidate, circumradius


def _compute_facet_circumradii(points):
    """
    Compute the circumradius (in radians) of each facet of the convex hull.
    Returns a list of (facet_indices, circumradius, circumcenter) tuples.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return []

    facets = []
    for simplex in hull.simplices:
        if len(simplex) == 3:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            cc, cr = _spherical_circumcenter_of_triangle(p1, p2, p3)
            if cc is not None:
                facets.append((simplex, cr, cc))
        elif len(simplex) == 4:
            max_cr = 0
            best_face = None
            best_cc = None
            for i in range(4):
                for j in range(i + 1, 4):
                    for k in range(j + 1, 4):
                        p1, p2, p3 = points[simplex[i]], points[simplex[j]], points[simplex[k]]
                        cc, cr = _spherical_circumcenter_of_triangle(p1, p2, p3)
                        if cc is not None and cr > max_cr:
                            max_cr = cr
                            best_face = (simplex[i], simplex[j], simplex[k])
                            best_cc = cc
            if best_face is not None:
                facets.append((np.array(best_face), max_cr, best_cc))

    return facets


def _optimize(points, test_points, max_iter=500, initial_step=0.03, decay=0.997, deadline=None):
    """Optimize points to minimize covering radius using targeted force heuristic."""
    points = points.copy()
    n = len(points)

    best_points = points.copy()
    best_radius = _covering_radius_chord(points, test_points)

    tree = cKDTree(points)
    step_size = initial_step

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        dists, idx = tree.query(test_points)
        current_radius = np.max(dists)

        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

        n_worst = max(1, len(test_points) // 10)
        worst_indices = np.argsort(dists)[-n_worst:]

        forces = np.zeros((n, 3))

        for j in worst_indices:
            i = idx[j]
            vec = test_points[j] - points[i]
            forces[i] += vec

        for i in range(n):
            for j in range(n):
                if i != j:
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-8:
                        forces[i] += 0.01 * d / dist

        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_forces = forces / safe_norms

        radius_ratio = current_radius / (best_radius + 1e-10)
        effective_step = step_size * min(1.0, radius_ratio)

        points += effective_step * unit_forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        tree = cKDTree(points)
        step_size *= decay

    return best_points


def _spherical_gradient_descent(points, max_iter=200, step_size=0.05, deadline=None):
    """
    Spherical Riemannian gradient descent to minimize the maximum facet circumradius.
    Aggregates gradients from all active facets (near max radius) and uses line search.
    """
    points = points.copy()
    n = len(points)
    best_points = points.copy()
    best_max_cr = float('inf')
    momentum = np.zeros_like(points)
    momentum_factor = 0.6

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        facets = _compute_facet_circumradii(points)
        if not facets:
            break

        # Find the max circumradius
        max_cr = max(f[1] for f in facets)

        if max_cr < best_max_cr:
            best_max_cr = max_cr
            best_points = points.copy()

        # Collect all facets within 5% of max radius (active facets)
        threshold = max_cr * 0.95
        active_facets = [f for f in facets if f[1] >= threshold]

        # Compute aggregated force from all active facets
        forces = np.zeros((n, 3))
        counts = np.zeros(n)

        for simplex, cr, cc in active_facets:
            for vi in simplex:
                v = points[vi]
                if cc is not None:
                    # Direction from vertex to circumcenter
                    d = cc - v
                    # Project onto tangent space at v: d - (d·v)v
                    d_tangent = d - np.dot(d, v) * v
                    d_norm = np.linalg.norm(d_tangent)
                    if d_norm > 1e-10:
                        forces[vi] += d_tangent / d_norm
                else:
                    # Fallback: move away from centroid of the facet
                    centroid = np.mean(points[simplex], axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm > 1e-10:
                        centroid = centroid / centroid_norm
                        d = centroid - v
                        d_tangent = d - np.dot(d, v) * v
                        d_norm = np.linalg.norm(d_tangent)
                        if d_norm > 1e-10:
                            forces[vi] += d_tangent / d_norm
                counts[vi] += 1

        # Normalize forces by count
        for i in range(n):
            if counts[i] > 0:
                forces[i] /= counts[i]

        # Apply momentum
        momentum = momentum_factor * momentum + step_size * forces
        points += momentum
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Decay step size
        step_size *= 0.995

    return best_points, best_max_cr


def _refine_with_momentum(points, start_time, time_limit, rng):
    """
    Refine points using momentum-based optimization on the convex hull facets.
    Moves each point toward the average circumcenter of its incident facets.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _calc_covering_radius(points)

    momentum = np.zeros_like(points)
    lr = 0.08
    lr_decay = 0.9995
    momentum_factor = 0.6
    no_improve_count = 0
    patience = 500

    iterations = 0
    while time.time() - start_time < time_limit and iterations < 15000:
        iterations += 1

        try:
            hull = ConvexHull(points)
            simplices = hull.simplices

            targets = np.zeros_like(points)
            counts = np.zeros(N)

            for s in simplices:
                v1, v2, v3 = points[s[0]], points[s[1]], points[s[2]]
                n = np.cross(v2 - v1, v3 - v1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                if np.dot(n, v1) < 0:
                    n = -n
                dot_val = np.dot(n, v1)
                if abs(dot_val) < 1e-10:
                    continue
                k = 1.0 / (2.0 * dot_val)
                c = k * n
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue
                c = c / c_norm

                targets[s[0]] += c
                targets[s[1]] += c
                targets[s[2]] += c
                counts[s[0]] += 1
                counts[s[1]] += 1
                counts[s[2]] += 1

            for j in range(N):
                if counts[j] > 0:
                    targets[j] /= counts[j]
                else:
                    targets[j] = points[j]

            diff = targets - points
            momentum = momentum_factor * momentum + lr * diff
            points += momentum
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        except Exception:
            break

        current_radius = _calc_covering_radius(points)
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        lr *= lr_decay

        # Add noise to escape local minima
        if iterations % 200 == 0 or no_improve_count > patience:
            noise_scale = 0.02 if no_improve_count < patience else 0.05
            noise = rng.standard_normal(points.shape) * noise_scale
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            no_improve_count = 0

        if no_improve_count > patience * 3:
            break

    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    rng = np.random.default_rng(42)

    # Test points: use a moderate density for speed
    N_test = 3000
    test_points = _fibonacci_sphere(N_test)

    best_points = None
    best_radius = float('inf')

    # Multiple restarts with different initializations
    restarts = [
        ("fibonacci", _fibonacci_sphere(N, offset=0.0)),
        ("fibonacci_rot", _fibonacci_sphere(N, offset=np.pi / 7)),
        ("random", _random_sphere(N, seed=42)),
        ("random2", _random_sphere(N, seed=123)),
        ("random3", _random_sphere(N, seed=456)),
        ("random4", _random_sphere(N, seed=789)),
        ("random5", _random_sphere(N, seed=101)),
        ("random6", _random_sphere(N, seed=202)),
    ]

    for name, init_points in restarts:
        if time.time() > deadline - 15:
            break

        # Coarse optimization
        opt_points = _optimize(init_points, test_points, max_iter=300, initial_step=0.04,
                               decay=0.996, deadline=deadline)

        radius = _covering_radius_chord(opt_points, test_points)
        if radius < best_radius:
            best_radius = radius
            best_points = opt_points.copy()

        # If we've found a good solution, refine with momentum-based method
        if best_points is not None and time.time() < deadline - 25:
            refined, refined_rad = _refine_with_momentum(best_points, start_time, deadline - 20, rng)
            if refined_rad < best_radius:
                best_radius = refined_rad
                best_points = refined.copy()

    # Fine polish: use denser test points for the best configuration
    if best_points is not None and time.time() < deadline - 20:
        N_test_fine = 10000
        test_points_fine = _fibonacci_sphere(N_test_fine)

        polished = _optimize(best_points, test_points_fine, max_iter=200, initial_step=0.01,
                             decay=0.998, deadline=deadline)

        radius_fine = _covering_radius_chord(polished, test_points_fine)
        if radius_fine < best_radius:
            best_points = polished

    # Spherical gradient descent to minimize facet circumradii
    if best_points is not None and time.time() < deadline - 15:
        sg_points, sg_cr = _spherical_gradient_descent(best_points, max_iter=100, step_size=0.03, deadline=deadline)

        radius_sg = _covering_radius_chord(sg_points, test_points_fine)
        if radius_sg < best_radius:
            best_points = sg_points

    # Final fine-tuning with very small steps
    if best_points is not None and time.time() < deadline - 10:
        fine_points = best_points.copy()
        fine_radius = _calc_covering_radius(fine_points)
        momentum = np.zeros_like(fine_points)
        lr = 0.01

        for _ in range(2000):
            if time.time() - start_time >= deadline - 5:
                break
            try:
                hull = ConvexHull(fine_points)
                simplices = hull.simplices
                targets = np.zeros_like(fine_points)
                counts = np.zeros(N)
                for s in simplices:
                    v1, v2, v3 = fine_points[s[0]], fine_points[s[1]], fine_points[s[2]]
                    n = np.cross(v2 - v1, v3 - v1)
                    n_norm = np.linalg.norm(n)
                    if n_norm < 1e-10:
                        continue
                    n = n / n_norm
                    if np.dot(n, v1) < 0:
                        n = -n
                    dot_val = np.dot(n, v1)
                    if abs(dot_val) < 1e-10:
                        continue
                    k = 1.0 / (2.0 * dot_val)
                    c = k * n
                    c_norm = np.linalg.norm(c)
                    if c_norm < 1e-10:
                        continue
                    c = c / c_norm
                    targets[s[0]] += c
                    targets[s[1]] += c
                    targets[s[2]] += c
                    counts[s[0]] += 1
                    counts[s[1]] += 1
                    counts[s[2]] += 1
                for j in range(N):
                    if counts[j] > 0:
                        targets[j] /= counts[j]
                    else:
                        targets[j] = fine_points[j]
                diff = targets - fine_points
                momentum = 0.7 * momentum + lr * diff
                fine_points += momentum
                norms = np.linalg.norm(fine_points, axis=1, keepdims=True)
                fine_points = fine_points / norms
            except Exception:
                break

            current_rad = _calc_covering_radius(fine_points)
            if current_rad < fine_radius:
                fine_radius = current_rad
                best_points = fine_points.copy()

            lr *= 0.999

    return best_points
# EVOLVE-BLOCK-END
