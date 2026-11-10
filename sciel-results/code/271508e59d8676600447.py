"""Rewritten program: force-based repulsion + local search for 50-point sphere covering."""
N = 50
import time
import numpy as np
from scipy.spatial import ConvexHull


def _cover_radius(pts):
    """Compute covering radius (in radians) as max circumradius over hull facets."""
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_angle = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

        # Spherical circumcenter: normalize the sum of vertices weighted by sin of opposite angles
        # sin(angle at a) = |b-c| / (2*|b||c|) but since unit, = |b-c|/2
        # Actually for unit vectors, the circumcenter direction is:
        # (a * sin(A) + b * sin(B) + c * sin(C)) where A,B,C are angles opposite a,b,c
        # sin(A) = |b x c| / (|b||c|) = |b x c| for unit vectors
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)

        w_a = np.linalg.norm(cross_bc)
        w_b = np.linalg.norm(cross_ca)
        w_c = np.linalg.norm(cross_ab)

        total = w_a + w_b + w_c
        if total < 1e-15:
            continue

        circum = a * w_a + b * w_b + c * w_c
        cnorm = np.linalg.norm(circum)
        if cnorm < 1e-15:
            continue
        circum /= cnorm

        # The circumcenter should be on the same side as the facet normal (outward)
        # Check: dot with facet normal should be positive
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm > 1e-15:
            facet_normal /= fnorm
            if np.dot(circum, facet_normal) < 0:
                circum = -circum

        # Angular distance from circumcenter to each vertex
        dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
        min_dot = np.clip(min(dots), -1.0, 1.0)
        angle = np.arccos(min_dot)
        if angle > max_angle:
            max_angle = angle

    return max_angle


def _spherical_kmeans_init(rng, n_clusters=N, n_iter=50):
    """Initialize using spherical k-means for better starting configuration."""
    # Start with random points
    centers = rng.randn(n_clusters, 3)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    for _ in range(n_iter):
        # Assign each of many sample points to nearest center
        # Use a grid of points for assignment
        n_samples = 2000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        # Compute distances
        dots = samples @ centers.T  # (n_samples, n_clusters)
        labels = np.argmax(dots, axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centers[k] = np.sum(samples[mask], axis=0)
            else:
                new_centers[k] = centers[k]

        norms = np.linalg.norm(new_centers, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        new_centers /= norms

        # Check convergence
        if np.allclose(centers, new_centers, atol=1e-8):
            centers = new_centers
            break
        centers = new_centers

    return centers


def _force_based_optimize(pts, max_iters=2000, deadline=None):
    """
    Force-based optimization: points repel each other with a force that grows
    when they are too close, and are attracted toward the farthest facet circumcenters.
    """
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    best_pts = pts.copy()

    step = 0.05
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Compute current covering radius
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        # Find worst facet and its circumcenter
        worst_radius = 0.0
        worst_circum = None
        worst_facet = None

        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cross_bc = np.cross(b, c)
            cross_ca = np.cross(c, a)
            cross_ab = np.cross(a, b)
            w_a = np.linalg.norm(cross_bc)
            w_b = np.linalg.norm(cross_ca)
            w_c = np.linalg.norm(cross_ab)
            total = w_a + w_b + w_c
            if total < 1e-15:
                continue
            circum = a * w_a + b * w_b + c * w_c
            cnorm = np.linalg.norm(circum)
            if cnorm < 1e-15:
                continue
            circum /= cnorm
            facet_normal = np.cross(b - a, c - a)
            fnorm = np.linalg.norm(facet_normal)
            if fnorm > 1e-15:
                facet_normal /= fnorm
                if np.dot(circum, facet_normal) < 0:
                    circum = -circum
            dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
            min_dot = np.clip(min(dots), -1.0, 1.0)
            angle = np.arccos(min_dot)
            if angle > worst_radius:
                worst_radius = angle
                worst_circum = circum
                worst_facet = simplex

        if worst_circum is None:
            break

        # For each point, compute a force:
        # 1. Attraction toward the worst circumcenter (if this point is nearest to it)
        # 2. Repulsion from other points that are too close
        forces = np.zeros_like(pts)

        # Find nearest point to worst circumcenter
        dots_to_circum = pts @ worst_circum
        nearest_idx = np.argmin(dots_to_circum)

        # Attraction force toward circumcenter for the nearest point
        forces[nearest_idx] += 0.5 * (worst_circum - pts[nearest_idx])

        # Repulsion: for each pair, if angular distance < threshold, repel
        # Use a soft repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dot_ij = np.dot(pts[i], pts[j])
                dot_ij = np.clip(dot_ij, -1.0, 1.0)
                angle_ij = np.arccos(dot_ij)
                # Repulsion if angle < ~10 degrees
                if angle_ij < 0.2:  # ~11.5 degrees
                    rep_strength = (0.2 - angle_ij) / 0.2
                    diff = pts[i] - pts[j]
                    dnorm = np.linalg.norm(diff)
                    if dnorm > 1e-10:
                        diff /= dnorm
                        forces[i] += rep_strength * diff
                        forces[j] -= rep_strength * diff

        # Normalize forces and apply
        for i in range(N):
            fnorm = np.linalg.norm(forces[i])
            if fnorm > 1e-10:
                forces[i] /= fnorm

        # Move points
        new_pts = pts + step * forces
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)

        # Check improvement
        new_radius = _cover_radius(new_pts)
        if new_radius < best_radius - 1e-10:
            pts = new_pts
            best_radius = new_radius
            best_pts = pts.copy()
        else:
            # Try smaller step
            step *= 0.85
            if step < 1e-6:
                break

    return best_pts, best_radius


def _hill_climbing(pts, max_iters=1000, deadline=None):
    """
    Hill climbing: try moving each point in random directions, keep improvements.
    """
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(12345)
    step = 0.02

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Generate random direction on tangent plane
            random_dir = rng.randn(3)
            # Project onto tangent plane at pts[i]
            tangent = random_dir - np.dot(random_dir, pts[i]) * pts[i]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm

            new_pts = pts.copy()
            new_pts[i] = pts[i] + step * tangent
            new_pts[i] /= np.linalg.norm(new_pts[i])

            new_radius = _cover_radius(new_pts)
            if new_radius < best_radius - 1e-10:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True
            else:
                # Try opposite direction
                new_pts[i] = pts[i] - step * tangent
                new_pts[i] /= np.linalg.norm(new_pts[i])
                new_radius = _cover_radius(new_pts)
                if new_radius < best_radius - 1e-10:
                    pts = new_pts
                    best_radius = new_radius
                    best_pts = pts.copy()
                    improved = True

        if not improved:
            step *= 0.9
            if step < 1e-6:
                break

    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    rng = np.random.RandomState(42)

    best_pts = None
    best_radius = np.pi

    # Strategy 1: Spherical K-Means initialization + force-based optimization
    for attempt in range(5):
        if time.time() > deadline - 15:
            break

        init_pts = _spherical_kmeans_init(rng, n_clusters=N, n_iter=30)
        opt_pts, opt_radius = _force_based_optimize(init_pts, max_iters=500, deadline=deadline - 5)

        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()

    # Strategy 2: Fibonacci spiral + hill climbing
    if time.time() < deadline - 10:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

        opt_pts, opt_radius = _hill_climbing(fib_pts, max_iters=1000, deadline=deadline - 5)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()

    # Strategy 3: Random restarts + force-based
    for restart in range(10):
        if time.time() > deadline - 10:
            break

        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)

        opt_pts, opt_radius = _force_based_optimize(random_pts, max_iters=300, deadline=deadline - 5)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()

    # Final polish with hill climbing on the best result
    if best_pts is not None and time.time() < deadline - 3:
        final_pts, final_radius = _hill_climbing(best_pts, max_iters=500, deadline=deadline - 1)
        if final_radius < best_radius:
            best_radius = final_radius
            best_pts = final_pts.copy()

    if best_pts is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Ensure all points are on the unit sphere
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)

    return best_pts
