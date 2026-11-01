"""
Covering the sphere with 130 equal caps.

Strategy:
  1. Start from a Fibonacci spiral (deterministic, good spread).
  2. Run Lloyd relaxation (Voronoi centroid method) on the sphere:
       - Compute the spherical Voronoi region for each point.
       - Move each point to the centroid (spherical mean) of its region.
       - Repeat for many iterations with decreasing step size.
  3. Use a time budget to ensure we finish well before the 150 s limit.
  4. Final polish: a few extra Lloyd steps with very small step size.

The covering radius is the maximum over all points on the sphere of the
angular distance to the nearest configuration point.  For a set of points
on the sphere this equals the largest circumradius of the spherical
Voronoi cells (equivalently, the largest facet circumradius of the
convex hull).  Lloyd relaxation is known to reduce the covering radius
monotonically in practice.
"""

import numpy as np
import time

N = 130


def _normalize(pts):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _spherical_voronoi_centroids(pts, n_samples=40000):
    """
    Approximate spherical Voronoi centroids via Monte-Carlo sampling.

    For each sample point on the sphere, assign it to the nearest configuration
    point.  Then the centroid of the samples assigned to point i is an
    approximation of the spherical Voronoi centroid.

    Returns an array of shape (N, 3) of approximate centroids.
    """
    # Generate sample points uniformly on the sphere
    # Use stratified sampling for better coverage
    # We'll use a combination of a dense Fibonacci grid and random jitter
    n_grid = int(np.sqrt(n_samples))
    # Create a Fibonacci-like grid for stratification
    i_grid = np.arange(n_grid)
    # Use golden ratio for stratified sampling
    golden = (1 + np.sqrt(5)) / 2.0

    # Actually, let's just use a large Fibonacci spiral for samples
    i_s = np.arange(n_samples) + 0.5
    z_s = 1.0 - 2.0 * i_s / n_samples
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(np.maximum(0.0, 1.0 - z_s * z_s))
    samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])

    # For each sample, find the nearest configuration point
    # Compute dot products: samples (n_samples, 3) @ pts.T (3, N) -> (n_samples, N)
    # This can be memory-heavy; do it in chunks
    chunk_size = 10000
    assignments = np.empty(n_samples, dtype=np.int32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        dots = samples[start:end] @ pts.T  # (chunk, N)
        assignments[start:end] = np.argmax(dots, axis=1)

    # Compute centroids
    centroids = np.zeros((N, 3))
    counts = np.zeros(N, dtype=np.float64)
    for k in range(N):
        mask = assignments == k
        if mask.sum() > 0:
            centroids[k] = samples[mask].mean(axis=0)
            counts[k] = mask.sum()

    # Normalize centroids; if a centroid is zero (no samples), keep original point
    for k in range(N):
        norm = np.linalg.norm(centroids[k])
        if norm < 1e-10:
            centroids[k] = pts[k]
        else:
            centroids[k] /= norm

    return centroids


def _covering_radius(pts, n_samples=20000):
    """
    Estimate the covering radius in degrees.
    For each sample point, compute the max angular distance to nearest pt.
    """
    i_s = np.arange(n_samples) + 0.5
    z_s = 1.0 - 2.0 * i_s / n_samples
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(np.maximum(0.0, 1.0 - z_s * z_s))
    samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])

    chunk_size = 10000
    max_dot = -1.0
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        dots = samples[start:end] @ pts.T
        nearest_dots = dots.max(axis=1)
        max_dot = max(max_dot, nearest_dots.max() if len(nearest_dots) > 0 else -1.0)
        # Actually we want the MIN over samples of the max dot? No.
        # Covering radius = max over y of min over x of angle(y, x)
        # = max over y of arccos(max over x of dot(y, x))
        # So for each sample, the relevant value is max over x of dot.
        # Then covering radius = arccos(min over samples of that max).
    # Wait: max angular distance = arccos(min over y of max over x dot(y,x))
    # So we need the MINIMUM over samples of the maximum dot product.
    min_of_max_dots = 1.0
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        dots = samples[start:end] @ pts.T
        nearest_dots = dots.max(axis=1)
        min_of_max_dots = min(min_of_max_dots, nearest_dots.min() if len(nearest_dots) > 0 else 1.0)

    # Clamp to [-1, 1]
    min_of_max_dots = np.clip(min_of_max_dots, -1.0, 1.0)
    return np.degrees(np.arccos(min_of_max_dots))


def _lloyd_relax(pts, n_iter=200, step_schedule=None):
    """
    Lloyd relaxation on the sphere.
    """
    pts = _normalize(pts.copy())
    for it in range(n_iter):
        # Determine step size
        if step_schedule is not None:
            alpha = step_schedule(it)
        else:
            alpha = 1.0 / (1.0 + it * 0.01)

        centroids = _spherical_voronoi_centroids(pts, n_samples=30000)

        # Move each point toward its centroid
        for k in range(N):
            pts[k] = pts[k] + alpha * (centroids[k] - pts[k])
        pts = _normalize(pts)

    return pts


def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill

    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)

    best_pts = pts.copy()
    best_radius = _covering_radius(pts, n_samples=20000)

    # Phase 1: Coarse Lloyd relaxation
    n_iter_coarse = 50
    for it in range(n_iter_coarse):
        if time.time() > deadline - 30.0:
            break
        alpha = 0.8 / (1.0 + it * 0.05)
        centroids = _spherical_voronoi_centroids(pts, n_samples=30000)
        for k in range(N):
            pts[k] = pts[k] + alpha * (centroids[k] - pts[k])
        pts = _normalize(pts)

        if (it + 1) % 10 == 0:
            rad = _covering_radius(pts, n_samples=20000)
            if rad < best_radius:
                best_radius = rad
                best_pts = pts.copy()

    # Phase 2: Fine Lloyd relaxation with smaller steps
    n_iter_fine = 50
    for it in range(n_iter_fine):
        if time.time() > deadline - 15.0:
            break
        alpha = 0.3 / (1.0 + it * 0.1)
        centroids = _spherical_voronoi_centroids(pts, n_samples=50000)
        for k in range(N):
            pts[k] = pts[k] + alpha * (centroids[k] - pts[k])
        pts = _normalize(pts)

        if (it + 1) % 10 == 0:
            rad = _covering_radius(pts, n_samples=30000)
            if rad < best_radius:
                best_radius = rad
                best_pts = pts.copy()

    # Phase 3: Final polish
    n_iter_polish = 20
    for it in range(n_iter_polish):
        if time.time() > deadline - 5.0:
            break
        alpha = 0.1 / (1.0 + it * 0.2)
        centroids = _spherical_voronoi_centroids(pts, n_samples=80000)
        for k in range(N):
            pts[k] = pts[k] + alpha * (centroids[k] - pts[k])
        pts = _normalize(pts)

    # Final evaluation
    rad = _covering_radius(pts, n_samples=50000)
    if rad < best_radius:
        best_radius = rad
        best_pts = pts.copy()

    # Ensure no coincident points and valid shape
    best_pts = _normalize(best_pts)

    # Check for near-coincident points and separate them if needed
    for i1 in range(N):
        for i2 in range(i1 + 1, N):
            d = np.linalg.norm(best_pts[i1] - best_pts[i2])
            if d < 1e-8:
                # Perturb
                best_pts[i2] = best_pts[i2] + 1e-6 * np.random.randn(3)
                best_pts[i2] = _normalize(best_pts[i2].reshape(1, 3))[0]

    return best_pts
