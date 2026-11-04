"""Heilbronn convex regions n=14: Voronoi relaxation + Nelder-Mead polish."""
import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from scipy.optimize import minimize
import time

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # safety margin before 150s kill


def _hull_area(points):
    """Compute area of convex hull of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute minimum area over all triples of points (vectorized)."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _soft_min_triangle_area(points, beta=80.0):
    """Soft-min surrogate for minimum triangle area using log-sum-exp."""
    n = len(points)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                areas[idx] = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                idx += 1
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum / beta


def _soft_score(points, beta=80.0):
    """Soft score surrogate."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    soft_min = _soft_min_triangle_area(points, beta)
    return soft_min / hull_area


def _negative_soft_score(params, beta=80.0):
    """Negative soft score for minimization."""
    points = params.reshape((-1, 2))
    return -_soft_score(points, beta)


def _voronoi_centroid(points, radius=2.0):
    """
    Compute centroid of each point's Voronoi cell clipped to a disk of given radius.
    Returns array of same shape as points.
    """
    n = len(points)
    centroids = points.copy()
    try:
        vor = Voronoi(points)
        # Get the bounding box of the Voronoi diagram
        # We'll clip to a circle of given radius centered at origin
        for i in range(n):
            # Get the vertices of the Voronoi region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0 or -1 in region:
                # Unbounded region; use the point itself or a large offset
                # For now, keep the point as centroid
                continue
            # Get the vertices of this region
            verts = vor.vertices[region]
            # Filter to points within the disk
            dists = np.linalg.norm(verts, axis=1)
            mask = dists <= radius
            if np.sum(mask) < 3:
                continue
            verts_clipped = verts[mask]
            # Compute centroid of the polygon
            # Use the average of vertices as a simple approximation
            centroid = np.mean(verts_clipped, axis=0)
            # But better: use the polygon centroid formula
            # Shoelace formula for centroid
            x = verts_clipped[:, 0]
            y = verts_clipped[:, 1]
            # Close the polygon
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            cross = x * y_next - x_next * y
            area_poly = 0.5 * np.sum(cross)
            if abs(area_poly) < 1e-12:
                continue
            cx = np.sum((x + x_next) * cross) / (6.0 * area_poly)
            cy = np.sum((y + y_next) * cross) / (6.0 * area_poly)
            centroids[i] = [cx, cy]
    except Exception:
        pass
    return centroids


def _voronoi_relax(points, n_iters=20, radius=2.0):
    """
    Apply Voronoi relaxation: iteratively move each point to the centroid
    of its Voronoi cell within a disk.
    """
    pts = points.copy()
    for _ in range(n_iters):
        new_pts = _voronoi_centroid(pts, radius)
        # Blend to avoid instability
        pts = 0.7 * new_pts + 0.3 * pts
        # Center the points
        center = np.mean(pts, axis=0)
        pts = pts - center
    return pts


def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    np.random.seed(42)

    # Two-ring configurations
    for r_in in [0.3, 0.4, 0.5, 0.6]:
        for r_out in [1.0, 1.1]:
            for offset in [0.0, np.pi / 14, np.pi / 7]:
                pts = []
                for i in range(7):
                    angle = 2 * np.pi * i / 7 + offset
                    pts.append([r_in * np.cos(angle), r_in * np.sin(angle)])
                for i in range(7):
                    angle = 2 * np.pi * i / 7 + offset + np.pi / 7
                    pts.append([r_out * np.cos(angle), r_out * np.sin(angle)])
                configs.append(np.array(pts))

    # Three-ring configurations
    for r1, r2, r3 in [(0.3, 0.6, 1.0), (0.4, 0.7, 1.1), (0.2, 0.5, 0.9)]:
        for offset in [0.0, np.pi / 10]:
            pts = []
            for i in range(4):
                angle = 2 * np.pi * i / 4 + offset
                pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
            for i in range(5):
                angle = 2 * np.pi * i / 5 + offset
                pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            for i in range(5):
                angle = 2 * np.pi * i / 5 + offset + np.pi / 5
                pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
            configs.append(np.array(pts))

    # Random configurations
    for _ in range(8):
        pts = np.random.randn(N, 2) * 0.8
        center = np.mean(pts, axis=0)
        pts = pts - center
        configs.append(pts)

    return configs


def _coordinate_descent_polish(points, n_iters=50):
    """
    Coordinate descent polish: for each point, try small perturbations
    and keep the one that improves the score.
    """
    pts = points.copy()
    current_score = _score(pts)
    
    for iteration in range(n_iters):
        improved = False
        for i in range(N):
            best_local = pts[i].copy()
            best_local_score = current_score
            # Try small perturbations in x and y
            for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    test_pts = pts.copy()
                    test_pts[i] = [pts[i][0] + dx, pts[i][1] + dy]
                    s = _score(test_pts)
                    if s > best_local_score:
                        best_local_score = s
                        best_local = test_pts[i].copy()
            if not np.allclose(best_local, pts[i]):
                pts[i] = best_local
                current_score = best_local_score
                improved = True
        if not improved:
            break
    
    return pts


def construct():
    """Construct 14 points to maximize min triangle area / convex hull area."""
    best_points = None
    best_score = -1.0
    deadline = DEADLINE

    # Generate initial configurations
    initial_configs = _generate_initial_configs()

    # Process each configuration
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break

        # Center and normalize
        center = np.mean(init_pts, axis=0)
        init_pts = init_pts - center
        max_norm = np.max(np.linalg.norm(init_pts, axis=1))
        if max_norm > 1e-10:
            init_pts = init_pts / max_norm

        # Apply Voronoi relaxation
        vor_pts = _voronoi_relax(init_pts, n_iters=15, radius=1.5)
        vor_score = _score(vor_pts)

        # Try Nelder-Mead optimization from the Voronoi-relaxed point
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    _negative_soft_score,
                    vor_pts.flatten(),
                    args=(80.0,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                nm_pts = result.x.reshape((-1, 2))
                nm_score = _score(nm_pts)
            except Exception:
                nm_pts = vor_pts
                nm_score = vor_score
        else:
            nm_pts = vor_pts
            nm_score = vor_score

        # Also try from the original init
        if time.time() < deadline - 10:
            try:
                result2 = minimize(
                    _negative_soft_score,
                    init_pts.flatten(),
                    args=(80.0,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                nm2_pts = result2.x.reshape((-1, 2))
                nm2_score = _score(nm2_pts)
            except Exception:
                nm2_pts = init_pts
                nm2_score = _score(init_pts)
        else:
            nm2_pts = init_pts
            nm2_score = _score(init_pts)

        # Pick the best from this configuration
        candidates = [(vor_pts, vor_score), (nm_pts, nm_score), (nm2_pts, nm2_score)]
        best_in_config = max(candidates, key=lambda x: x[1])
        if best_in_config[1] > best_score:
            best_score = best_in_config[1]
            best_points = best_in_config[0].copy()

    # Final polish with coordinate descent
    if best_points is not None and time.time() < deadline - 5:
        polished = _coordinate_descent_polish(best_points, n_iters=30)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished

    # If no valid result, return a default
    if best_points is None:
        pts = []
        for i in range(7):
            angle = 2 * np.pi * i / 7
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        for i in range(7):
            angle = 2 * np.pi * i / 7 + np.pi / 7
            pts.append([np.cos(angle), np.sin(angle)])
        best_points = np.array(pts)
        center = np.mean(best_points, axis=0)
        best_points = best_points - center

    return best_points


if __name__ == "__main__":
    pts = construct()
    print("Score:", _score(pts))
    print("Points shape:", pts.shape)
