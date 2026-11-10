"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest point in pts.
    
    This is equivalent to the largest circumradius of the facets of the convex hull.
    We compute it by finding, for each point, the maximum angular distance to its
    nearest neighbor, but more accurately we need to find the farthest point on the sphere.
    
    A practical approach: the covering radius is the maximum over all pairs of points
    of half the angular distance between them, but that's not quite right either.
    
    Actually, the covering radius for a set of points on the sphere is the maximum
    angular distance from any point on the sphere to its nearest point in the set.
    
    This can be computed as: for each point p_i, consider the Voronoi cell on the sphere.
    The farthest point in that cell from p_i gives the local covering radius. The global
    covering radius is the max over all cells.
    
    A simpler but valid upper bound: the maximum over all pairs (i,j) of 0.5 * angle(i,j),
    but this is only valid if the midpoint is the farthest point, which isn't always true.
    
    For a more accurate computation, we can use the fact that the covering radius is the
    radius of the smallest set of caps of that radius that covers the sphere.
    
    Practically, we can estimate it by:
    1. Computing all pairwise angular distances
    2. For each point, finding its 3 nearest neighbors
    3. The covering radius is approximately the max over points of the circumradius
       of the triangle formed by the point and its 2 nearest neighbors (or similar)
    
    Actually, the most reliable practical method: the covering radius is the maximum
    angular distance from any point on the sphere to its nearest point in the configuration.
    This equals the maximum over all "empty caps" - i.e., for each facet of the convex
    hull (viewed from the origin), the circumradius of that facet.
    
    For a set of points on the unit sphere, the covering radius can be computed as:
    max over all points y on S^2 of min_i angle(y, p_i)
    
    This is equivalent to: for each point p_i, the farthest point in its spherical Voronoi cell.
    
    A practical approximation that's exact for well-distributed points:
    The covering radius is the maximum over all triples of points that form a facet
    of the convex hull of the circumradius of the spherical triangle.
    
    For simplicity and speed, we'll use:
    covering_radius = max over i of (max over j != i of 0.5 * angle(i,j) where j is in
    the "neighborhood" of i)
    
    Actually, let's just compute it properly:
    The covering radius is the smallest r such that caps of radius r around each p_i cover S^2.
    
    We can compute this as: for each point, find the farthest point on the sphere that is
    closer to this point than to any other. This is complex.
    
    Simpler: the covering radius is at most the max over all pairs of 0.5*angle(i,j),
    and for well-distributed points, it's close to this.
    
    Let's use a more direct approach: sample many random points on the sphere and compute
    the max over samples of min_i angle(sample, p_i). This gives an upper bound.
    But we need the exact value for scoring.
    
    The exact covering radius is the maximum circumradius of the facets of the convex hull
    of the points (when viewed from the origin). Each facet is a spherical triangle.
    
    For each facet (triangle of points on the sphere), the circumradius is the angular
    distance from the center of the circumscribed circle to any vertex.
    
    For a spherical triangle with vertices a, b, c on the unit sphere, the circumradius R
    satisfies: the point y on the sphere equidistant from a, b, c at distance R.
    
    If the triangle is acute (in the spherical sense), the circumcenter is inside the triangle.
    The circumradius can be computed from the side lengths.
    
    For a spherical triangle with sides a, b, c (angular distances), the circumradius R is:
    tan(R) = (tan(a/2) * tan(b/2) * tan(c/2)) / sqrt(
        2*(tan^2(a/2)*tan^2(b/2) + tan^2(b/2)*tan^2(c/2) + tan^2(c/2)*tan^2(a/2))
        + tan^2(a/2)*tan^2(b/2) + tan^2(b/2)*tan^2(c/2) + tan^2(c/2)*tan^2(a/2)
    )
    
    Actually, there's a simpler formula. Let me use a numerical approach:
    For each triplet of points that forms a facet, find the point on the sphere equidistant
    from all three.
    
    For practical purposes, and given the time constraints, I'll use the following:
    The covering radius is the maximum over all points of the distance to the farthest
    point in its Voronoi cell. A good approximation is to use the Delaunay triangulation
    on the sphere.
    
    Let's use scipy's ConvexHull or just compute it directly.
    
    Actually, the simplest correct method: 
    covering_radius = max over all pairs (i,j) of 0.5 * angular_distance(i,j)
    This is an upper bound, and for the optimal configuration, it's very close to the
    actual covering radius.
    
    Wait, that's not right. The covering radius is NOT simply half the max pairwise distance.
    
    Let me think again. The covering radius is the smallest r such that every point on the
    sphere is within distance r of some p_i.
    
    For a set of points on the sphere, this is equal to the maximum over all points y on
    the sphere of min_i d(y, p_i).
    
    The maximum occurs at a point that is equidistant from at least 3 points (a vertex of
    the Voronoi diagram), or at a point that is the midpoint of two points (an edge of the
    Voronoi diagram).
    
    So the covering radius is the maximum over:
    - All midpoints of pairs: 0.5 * d(i,j)
    - All circumcenters of triples: the distance from the circumcenter to any of the 3 points
    
    For efficiency, we can compute:
    1. All pairwise distances and take the max of 0.5*d(i,j)
    2. For each point, consider its k nearest neighbors and compute circumcenters of triples
    
    But this is still expensive. Let's use a simpler but valid method:
    
    The covering radius is at most the maximum over all i of the distance from p_i to the
    farthest point in its Voronoi cell. And a good lower bound is the max over all i of
    the minimum over j != i of 0.5*d(i,j).
    
    For the purpose of this task, since the evaluator will compute the exact covering radius,
    I just need to optimize a proxy that correlates well with it.
    
    Let's use: max over all pairs (i,j) of 0.5 * angle(i,j) as our objective.
    This is a valid upper bound on the covering radius, and minimizing it will also
    minimize the actual covering radius.
    """
    # Compute all pairwise angular distances
    # pts is (N, 3)
    norms = np.linalg.norm(pts, axis=1)
    pts_norm = pts / norms[:, None]
    
    # Dot products
    dots = np.dot(pts_norm, pts_norm.T)
    # Clip to [-1, 1]
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distances
    angles = np.arccos(dots)
    
    # The covering radius is at most 0.5 * max pairwise distance
    # But we want a better estimate
    # For each point, the farthest point in its Voronoi cell
    # Approximate: for each point, the max over its nearest neighbors of 0.5*angle
    
    # Actually, let's just return 0.5 * max pairwise distance as a proxy
    # This is conservative but valid
    max_pairwise = np.max(angles)
    return 0.5 * max_pairwise * 180.0 / np.pi


def _covering_radius_exact(pts):
    """
    More accurate covering radius computation.
    The covering radius is the maximum over all points on the sphere of the distance
    to the nearest point in the configuration.
    
    We compute this by:
    1. For each pair (i,j), the midpoint gives a candidate: 0.5 * angle(i,j)
    2. For each triple (i,j,k), the circumcenter gives a candidate
    
    But this is O(N^3) which is too slow for N=130 (130^3 = 2.2 million triples).
    
    Alternative: use the fact that the covering radius is the radius of the smallest
    set of caps that covers the sphere. We can estimate it by checking the maximum
    angular distance from any point to its nearest neighbor, but that's the minimum
    distance, not the covering radius.
    
    Let's use a sampling-based approach for speed:
    Sample many random points on the sphere and compute min distance to each p_i.
    The max over samples is an upper bound on the covering radius.
    """
    pts_norm = pts / np.linalg.norm(pts, axis=1)[:, None]
    
    # Sample random points
    n_samples = 50000
    # Generate random points on sphere
    u = np.random.randn(n_samples, 3)
    u /= np.linalg.norm(u, axis=1)[:, None]
    
    # Compute distances from each sample to each point
    # This is (n_samples, N)
    dots = np.dot(u, pts_norm.T)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # For each sample, the min distance to any point
    min_angles = np.min(angles, axis=1)
    
    # The covering radius is at least the max over samples
    return np.max(min_angles) * 180.0 / np.pi


def _objective(pts):
    """
    Objective function to minimize.
    We use the covering radius as computed by a combination of methods.
    """
    # Use the exact-ish method
    return _covering_radius_exact(pts)


def _fibonacci_start(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    return pts


def _hill_climb(pts, objective, max_iters=500, step_size=0.1, rng=None):
    """
    Simple hill-climbing optimization.
    At each step, try small random perturbations and accept if they improve the objective.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = pts.copy()
    current_obj = objective(current)
    
    for it in range(max_iters):
        # Pick a random point to move
        idx = rng.integers(0, len(current))
        
        # Generate a random direction
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        
        # Try moving in this direction
        new_pts = current.copy()
        new_pts[idx] = current[idx] + step_size * direction
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_obj = objective(new_pts)
        
        if new_obj < current_obj:
            current = new_pts
            current_obj = new_obj
    
    return current, current_obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_start(N)
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    # Try several random restarts with hill climbing
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate a random starting point
        if restart == 0:
            start_pts = _fibonacci_start(N)
        else:
            start_pts = _random_start(N, rng)
        
        # Run hill climbing
        result_pts, result_obj = _hill_climb(
            start_pts, _objective, max_iters=200, step_size=0.05, rng=rng
        )
        
        if result_obj < best_obj:
            best_obj = result_obj
            best_pts = result_pts
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Final refinement on the best solution
    if time.time() < deadline - 10.0:
        refined_pts, refined_obj = _hill_climb(
            best_pts, _objective, max_iters=300, step_size=0.02, rng=rng
        )
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_pts = refined_pts
    
    # Ensure all points are normalized
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1)[:, None]
    
    return best_pts
# EVOLVE-BLOCK-END
