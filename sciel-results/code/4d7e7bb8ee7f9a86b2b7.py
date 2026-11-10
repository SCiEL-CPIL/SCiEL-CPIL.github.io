"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_DEADLINE = 145.0  # Safety margin before 150s kill
_N_SAMPLES_FAST = 8000  # Fast sample count for inner-loop evaluation


def _normalize(pts):
    """Normalize points to unit sphere. Handles 1D and 2D inputs."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms


def _generate_fibonacci_samples(n_samples):
    """Generate well-distributed sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples


def _covering_radius_sampled(pts, samples):
    """
    Fast estimate of covering radius using sample points.
    Returns value in degrees.
    """
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = dots.max(axis=1)
    min_dot = np.min(max_dots)
    radius_rad = np.arccos(np.clip(min_dot, -1.0, 1.0))
    return np.degrees(radius_rad)


def _covering_radius_hull(pts):
    """
    Exact covering radius via convex hull: find the facet with largest circumradius.
    The covering radius equals the maximum circumradius of all hull facets (caps).
    Returns value in degrees.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback to sampled estimate
        samples = _generate_fibonacci_samples(5000)
        return _covering_radius_sampled(pts, samples)
    
    max_circum_rad = 0.0
    for simplex in hull.simplices:
        # simplex contains 3 indices for a triangular facet in 3D
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute circumcenter of triangle abc on the sphere
        # The circumradius of the spherical triangle is the angle from circumcenter to any vertex
        # For a planar triangle, circumradius R = abc / (4*Area)
        # But we need the spherical circumradius, which is the angular distance from the
        # spherical circumcenter to the vertices.
        
        # For a triangle on the unit sphere, the spherical circumradius can be computed
        # as the angle subtended by the arc. A simpler approach: the covering radius is
        # the maximum over all facets of the angular distance from the facet's circumcenter
        # to its vertices. For a planar triangle inscribed in a unit sphere, the circumradius
        # in 3D equals sin(spherical_circumradius) if the circumcenter is at the origin,
        # but that's not generally true.
        
        # Better approach: the farthest point on the sphere from all three vertices
        # of the facet is the antipode of the facet's normal direction (if the facet
        # is a "cap"). Actually, the covering radius for a facet is the angular distance
        # from the facet's circumcenter (on the sphere) to the farthest vertex.
        
        # For a spherical triangle, the circumradius R_c satisfies:
        # sin(R_c) = (a * b * c) / (4 * sin(A) * sin(B) * sin(C))
        # where a,b,c are side lengths (angular) and A,B,C are angles.
        # This is complex. Let's use a numerical approach:
        # The circumcenter of a spherical triangle is the point equidistant from all three vertices.
        # It lies in the plane of the triangle (in 3D) at some distance from origin.
        
        # Compute the 3D circumcenter of the triangle
        # The circumcenter in 3D of triangle abc is the point equidistant from a, b, c.
        # It lies in the plane of the triangle.
        edge_ab = b - a
        edge_ac = c - a
        cross = np.cross(edge_ab, edge_ac)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-12:
            continue
        normal = cross / cross_norm
        
        # The circumcenter in 3D: solve for point p = a + s*edge_ab + t*edge_ac
        # such that |p-a|^2 = |p-b|^2 = |p-c|^2
        # |p-a|^2 = |p-b|^2 => 2*p.(b-a) = |b|^2 - |a|^2
        # Since all on unit sphere: |a|^2 = |b|^2 = |c|^2 = 1
        # So 2*p.(b-a) = 0 => p.(b-a) = 0
        # Similarly p.(c-a) = 0
        # So p is in the plane through a perpendicular to (b-a) and (c-a)? No...
        # Actually: p.(b-a) = 0 and p.(c-a) = 0 means p is perpendicular to both (b-a) and (c-a),
        # so p is parallel to the normal. Thus the 3D circumcenter is along the normal.
        
        # p = k * normal for some k
        # |p - a|^2 = |p - b|^2
        # |k*n - a|^2 = |k*n - b|^2
        # k^2 - 2k*n.a + |a|^2 = k^2 - 2k*n.b + |b|^2
        # -2k*n.a + 1 = -2k*n.b + 1
        # n.a = n.b (which should be true by symmetry of circumcenter)
        
        # Actually, let me just compute the circumradius directly.
        # For a triangle with vertices on unit sphere, the circumradius R (angular) satisfies:
        # The three points define a circle in 3D. The center of this circle is at distance
        # d from origin, and the radius of the circle is r = sqrt(1 - d^2) if the circle
        # is in a plane at distance d from origin. The angular circumradius is arccos(d/r * something)...
        
        # Simpler: use the formula for circumradius of a triangle given side lengths.
        # Side lengths (angular): 
        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
        
        # Spherical circumradius formula:
        # sin(R_c) = sin(ab) * sin(bc) * sin(ca) / (2 * sin(s) * sin(s-ab) * sin(s-bc) * sin(s-ca))
        # where s = (ab + bc + ca) / 2
        # This is the spherical law of sines applied to the circumradius.
        
        s = (ab + bc + ca) / 2.0
        denom = 2.0 * np.sin(s) * np.sin(s - ab) * np.sin(s - bc) * np.sin(s - ca)
        if abs(denom) < 1e-15:
            continue
        sin_rc = (np.sin(ab) * np.sin(bc) * np.sin(ca)) / denom
        sin_rc = np.clip(sin_rc, 0.0, 1.0)
        circum_rad = np.arcsin(sin_rc)
        
        if circum_rad > max_circum_rad:
            max_circum_rad = circum_rad
    
    return np.degrees(max_circum_rad)


def _fast_circumradius_hull(pts):
    """
    Faster approximate hull-based covering radius using only a subset of facets
    or a direct geometric approach. For robustness, we compute the max circumradius
    over all hull facets but with vectorized operations where possible.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        samples = _generate_fibonacci_samples(5000)
        return _covering_radius_sampled(pts, samples)
    
    max_circum_rad = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
        
        s = (ab + bc + ca) / 2.0
        denom = 2.0 * np.sin(s) * np.sin(s - ab) * np.sin(s - bc) * np.sin(s - ca)
        if abs(denom) < 1e-15:
            continue
        sin_rc = (np.sin(ab) * np.sin(bc) * np.sin(ca)) / denom
        sin_rc = np.clip(sin_rc, 0.0, 1.0)
        circum_rad = np.arcsin(sin_rc)
        
        if circum_rad > max_circum_rad:
            max_circum_rad = circum_rad
    
    return np.degrees(max_circum_rad)


def _perturb(pts, rng, step_size):
    """Perturb points by adding Gaussian noise and renormalizing."""
    noise = rng.normal(0, step_size, size=pts.shape)
    new_pts = pts + noise
    return _normalize(new_pts)


def _swap_pair(pts, rng):
    """Randomly swap two points (or rotate one towards another)."""
    i, j = rng.choice(N, size=2, replace=False)
    # Rotate point i slightly towards point j
    angle = rng.uniform(0.05, 0.3)
    # Find direction from i to j
    diff = pts[j] - pts[i]
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-10:
        return pts
    direction = diff / diff_norm
    # Rotate pts[i] towards direction by angle
    # Use Rodrigues' rotation formula
    axis = np.cross(pts[i], direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return pts
    axis = axis / axis_norm
    v = pts[i]
    v_dot_k = np.dot(v, axis)
    v_cross_k = np.cross(axis, v)
    rotated = (v * np.cos(angle) + v_cross_k * np.sin(angle) + axis * v_dot_k * (1 - np.cos(angle)))
    new_pts = pts.copy()
    new_pts[i] = rotated
    return _normalize(new_pts)


def _simulated_annealing(pts_init, samples, rng, start_time, deadline):
    """
    Simulated annealing to minimize covering radius.
    Uses fast sampled evaluation for inner loop, hull-based for final check.
    """
    pts = _normalize(pts_init).copy()
    best_pts = pts.copy()
    
    # Initial energy
    T = 1.0  # Initial temperature (in degrees)
    cooling_rate = 0.995
    min_T = 0.01
    
    best_energy = _covering_radius_sampled(pts, samples)
    
    step_size = 0.15
    
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Generate candidate
        move_type = rng.random()
        if move_type < 0.7:
            candidate = _perturb(pts, rng, step_size)
        else:
            candidate = _swap_pair(pts, rng)
        
        # Evaluate candidate
        candidate_energy = _covering_radius_sampled(candidate, samples)
        
        # Accept or reject
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate.copy()
            pts = candidate.copy()
            # Occasionally use hull-based evaluation for accuracy
            if rng.random() < 0.1:
                hull_energy = _fast_circumradius_hull(candidate)
                if hull_energy < best_energy:
                    best_energy = hull_energy
                    best_pts = candidate.copy()
        else:
            # Metropolis criterion
            delta = candidate_energy - best_energy
            if delta < 0 or rng.random() < np.exp(-delta / T):
                pts = candidate.copy()
        
        # Cool
        T *= cooling_rate
        step_size = max(0.01, step_size * 0.999)
    
    return best_pts, best_energy


def _generate_initial_points(method='fibonacci', rng=None):
    """Generate initial point configurations using different methods."""
    if method == 'fibonacci':
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)
    elif method == 'random':
        if rng is None:
            rng = np.random.default_rng(42)
        pts = rng.normal(size=(N, 3))
        return _normalize(pts)
    elif method == 'perturbed_fibonacci':
        base = _generate_initial_points('fibonacci', rng)
        if rng is None:
            rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.2, size=base.shape)
        return _normalize(base + noise)
    else:
        raise ValueError(f"Unknown method: {method}")


def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses simulated annealing with multiple restarts and hull-based verification.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Generate fixed samples for fast evaluation
    samples = _generate_fibonacci_samples(_N_SAMPLES_FAST)
    
    # Track best result
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Fibonacci init + simulated annealing
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        pts_init = _generate_initial_points('fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 15
        if remaining > 5:
            sa_pts, sa_radius = _simulated_annealing(pts_init, samples, rng, start_time, start_time + remaining)
            if sa_radius < best_radius:
                best_radius = sa_radius
                best_pts = sa_pts.copy()
    
    # Strategy 2: Random init + simulated annealing
    for i in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        pts_init = _generate_initial_points('random', rng)
        remaining = deadline - (time.time() - start_time) - 15
        if remaining < 5:
            break
        sa_pts, sa_radius = _simulated_annealing(pts_init, samples, rng, start_time, start_time + remaining)
        if sa_radius < best_radius:
            best_radius = sa_radius
            best_pts = sa_pts.copy()
    
    # Strategy 3: Perturbed Fibonacci + simulated annealing
    for i in range(3):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        pts_init = _generate_initial_points('perturbed_fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 15
        if remaining < 5:
            break
        sa_pts, sa_radius = _simulated_annealing(pts_init, samples, rng, start_time, start_time + remaining)
        if sa_radius < best_radius:
            best_radius = sa_radius
            best_pts = sa_pts.copy()
    
    # Final polish: use hull-based evaluation to refine
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Do a short SA run with hull-based evaluation for the best candidate
            # Use a very short run just to verify and potentially improve
            remaining = deadline - elapsed - 5
            if remaining > 3:
                # Short SA with hull evaluation
                pts = best_pts.copy()
                T = 0.5
                cooling_rate = 0.98
                step_size = 0.05
                for it in range(100):
                    if time.time() > start_time + remaining:
                        break
                    candidate = _perturb(pts, rng, step_size)
                    candidate_energy = _fast_circumradius_hull(candidate)
                    if candidate_energy < best_radius:
                        best_radius = candidate_energy
                        best_pts = candidate.copy()
                        pts = candidate.copy()
                    else:
                        delta = candidate_energy - best_radius
                        if delta < 0 or rng.random() < np.exp(-delta / T):
                            pts = candidate.copy()
                    T *= cooling_rate
                    step_size = max(0.01, step_size * 0.99)
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _generate_initial_points('fibonacci', rng)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
