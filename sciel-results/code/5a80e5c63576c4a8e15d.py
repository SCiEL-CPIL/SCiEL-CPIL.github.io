"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]


def _covering_radius_fast(pts, n_samples=50000):
    """Compute covering radius using a dense Fibonacci spiral sample."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    max_angle = np.max(angles)
    return max_angle * 180.0 / np.pi


def _covering_radius_exact(pts):
    """Compute exact covering radius using circumradii of Delaunay triangles on sphere."""
    try:
        hull = ConvexHull(pts)
        tri_idx = hull.simplices
        
        max_angle = 0.0
        for tri in tri_idx:
            p1, p2, p3 = pts[tri]
            
            # Compute the circumcenter of the triangle in 3D
            # The circumcenter C satisfies |C - p1| = |C - p2| = |C - p3|
            # and lies in the plane of the triangle.
            # We can find it by solving the linear system.
            
            # Edge vectors
            e1 = p2 - p1
            e2 = p3 - p1
            
            # The circumcenter is p1 + a*e1 + b*e2 where:
            # 2*a*(e1.e1) + 2*b*(e1.e2) = |p2|^2 - |p1|^2 = 0 (unit sphere)
            # 2*a*(e2.e1) + 2*b*(e2.e2) = |p3|^2 - |p1|^2 = 0
            # Wait, that's for the case where all points are on a sphere centered at origin.
            # Actually, for points on the unit sphere, the circumcenter of the triangle
            # (in 3D) is NOT the same as the point on the unit sphere equidistant from them.
            
            # Let me think differently. The covering radius of a Voronoi cell on the sphere
            # is the maximum angular distance from any point in the cell to the nearest site.
            # The worst point is either:
            # 1. The circumcenter of the spherical triangle (if it lies inside)
            # 2. A point on an edge (midpoint of the longest edge, if circumcenter is outside)
            
            # For a spherical triangle, the circumcenter is the point on the sphere
            # equidistant from all three vertices. It's the normalization of the vector
            # that is equiangular to p1, p2, p3.
            
            # The circumcenter direction can be found as the normalized vector:
            # C_dir = (p1/sin(A) + p2/sin(B) + p3/sin(C)) ... no, that's not right either.
            
            # Better approach: The circumcenter of a spherical triangle is the pole of
            # the plane containing the three points, but only if the triangle is "small".
            # Actually, the normal to the plane through p1, p2, p3 is n = (p2-p1) x (p3-p1).
            # The circumcenter on the sphere is in the direction of n (or -n), but this
            # is only correct if the triangle is acute.
            
            # Let's use a more robust method: find the point on the sphere that maximizes
            # the minimum distance to p1, p2, p3, constrained to the spherical triangle.
            # This is equivalent to finding the circumcenter.
            
            # The circumcenter of three points on the unit sphere can be computed as:
            # Solve for x such that x . p1 = x . p2 = x . p3 = cos(r) and |x| = 1.
            # From x . p1 = x . p2: x . (p1 - p2) = 0
            # From x . p1 = x . p3: x . (p1 - p3) = 0
            # So x is perpendicular to both (p1-p2) and (p1-p3), i.e., x is parallel to
            # (p1-p2) x (p1-p3).
            
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n /= n_norm
            
            # The circumcenter is either n or -n. Choose the one that is "inside" the triangle.
            # Check which one gives a smaller maximum angular distance.
            for candidate in [n, -n]:
                dots = np.array([np.dot(candidate, p1), np.dot(candidate, p2), np.dot(candidate, p3)])
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                max_ang = np.max(angles)
                
                # Check if the circumcenter lies inside the spherical triangle.
                # A point is inside a spherical triangle if it is on the same side of
                # each edge as the opposite vertex.
                # For edge p1-p2, the normal is (p1 x p2) normalized. The point is on the
                # correct side if dot(candidate, p1 x p2) has the same sign as dot(p3, p1 x p2).
                
                def _inside_spherical_triangle(pt, a, b, c):
                    # Check if pt is inside spherical triangle abc
                    # For each edge, check the sign of the cross product
                    e1 = np.cross(a, b)
                    e2 = np.cross(b, c)
                    e3 = np.cross(c, a)
                    
                    s1 = np.dot(pt, e1) * np.dot(c, e1)
                    s2 = np.dot(pt, e2) * np.dot(a, e2)
                    s3 = np.dot(pt, e3) * np.dot(b, e3)
                    
                    return s1 >= -1e-10 and s2 >= -1e-10 and s3 >= -1e-10
                
                if _inside_spherical_triangle(candidate, p1, p2, p3):
                    if max_ang > max_angle:
                        max_angle = max_ang
            
            # Also check edge midpoints in case circumcenter is outside
            for i_edge, j_edge in [(0, 1), (1, 2), (2, 0)]:
                pe = (pts[tri[i_edge]] + pts[tri[j_edge]])
                pe = _normalize(pe)
                dots = np.array([np.dot(pe, p1), np.dot(pe, p2), np.dot(pe, p3)])
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                max_ang = np.max(angles)
                if max_ang > max_angle:
                    max_angle = max_ang
        
        return max_angle * 180.0 / np.pi
    except:
        return _covering_radius_fast(pts, n_samples=100000)


def _repulsive_forces(pts, k=5, strength=1.0):
    """Compute repulsive forces for each point based on nearest neighbors."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Compute pairwise distances (angular)
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    for i in range(n):
        # Get k nearest neighbors
        sorted_idx = np.argsort(dists[i])
        sorted_idx = sorted_idx[1:k+1]  # Exclude self
        
        force = np.zeros(3)
        for j in sorted_idx:
            if i == j:
                continue
            # Repulsive force: direction from j to i, scaled by inverse distance
            diff = pts[i] - pts[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            diff_dir = diff / diff_norm
            # Scale by inverse of angular distance (stronger repulsion when closer)
            ang_dist = dists[i, j]
            if ang_dist < 1e-10:
                scale = strength * 10.0
            else:
                scale = strength / (ang_dist + 0.01)
            force += scale * diff_dir
        
        forces[i] = force
    
    return forces


def _local_optimize(pts, max_time=30.0):
    """Local optimization using a combination of random perturbations and gradient-like steps."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    step_size = 0.03
    n_iter = 0
    patience = 0
    
    while time.time() - start_time < max_time - 5.0:
        n_iter += 1
        improved = False
        
        # Compute repulsive forces to guide the search
        forces = _repulsive_forces(pts, k=3, strength=0.5)
        
        for i in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Try several perturbations: random + force-guided
            candidates = []
            
            # Random directions
            for trial in range(5):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                candidates.append(direction)
            
            # Force-guided direction
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-10:
                candidates.append(forces[i] / f_norm)
            
            for direction in candidates:
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius_fast(new_pts, n_samples=20000)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            patience += 1
            step_size *= 0.9
            if step_size < 1e-5 or patience > 50:
                break
        else:
            patience = 0
    
    return best_pts, best_cr


def _hill_climb(pts, max_time=20.0):
    """Pure hill-climbing optimization."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    step_size = 0.02
    
    while time.time() - start_time < max_time - 3.0:
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 3.0:
                break
            
            for trial in range(8):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius_fast(new_pts, n_samples=15000)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.85
            if step_size < 1e-5:
                break
    
    return best_pts, best_cr


def _fibonacci_spiral(n):
    """Generate points on a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    pts = rng.randn(n, 3)
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    best_pts = None
    best_cr = float('inf')
    
    # Multiple starting configurations
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # Random starts with different seeds
    for seed in [0, 1, 2, 3, 4]:
        starts.append(_random_points(N, seed=seed))
    
    # Optimized random: start with Fibonacci, then perturb randomly
    fib = _fibonacci_spiral(N)
    perturbed = fib + 0.1 * np.random.randn(N, 3)
    starts.append(_normalize(perturbed))
    
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        start_pts = _normalize(start_pts)
        
        # Quick evaluation
        cr = _covering_radius_fast(start_pts, n_samples=20000)
        
        # Local optimization
        remaining = deadline - 20.0 - time.time()
        if remaining < 10.0:
            break
        
        opt_pts, opt_cr = _local_optimize(start_pts, max_time=min(remaining, 40.0))
        
        if opt_cr < best_cr:
            best_cr = opt_cr
            best_pts = opt_pts
    
    # Final polish with exact radius computation if time permits
    if best_pts is not None and time.time() < deadline - 10.0:
        # Do a few more iterations with exact radius
        polish_pts = best_pts.copy()
        polish_cr = _covering_radius_exact(polish_pts)
        
        if polish_cr < best_cr:
            best_cr = polish_cr
            best_pts = polish_pts
        
        # Quick hill climb with exact radius
        step_size = 0.01
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = polish_pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius_exact(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    polish_pts = new_pts.copy()
                    improved = True
                    break
            if not improved:
                step_size *= 0.8
                if step_size < 1e-5:
                    break
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
