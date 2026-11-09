"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _make_grid(n_grid):
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _cover_radius_grid(pts, n_grid=8000):
    """Compute covering radius using a Fibonacci grid."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    grid = _make_grid(n_grid)
    dots = grid @ pts.T
    min_dots = np.max(dots, axis=1)
    return np.arccos(np.clip(min_dots, -1.0, 1.0)).max()

def _hull_covering_radius(pts):
    """
    Compute covering radius from convex hull facets.
    For each triangular facet, compute the circumradius (angular distance
    from facet center to its vertices on the sphere).
    The covering radius is the max over all facets.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: invalid hull
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # simplex has 3 vertices for a sphere hull
        if len(simplex) != 3:
            continue
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Normal to the plane through v1, v2, v3
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        
        # The circumcenter on the sphere is in the direction of n (or -n)
        # We need to determine which direction. The center should be on the same
        # side as the origin for a convex hull on the unit sphere... actually,
        # for points on the unit sphere, the circumcenter of a facet is the
        # point on the sphere that is equidistant from the 3 vertices.
        
        # For 3 points on the unit sphere, the circumcenter of the
        # spherical triangle is the intersection of the sphere with the line
        # through the origin and the circumcenter of the planar triangle.
        
        # Planar circumcenter of triangle v1, v2, v3:
        # Solve: |c - v1|^2 = |c - v2|^2 = |c - v3|^2
        # This gives c in the plane of v1, v2, v3.
        
        # Alternative: the circumradius of the spherical triangle can be computed
        # from the side lengths (angular distances between vertices).
        
        # Side lengths (angular distances)
        d12 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        d23 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
        d31 = np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0))
        
        # Spherical circumradius formula:
        # For a spherical triangle with sides a, b, c (angular),
        # the circumradius R satisfies:
        # cos(R) = (cos(a)cos(b)cos(c) + 2cos(a)cos(b)cos(c)) / (cos^2(a)+cos^2(b)+cos^2(c) - 2cos(a)cos(b)cos(c) - 1)
        # Actually, let me use a more direct approach.
        
        # The circumcenter is the point equidistant from v1, v2, v3 on the sphere.
        # Let c be the circumcenter. Then |c - v1| = |c - v2| = |c - v3| (in 3D Euclidean).
        # Since c, v1, v2, v3 are all on the unit sphere:
        # |c - v1|^2 = 2 - 2*c·v1
        # So c·v1 = c·v2 = c·v3
        # This means c is in the null space of [v1-v2; v2-v3] (in the sense that
        # (v1-v2)·c = 0 and (v2-v3)·c = 0), so c is parallel to n = (v1-v2)×(v2-v3).
        
        # So c = ±n / |n| (normalized to unit length)
        # We need to pick the sign such that c is "outside" the triangle (toward the
        # center of the cap). For a convex hull on the sphere, the circumcenter of
        # a facet should be on the side of the facet away from the interior.
        
        # The normal n points outward from the hull (by ConvexHull convention).
        # The circumcenter should be in the direction of n (outward).
        
        c = n  # already normalized
        
        # Verify: check that c·v1 ≈ c·v2 ≈ c·v3
        # The circumradius is arccos(c·v1)
        
        dot_vals = np.array([np.dot(c, v1), np.dot(c, v2), np.dot(c, v3)])
        # They should be approximately equal
        if np.max(dot_vals) - np.min(dot_vals) > 0.1:
            # Try the other direction
            c = -n
            dot_vals = np.array([np.dot(c, v1), np.dot(c, v2), np.dot(c, v3)])
        
        r = np.arccos(np.clip(dot_vals[0], -1.0, 1.0))
        if r > max_radius:
            max_radius = r
    
    return max_radius

def _lse_objective_and_grad(x, grid, beta):
    """
    Log-sum-exp surrogate for the covering radius.
    Returns (objective, gradient).
    """
    p = x.reshape(-1, 3)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    dots = grid @ p.T
    max_dots = np.max(dots, axis=1)
    
    max_dots_clipped = np.clip(max_dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angular_dists = np.arccos(max_dots_clipped)
    
    max_ad = np.max(angular_dists)
    lse = max_ad + np.log(np.sum(np.exp(beta * (angular_dists - max_ad))))
    
    weights = beta * np.exp(beta * (angular_dists - max_ad))
    weights /= np.sum(weights)
    
    denom = np.sqrt(np.clip(1.0 - max_dots * max_dots, 1e-15, None))
    dad = -weights / denom
    
    argmax_j = np.argmax(dots, axis=1)
    
    grad = np.zeros_like(x)
    for i in range(len(grid)):
        j = argmax_j[i]
        grad[j*3:(j+1)*3] += dad[i] * grid[i]
    
    return lse, grad

def _voronoi_centroid_move(pts, grid, beta=5.0, step_size=0.1):
    """
    Move each point toward the weighted centroid of its Voronoi cell (grid points assigned to it).
    This is a structural change from the facet-based approach.
    """
    p = pts.copy()
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / norms
    
    dots = grid @ p.T
    nearest_idx = np.argmax(dots, axis=1)
    
    cos_theta = dots[np.arange(len(grid)), nearest_idx]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    dists = np.arccos(cos_theta)
    
    R_max = np.max(dists)
    
    # Weight by distance to emphasize far points
    weights = np.exp(dists / (R_max + 1e-10) * beta)
    
    target_vecs = np.zeros((N, 3))
    weights_sum = np.zeros(N)
    
    np.add.at(target_vecs, nearest_idx, grid * weights[:, np.newaxis])
    np.add.at(weights_sum, nearest_idx, weights)
    
    new_pts = p.copy()
    for i in range(N):
        if weights_sum[i] > 1e-10:
            centroid_dir = target_vecs[i] / np.linalg.norm(target_vecs[i])
            new_pts[i] = p[i] + step_size * centroid_dir
            new_pts[i] /= np.linalg.norm(new_pts[i])
        else:
            # Should not happen, but if so, jitter
            new_pts[i] += 1e-5 * np.random.randn(3)
            new_pts[i] /= np.linalg.norm(new_pts[i])
    
    return new_pts

def _facet_centroid_move(pts, step_size=0.05):
    """
    Move each point toward the circumcenter of its worst facet.
    For each point, find the facet where it's a vertex and has the largest
    circumradius, then move the point toward the circumcenter of that facet.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts.copy()
    
    new_pts = pts.copy()
    
    # For each facet, compute circumcenter and circumradius
    facet_data = []
    for idx, simplex in enumerate(hull.simplices):
        if len(simplex) != 3:
            continue
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        
        dot_vals = np.array([np.dot(n, v1), np.dot(n, v2), np.dot(n, v3)])
        if np.max(dot_vals) - np.min(dot_vals) > 0.1:
            n = -n
            dot_vals = -dot_vals
        
        r = np.arccos(np.clip(dot_vals[0], -1.0, 1.0))
        facet_data.append((idx, simplex, n, r))
    
    if not facet_data:
        return new_pts
    
    # For each point, find its worst facet
    point_worst = {}
    for idx, simplex, center, r in facet_data:
        for vi in simplex:
            if vi not in point_worst or r > point_worst[vi][1]:
                point_worst[vi] = (center, r)
    
    # Move points toward their worst facet's circumcenter
    for k in range(N):
        if k in point_worst:
            center, r = point_worst[k]
            new_pts[k] = pts[k] + step_size * (center - pts[k])
            new_pts[k] /= np.linalg.norm(new_pts[k])
    
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute grids
    grid_coarse = _make_grid(3000)
    grid_fine = _make_grid(6000)
    grid_vf = _make_grid(10000)
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_sphere(N))
    
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_pts = None
    best_radius = np.inf
    
    for init_idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
        
        current_pts = init_pts.copy()
        current_radius = _cover_radius_grid(current_pts, n_grid=5000)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        # Phase 1: L-BFGS with LSE surrogate (coarse)
        if time.time() > deadline - 18.0:
            break
        
        beta_coarse = 20.0
        def obj_grad_coarse(x):
            return _lse_objective_and_grad(x, grid_coarse, beta_coarse)
        
        try:
            result = minimize(
                obj_grad_coarse,
                current_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 300}
            )
            new_pts = result.x.reshape(-1, 3)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _cover_radius_grid(new_pts, n_grid=5000)
            if new_radius < current_radius:
                current_pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
        except Exception:
            pass
        
        # Phase 2: Voronoi centroid moves (new structural approach)
        if time.time() > deadline - 15.0:
            break
        
        for step in [0.1, 0.05, 0.03, 0.02, 0.01]:
            if time.time() > deadline - 12.0:
                break
            for _ in range(5):
                if time.time() > deadline - 11.0:
                    break
                new_pts = _voronoi_centroid_move(current_pts, grid_fine, beta=5.0, step_size=step)
                new_radius = _cover_radius_grid(new_pts, n_grid=5000)
                if new_radius < current_radius:
                    current_pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()
                else:
                    break
        
        # Phase 3: L-BFGS with LSE surrogate (fine)
        if time.time() > deadline - 10.0:
            break
        
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 8.0:
                break
            
            def obj_grad_fine(x, grid=grid_fine, beta=beta):
                return _lse_objective_and_grad(x, grid, beta)
            
            try:
                result = minimize(
                    obj_grad_fine,
                    current_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _cover_radius_grid(new_pts, n_grid=8000)
                if new_radius < current_radius:
                    current_pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()
            except Exception:
                pass
        
        # Phase 4: More Voronoi centroid moves with finer step
        if time.time() > deadline - 5.0:
            break
        
        for step in [0.01, 0.005]:
            if time.time() > deadline - 4.0:
                break
            for _ in range(5):
                if time.time() > deadline - 3.5:
                    break
                new_pts = _voronoi_centroid_move(current_pts, grid_fine, beta=5.0, step_size=step)
                new_radius = _cover_radius_grid(new_pts, n_grid=8000)
                if new_radius < current_radius:
                    current_pts = new_pts
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()
                else:
                    break
    
    # Final polish
    if best_pts is not None and time.time() < deadline - 5.0:
        for beta in [200.0, 400.0]:
            if time.time() > deadline - 3.0:
                break
            
            def obj_grad_vf(x, grid=grid_vf, beta=beta):
                return _lse_objective_and_grad(x, grid, beta)
            
            try:
                result = minimize(
                    obj_grad_vf,
                    best_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100}
                )
                new_pts = result.x.reshape(-1, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _cover_radius_grid(new_pts, n_grid=12000)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass
        
        # Final Voronoi centroid moves
        if time.time() < deadline - 2.0:
            for step in [0.005, 0.002]:
                if time.time() > deadline - 1.0:
                    break
                for _ in range(5):
                    if time.time() > deadline - 0.5:
                        break
                    new_pts = _voronoi_centroid_move(best_pts, grid_vf, beta=5.0, step_size=step)
                    new_radius = _cover_radius_grid(new_pts, n_grid=12000)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_pts = new_pts.copy()
                    else:
                        break
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-6:
                # Perturb
                best_pts[j] += np.random.randn(3) * 0.01
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
