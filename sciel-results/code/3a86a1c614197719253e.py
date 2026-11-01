import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
TIME_LIMIT = 150.0
MARGIN = 8.0


def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _make_grid(n_theta=120, n_phi=240):
    """Create a dense grid of points on the unit sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _covering_radius_hull(points):
    """Exact covering radius via convex hull facets (for evaluation)."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    max_angle = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        e1, e2 = v2 - v1, v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        facet_r = max(d1, d2, d3)
        # Check cap emptiness
        dots = points @ normal
        mask = np.ones(N, dtype=bool)
        mask[simplex] = False
        if np.max(dots[mask]) > np.cos(facet_r) + 1e-10:
            continue
        max_angle = max(max_angle, facet_r)
    return max_angle


def _covering_radius_grid(points, grid):
    """Approximate covering radius using a grid of test points."""
    # For each grid point, find min angular distance to any config point
    # d(y, x) = arccos(y.x), so min d = arccos(max x)
    dots = grid @ points.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    angles = np.arccos(np.clip(max_dots, -1, 1))
    return np.max(angles)


def _smooth_covering_radius(points, grid, beta=50.0):
    """
    Differentiable smooth surrogate for the covering radius.
    Uses log-sum-exp to smooth the max over grid points.
    Returns (value, gradient w.r.t. points).
    """
    # dots[i, j] = grid[i] . points[j]
    dots = grid @ points.T  # (M, N)
    # For each grid point i, max_j dots[i,j] gives cos of min angle
    # We want to maximize min angle = minimize max angle
    # max angle = arccos(min_j cos(d_i, p_j)) ... wait
    # min angle to nearest point: min_j arccos(d_i . p_j) = arccos(max_j d_i . p_j)
    # So covering radius = max_i arccos(max_j d_i . p_j)
    # Let m_i = max_j d_i . p_j. Then CR = max_i arccos(m_i)
    # Smooth: logsumexp over i of arccos(m_i) with beta
    # But arccos is not smooth at +-1. Use a smoother.
    
    # Alternative: minimize -logsumexp(-beta * angles_i) where angles_i = arccos(m_i)
    # Actually, let's use: f = logsumexp(beta * angles) / beta as smooth max
    # angles_i = arccos(m_i)
    
    m = np.max(dots, axis=1)  # (M,)
    m = np.clip(m, -1 + 1e-12, 1 - 1e-12)
    angles = np.arccos(m)  # (M,)
    
    # Smooth max via logsumexp
    max_angle = np.max(angles)
    shifted = beta * (angles - max_angle)
    logsumexp = max_angle + np.log(np.sum(np.exp(shifted))) / beta
    
    # Gradient: d/dp_k of logsumexp
    # logsumexp = max_a + log(sum(exp(beta*(a_i - max_a))))/beta
    # d logsumexp / da_i = (1/beta) * exp(beta*(a_i - max_a)) / sum(exp(beta*(a_j - max_a)))
    # = exp(beta*(a_i - max_a)) / sum(exp(beta*(a_j - max_a)))
    # da_i / dm_i = -1/sqrt(1-m_i^2)
    # dm_i / dp_k = grid[i] if k is argmax... but we need soft version
    
    # Actually, m_i = max_j d_i.p_j, which is not differentiable.
    # Use soft max: m_i = logsumexp(beta2 * d_i.p) / beta2
    beta2 = 20.0
    d_shifted = beta2 * (dots - np.max(dots, axis=1, keepdims=True))
    soft_m = np.max(dots, axis=1) + np.log(np.sum(np.exp(d_shifted), axis=1)) / beta2
    soft_m = np.clip(soft_m, -1 + 1e-12, 1 - 1e-12)
    angles = np.arccos(soft_m)
    
    max_a = np.max(angles)
    w = np.exp(beta * (angles - max_a))
    w /= np.sum(w)  # (M,) weights for smooth max
    
    # d logsumexp / d angles_i = w_i
    # d angles_i / d soft_m_i = -1/sqrt(1 - soft_m_i^2)
    ds_dm = -1.0 / np.sqrt(1.0 - soft_m**2 + 1e-12)  # (M,)
    
    # d soft_m_i / d dots[i,k] = exp(beta2*(dots[i,k]-max_i)) / sum_j exp(beta2*(dots[i,j]-max_i))
    # This is the softmax weights for the inner max
    d_shifted2 = beta2 * (dots - np.max(dots, axis=1, keepdims=True))
    exp_d = np.exp(d_shifted2)
    softmax_w = exp_d / np.sum(exp_d, axis=1, keepdims=True)  # (M, N)
    
    # d soft_m_i / d p_k = sum_i w_i * ds_dm_i * softmax_w[i,k] * grid[i]
    # = sum_i (w_i * ds_dm_i * softmax_w[i]) . grid[i] for each k
    # grad_k = sum_i w_i * ds_dm_i * softmax_w[i,k] * grid[i]
    # In matrix form: grad = (softmax_w.T * (w * ds_dm)) @ grid
    # softmax_w.T is (N, M), (w*ds_dm) is (M,), so (N,M)*(M,) -> (N,M) elementwise? No.
    # We want: grad[k] = sum_i (w_i * ds_dm_i * softmax_w[i,k]) * grid[i]
    # = sum_i (w_i * ds_dm_i) * softmax_w[i,k] * grid[i]
    # Let a_i = w_i * ds_dm_i  (M,)
    # grad = (softmax_w.T * a[None, :]) @ grid  -> (N,M) @ (M,3) = (N,3)
    
    a = w * ds_dm  # (M,)
    grad = (softmax_w.T * a[None, :]) @ grid  # (N, 3)
    
    return logsumexp, grad


def _riemannian_optimize(points, grid, beta=50.0, max_iters=2000, deadline=None):
    """
    Riemannian gradient descent with momentum on the sphere.
    """
    points = points.copy().astype(float)
    n = len(points)
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    best_val = _covering_radius_grid(points, grid)
    best_pts = points.copy()
    no_improve = 0
    best_lr = lr
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = _smooth_covering_radius(points, grid, beta=beta)
        
        # Project gradient to tangent space
        radial = (grad * points).sum(axis=1, keepdims=True) * points
        tangent_grad = grad - radial
        
        # Momentum update
        velocity = momentum * velocity - lr * tangent_grad
        
        # Project velocity to tangent space
        vel_radial = (velocity * points).sum(axis=1, keepdims=True) * points
        velocity = velocity - vel_radial
        
        # Update
        points = points + velocity
        
        # Renormalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Check improvement periodically
        if it % 20 == 0:
            cr = _covering_radius_grid(points, grid)
            if cr < best_val - 1e-5:
                best_val = cr
                best_pts = points.copy()
                no_improve = 0
                # Increase beta to sharpen
                beta = min(beta * 1.05, 200.0)
            else:
                no_improve += 1
                if no_improve > 5:
                    lr *= 0.7
                    momentum = min(momentum + 0.05, 0.99)
                    no_improve = 0
                    if lr < 1e-5:
                        break
    
    return best_pts


def _lloyd_spherical(points, grid, max_iters=50, deadline=None):
    """
    Lloyd-type iteration: assign grid points to nearest config point,
    move each config point to the centroid (spherical mean) of its assigned points.
    """
    points = points.copy().astype(float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Assign each grid point to nearest config point
        dots = grid @ points.T  # (M, N)
        assignments = np.argmax(dots, axis=1)  # (M,)
        
        new_points = np.zeros_like(points)
        for k in range(len(points)):
            mask = assignments == k
            if np.sum(mask) > 0:
                # Spherical mean
                mean_vec = np.sum(grid[mask], axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-10:
                    new_points[k] = mean_vec / norm
                else:
                    new_points[k] = points[k]
            else:
                new_points[k] = points[k]
        
        # Check for degeneracy
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if points are too close (degenerate)
        min_dist = np.inf
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                d = np.arccos(np.clip(np.dot(new_points[i], new_points[j]), -1, 1))
                min_dist = min(min_dist, d)
        
        if min_dist < 1e-4:
            # Revert to avoid degeneracy
            break
        
        points = new_points
    
    return points


def _local_facet_optimize(points, max_iters=100, deadline=None):
    """
    Local optimization: for each facet of the convex hull, try to move
    the vertices to reduce the circumradius.
    """
    points = points.copy().astype(float)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        improved = False
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            
            d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
            d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
            facet_r = max(d1, d2, d3)
            
            # Check cap emptiness
            dots = points @ normal
            mask = np.ones(N, dtype=bool)
            mask[simplex] = False
            if np.max(dots[mask]) > np.cos(facet_r) + 1e-10:
                continue
            
            # Try small perturbations of the three vertices
            for idx in simplex:
                for trial in range(5):
                    # Random tangent direction
                    rand_vec = np.random.randn(3)
                    tang = rand_vec - np.dot(rand_vec, points[idx]) * points[idx]
                    tnorm = np.linalg.norm(tang)
                    if tnorm < 1e-10:
                        continue
                    tang /= tnorm
                    
                    # Try moving in both directions
                    for sign in [1.0, -1.0]:
                        step = 0.01 * sign
                        trial_pts = points.copy()
                        trial_pts[idx] = points[idx] + step * tang
                        trial_pts[idx] /= np.linalg.norm(trial_pts[idx])
                        
                        r_trial = _covering_radius_hull(trial_pts)
                        r_current = _covering_radius_hull(points)
                        if r_trial < r_current - 1e-6:
                            points = trial_pts
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - MARGIN
    
    # Generate grid for smooth optimization
    grid = _make_grid(n_theta=80, n_phi=160)  # 12800 points
    
    # Initial configurations
    inits = []
    inits.append(_fibonacci_sphere(N, offset=0.5))
    inits.append(_fibonacci_sphere(N, offset=0.1))
    
    rng = np.random.default_rng(42)
    for _ in range(4):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_points = None
    best_radius = np.pi
    
    # Evaluate initial configs
    for pts in inits:
        r = _covering_radius_grid(pts, grid)
        if r < best_radius:
            best_radius = r
            best_points = pts.copy()
        if time.time() > deadline - 20:
            break
    
    if best_points is None:
        best_points = inits[0]
    
    # Phase 1: Riemannian gradient descent from best initial
    if time.time() < deadline - 15:
        opt_pts = _riemannian_optimize(best_points, grid, beta=30.0, max_iters=1500, deadline=deadline - 10)
        r = _covering_radius_grid(opt_pts, grid)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
    
    # Phase 2: Lloyd spherical refinement
    if time.time() < deadline - 10:
        lloyd_pts = _lloyd_spherical(best_points, grid, max_iters=30, deadline=deadline - 8)
        r = _covering_radius_grid(lloyd_pts, grid)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
    
    # Phase 3: More Riemannian with higher beta (sharper)
    if time.time() < deadline - 8:
        opt_pts = _riemannian_optimize(best_points, grid, beta=80.0, max_iters=800, deadline=deadline - 5)
        r = _covering_radius_grid(opt_pts, grid)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
    
    # Phase 4: Local facet optimization
    if time.time() < deadline - 5:
        facet_pts = _local_facet_optimize(best_points, max_iters=50, deadline=deadline - 3)
        r = _covering_radius_hull(facet_pts)
        if r < best_radius:
            best_radius = r
            best_points = facet_pts.copy()
    
    # Phase 5: Final fine Riemannian
    if time.time() < deadline - 3:
        opt_pts = _riemannian_optimize(best_points, grid, beta=150.0, max_iters=500, deadline=deadline - 2)
        r = _covering_radius_hull(opt_pts)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
    
    # Final check and normalization
    if best_points is None:
        best_points = inits[0]
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-8:
                # Separate them slightly
                best_points[j] += 1e-6 * np.random.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points


if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    r = _covering_radius_hull(pts)
    print(f"Covering radius: {np.degrees(r):.6f} degrees")
