import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _spherical_circumcenters(hull, points):
    """
    Compute circumcenters and radii for all facets of the convex hull.
    Returns:
        centers: (M, 3) array of circumcenters (unit vectors)
        radii: (M,) array of angular radii (radians)
    """
    facets = hull.simplices
    centers = np.zeros((len(facets), 3))
    radii = np.zeros(len(facets))
    
    for i, face in enumerate(facets):
        v0 = points[face[0]]
        v1 = points[face[1]]
        v2 = points[face[2]]
        
        n = np.cross(v1 - v0, v2 - v1)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            centers[i] = v0
            radii[i] = 0
            continue
        
        center = n / norm
        
        eq = hull.equations[i]
        hull_normal = eq[:3]
        
        if np.dot(center, hull_normal) < 0:
            center = -center
            
        centers[i] = center
        radii[i] = np.arccos(np.clip(np.dot(center, v0), -1.0, 1.0))
        
    return centers, radii

def _covering_radius(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    centers, radii = _spherical_circumcenters(hull, points)
    return np.max(radii)

def _soft_max_radius(points, beta=50.0):
    """
    Smooth surrogate for max facet radius using log-sum-exp.
    Returns (soft_max_value, gradients w.r.t. points)
    Uses numerical gradient for the center-dependence term as an approximation.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    centers, radii = _spherical_circumcenters(hull, points)
    
    # Log-sum-exp for soft max
    max_r = np.max(radii)
    # Shift for numerical stability
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    weights /= weights.sum()
    soft_val = max_r - np.log(weights.sum()) / beta
    
    # Gradient: d(soft_max)/d(point_j)
    grads = np.zeros_like(points)
    facets = hull.simplices
    
    for i in range(len(facets)):
        w = weights[i]
        if w < 1e-15:
            continue
        face = facets[i]
        center = centers[i]
        r = radii[i]
        v0 = points[face[0]]
        
        # Direct term: d(arccos(center . vj))/dvj = -center / sqrt(1 - (center.vj)^2)
        for j in face:
            vj = points[j]
            dot_cj = np.dot(center, vj)
            sin_rj = np.sqrt(max(1.0 - dot_cj * dot_cj, 1e-15))
            grads[j] += w * (-center / sin_rj)
    
    # Numerical gradient for center-dependence: perturb each vertex and recompute
    # This captures how the circumcenter changes when vertices move
    eps = 1e-5
    for j in range(N):
        if time.time() - 0 > 100:  # safety
            break
        orig_vj = points[j].copy()
        for d in range(3):
            points[j][d] += eps
            # Renormalize
            pts_norm = np.linalg.norm(points[j])
            if pts_norm > 1e-10:
                points[j] /= pts_norm
            try:
                hull2 = ConvexHull(points)
                centers2, radii2 = _spherical_circumcenters(hull2, points)
                # Recompute soft max with new radii
                max_r2 = np.max(radii2)
                shifted2 = radii2 - max_r2
                weights2 = np.exp(beta * shifted2)
                weights2 /= weights2.sum()
                soft_val2 = max_r2 - np.log(weights2.sum()) / beta
            except Exception:
                points[j] = orig_vj
                continue
            points[j] = orig_vj
            # Central difference
            points[j][d] -= 2 * eps
            pts_norm = np.linalg.norm(points[j])
            if pts_norm > 1e-10:
                points[j] /= pts_norm
            try:
                hull3 = ConvexHull(points)
                centers3, radii3 = _spherical_circumcenters(hull3, points)
                max_r3 = np.max(radii3)
                shifted3 = radii3 - max_r3
                weights3 = np.exp(beta * shifted3)
                weights3 /= weights3.sum()
                soft_val3 = max_r3 - np.log(weights3.sum()) / beta
            except Exception:
                points[j] = orig_vj
                continue
            points[j] = orig_vj
            # Add numerical gradient contribution
            if soft_val2 != soft_val3:
                num_grad = (soft_val2 - soft_val3) / (2 * eps)
                # This is the gradient of the soft max w.r.t. points[j][d]
                # We need to project it onto the tangent space
                tangent_comp = num_grad - np.dot(points[j], np.zeros(3)) * 0  # simplified
                grads[j][d] += num_grad
            else:
                # One-sided
                points[j][d] += eps
                pts_norm = np.linalg.norm(points[j])
                if pts_norm > 1e-10:
                    points[j] /= pts_norm
                try:
                    hull2 = ConvexHull(points)
                    centers2, radii2 = _spherical_circumcenters(hull2, points)
                    max_r2 = np.max(radii2)
                    shifted2 = radii2 - max_r2
                    weights2 = np.exp(beta * shifted2)
                    weights2 /= weights2.sum()
                    soft_val2 = max_r2 - np.log(weights2.sum()) / beta
                except Exception:
                    points[j] = orig_vj
                    continue
                points[j] = orig_vj
                num_grad = (soft_val2 - soft_val) / eps
                grads[j][d] += num_grad
    
    return soft_val, grads

def _optimize_lbfgs(points, timeout=20.0, beta_schedule=None):
    """
    Optimize using L-BFGS-B on the soft-max surrogate.
    beta_schedule: list of (beta, maxiter) tuples for adaptive beta
    """
    start_time = time.time()
    points = points.copy().flatten()
    
    if beta_schedule is None:
        beta_schedule = [(30.0, 100), (50.0, 100), (80.0, 50)]
    
    current_x = points.copy()
    
    for beta, maxiter in beta_schedule:
        if time.time() - start_time > timeout:
            break
        
        def objective(x, b=beta):
            if time.time() - start_time > timeout:
                raise Exception("timeout")
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1
            pts /= norms
            val, _ = _soft_max_radius(pts, beta=b)
            return val
        
        def gradient(x, b=beta):
            if time.time() - start_time > timeout:
                raise Exception("timeout")
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1
            pts /= norms
            _, grads = _soft_max_radius(pts, beta=b)
            # Project gradient onto tangent space of sphere
            proj_grads = np.zeros_like(pts)
            for i in range(N):
                pi = pts[i]
                gi = grads[i]
                proj_grads[i] = gi - np.dot(gi, pi) * pi
            return proj_grads.flatten()
        
        try:
            result = minimize(
                objective, current_x, jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxiter * 3}
            )
            current_x = result.x
        except Exception:
            pass
    
    opt_pts = current_x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    opt_pts /= norms
    
    return opt_pts, _covering_radius(opt_pts)

def _local_polish(points, timeout=5.0):
    """
    Final polish: targeted moves of worst facet vertices toward circumcenters.
    Uses adaptive step size and checks improvement.
    """
    start_time = time.time()
    points = points.copy()
    current_radius = _covering_radius(points)
    
    for iteration in range(100):
        if time.time() - start_time > timeout:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        centers, radii = _spherical_circumcenters(hull, points)
        max_r = np.max(radii)
        
        if max_r < 1e-10:
            break
        
        # Find worst facet
        worst_idx = np.argmax(radii)
        face = hull.simplices[worst_idx]
        center = centers[worst_idx]
        
        # Adaptive step: start large, decrease
        base_step = 0.05 * (1.0 - 0.8 * iteration / 100.0)
        
        best_improvement = 0
        best_new_points = None
        
        for step_scale in [1.0, 0.5, 0.25, 0.1]:
            step = base_step * step_scale
            trial_points = points.copy()
            
            for j in face:
                v = trial_points[j]
                dir_vec = center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    trial_points[j] += step * dir_vec / norm_dir
            
            # Renormalize
            trial_points /= np.linalg.norm(trial_points, axis=1, keepdims=True)
            
            trial_radius = _covering_radius(trial_points)
            improvement = current_radius - trial_radius
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_new_points = trial_points.copy()
        
        if best_new_points is not None and best_improvement > 1e-12:
            points = best_new_points
            current_radius = _covering_radius(points)
        else:
            # No improvement found, try smaller steps or break
            if base_step < 0.001:
                break
    
    return points, _covering_radius(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [init_fib]
    
    # 2. Random starts with different seeds for diversity
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        p = rng.normal(size=(N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    # 3. Perturbed Fibonacci
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        p = init_fib.copy()
        noise = rng.normal(scale=0.1, size=(N, 3))
        p += noise
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    for idx, start in enumerate(starts):
        if time.time() > deadline - 15:
            break
            
        remaining = deadline - time.time()
        if remaining < 20:
            break
            
        # Phase 1: L-BFGS with adaptive beta
        lbfgs_timeout = min(remaining - 15, 40.0)
        opt_pts, radius = _optimize_lbfgs(start, timeout=lbfgs_timeout)
        
        # Phase 2: Local polish
        if time.time() < deadline - 8:
            polish_timeout = min(deadline - time.time() - 5, 8.0)
            opt_pts, radius = _local_polish(opt_pts, timeout=polish_timeout)
        
        if radius < best_radius:
            best_radius = radius
            best_points = opt_pts.copy()
            
    if best_points is None:
        return init_fib
        
    return best_points
