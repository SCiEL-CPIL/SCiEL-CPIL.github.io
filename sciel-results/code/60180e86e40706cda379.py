"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius in radians using convex hull facet circumradii.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use sampled approach
        rng = np.random.RandomState(0)
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2*np.pi, 400)
        theta, phi = np.meshgrid(theta, phi)
        y = np.column_stack([
            np.sin(theta)*np.cos(phi).ravel(),
            np.sin(theta)*np.sin(phi).ravel(),
            np.cos(theta).ravel()
        ])
        dots = y @ pts.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        min_angles = angles.min(axis=1)
        return np.max(min_angles)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        if len(simplex) == 3:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute circumcenter of the spherical triangle
            v = np.cross(b - a, c - a)
            vn = np.linalg.norm(v)
            if vn < 1e-12:
                continue
            v = v / vn
            
            # Ensure outward normal
            centroid = (a + b + c) / 3.0
            if np.dot(v, centroid) < 0:
                v = -v
            
            # Circumradius is angle between circumcenter and any vertex
            cos_R = np.clip(np.dot(v, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            max_radius = max(max_radius, R)
    
    return max_radius

def _facet_circumdata(pts):
    """
    For each triangular facet of the convex hull, compute:
      - circumradius (angular)
      - circumcenter (unit vector)
      - the three vertex indices
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute circumcenter using the linear system approach
        M = np.array([a, b, c])
        rhs = np.ones(3)
        
        try:
            p = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            v = np.cross(b - a, c - a)
            vn = np.linalg.norm(v)
            if vn < 1e-12:
                continue
            p = v / vn
        
        pn = np.linalg.norm(p)
        if pn < 1e-12:
            continue
        p = p / pn
        
        cos_R = np.dot(p, a)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check that p is on the correct side
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v = v / vn
        
        centroid = (a + b + c) / 3.0
        if np.dot(v, centroid) < 0:
            v = -v
        
        if np.dot(p, v) < 0:
            p = -p
            R = np.arccos(np.clip(-cos_R, -1.0, 1.0))
        
        out.append((R, p, simplex))
    
    return out

def _worst_facet_shrink(pts, alpha=0.05, deadline=None):
    """
    Identify the facet with the largest circumradius and move its vertices
    toward the circumcenter along the great circle.
    """
    pts = _normalize(pts)
    best = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(200):
        if deadline is not None and time.time() > deadline:
            break
        
        facets = _facet_circumdata(pts)
        if not facets:
            break
        
        max_idx = np.argmax([f[0] for f in facets])
        max_R, center, simplex = facets[max_idx]
        
        new_pts = pts.copy()
        for idx in simplex:
            p = pts[idx]
            cos_theta = np.clip(np.dot(p, center), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            if theta < 1e-10:
                continue
            sin_theta = np.sin(theta)
            t = alpha
            w1 = np.sin((1-t)*theta) / sin_theta
            w2 = np.sin(t*theta) / sin_theta
            new_p = w1 * p + w2 * center
            new_pts[idx] = new_p
        
        new_pts = _normalize(new_pts)
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr - 1e-15:
            pts = new_pts
            best = new_pts.copy()
            best_cr = new_cr
        else:
            alpha = alpha * 0.8
            if alpha < 1e-8:
                break
    
    return best, best_cr

def _fibonacci_points(n, seed=0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _jittered_fibonacci_points(n, seed=0, jitter=0.1):
    """Fibonacci points with random jitter for diversity."""
    base = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * jitter
    pts = base + noise
    return _normalize(pts)

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius (log-sum-exp over facet radii) and its
    gradient with respect to the input points.
    """
    points = _normalize(points)
    facets = _facet_circumdata(points)
    if not facets:
        return 0.0, np.zeros_like(points)
    
    radii = np.array([f[0] for f in facets])
    centers = np.array([f[1] for f in facets])
    simplices = [f[2] for f in facets]
    
    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    
    exps = np.exp(beta * (radii - m))
    lse = m + np.log(np.sum(exps)) / beta
    w = exps / np.sum(exps)
    
    grad = np.zeros_like(points)
    for i in range(len(radii)):
        wi = w[i]
        if wi < 1e-15:
            continue
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += wi * g_tan
    
    return lse, grad

def _riesz_energy_grad(points, s=12.0):
    """
    Riesz s-energy and gradient (tangent-projected).
    """
    points = _normalize(points)
    n = len(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)
    sins = np.sin(angles)
    sins = np.maximum(sins, 1e-15)

    ang_powers = angles ** (-s - 1)
    np.fill_diagonal(ang_powers, 0.0)
    np.fill_diagonal(sins, 1.0)
    np.fill_diagonal(dots, 1.0)

    weights = s * ang_powers / sins
    np.fill_diagonal(weights, 0.0)

    grad = weights @ points
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points

    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(angles[iu] ** (-s)))
    return energy, grad_tan

def _lbfgs_refine(points, beta=50.0, deadline=None, maxiter=300):
    """Refine points using L-BFGS-B on a smooth log-sum-exp surrogate."""
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    
    def objective_and_grad(x_flat):
        if deadline is not None and time.time() > deadline:
            raise Exception("timeout")
        x = x_flat.reshape(-1, 3)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
        )
        
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)
        
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass
    
    return best, best_cr

def _lbfgs_riesz(points, s=12.0, maxiter=200, deadline=None):
    """
    L-BFGS-B on Riesz energy.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        if deadline is not None and time.time() > deadline:
            raise Exception("timeout")
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _riesz_energy_grad(x, s=s)
        return val, grad.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": 1e-12,
                "gtol": 1e-10,
                "maxfun": 400,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _spherical_gd(points, lr=0.01, n_iter=100, deadline=None, beta=50.0):
    """
    Spherical gradient descent on the LSE surrogate.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = _smooth_covering_radius_grad(points, beta=beta)
        # Project gradient to tangent plane
        grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Gradient step
        new_pts = points - lr * grad_tan
        new_pts = _normalize(new_pts)
        
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr - 1e-15:
            points = new_pts
            best = new_pts.copy()
            best_cr = new_cr
        else:
            lr *= 0.8
            if lr < 1e-8:
                break
    
    return best, best_cr

def _lloyd_iteration(points, n_iter=10, deadline=None):
    """
    Lloyd-type iteration: for each point, move it to the centroid of its
    Voronoi cell (approximated by the mean of facet circumcenters it belongs to).
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        facets = _facet_circumdata(points)
        if not facets:
            break
        
        # Accumulate target positions for each point
        targets = np.zeros_like(points)
        counts = np.zeros(len(points))
        
        for R, center, simplex in facets:
            for idx in simplex:
                targets[idx] += center
                counts[idx] += 1
        
        # Update points toward their targets
        new_pts = points.copy()
        for idx in range(len(points)):
            if counts[idx] > 0:
                target = targets[idx] / counts[idx]
                tn = np.linalg.norm(target)
                if tn > 1e-10:
                    target = target / tn
                    # Slerp between current and target
                    cos_theta = np.clip(np.dot(points[idx], target), -1.0, 1.0)
                    theta = np.arccos(cos_theta)
                    if theta > 1e-10:
                        sin_theta = np.sin(theta)
                        w1 = np.sin(0.5 * (1 - 0.5) * theta) / sin_theta
                        w2 = np.sin(0.5 * 0.5 * theta) / sin_theta
                        new_pts[idx] = w1 * points[idx] + w2 * target
        
        new_pts = _normalize(new_pts)
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr - 1e-15:
            points = new_pts
            best = new_pts.copy()
            best_cr = new_cr
        else:
            break
    
    return best, best_cr

def _simulated_annealing(points, t_start=0.5, t_end=0.01, n_iter=200, deadline=None):
    """
    Simulated annealing on the covering radius.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    current_cr = best_cr
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Random perturbation
        rng = np.random.RandomState(i)
        noise = rng.randn(len(points), 3) * t
        new_pts = points + noise
        new_pts = _normalize(new_pts)
        
        new_cr = _covering_radius(new_pts)
        
        # Accept if better or with probability exp(-delta/t)
        if new_cr < current_cr:
            points = new_pts
            current_cr = new_cr
            if new_cr < best_cr:
                best = new_pts.copy()
                best_cr = new_cr
        else:
            delta = new_cr - current_cr
            if delta < 1e-12:
                points = new_pts
                current_cr = new_cr
            else:
                prob = np.exp(-delta / t)
                if rng.random() < prob:
                    points = new_pts
                    current_cr = new_cr
    
    return best, best_cr

def _multi_beta_lse(points, betas=[20, 50, 100, 200], deadline=None):
    """
    Multi-beta LSE cascade: refine with increasing beta for sharper approximation.
    """
    pts = _normalize(points)
    best = pts.copy()
    best_cr = _covering_radius(pts)
    
    for beta in betas:
        if deadline is not None and time.time() > deadline:
            break
        
        opt_pts, opt_cr = _lbfgs_refine(pts, beta=beta, deadline=deadline, maxiter=150)
        if opt_cr < best_cr:
            best = opt_pts
            best_cr = opt_cr
            pts = opt_pts
    
    return best, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points(N))
    
    # Jittered Fibonacci
    for seed in [1, 2, 3]:
        initializations.append(_jittered_fibonacci_points(N, seed=seed, jitter=0.05))
    
    # Random starts
    for seed in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        initializations.append(_random_points(N, seed=seed))
    
    best_pts = initializations[0]
    best_cr = _covering_radius(best_pts)
    
    # Optimize from each start
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        cr_init = _covering_radius(init_pts)
        if cr_init < best_cr:
            best_pts = init_pts.copy()
            best_cr = cr_init
        
        # 1. Worst-facet shrinkage
        if time.time() < deadline - 30.0:
            try:
                opt_pts, opt_cr = _worst_facet_shrink(init_pts, alpha=0.05, deadline=deadline - 20.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # 2. Lloyd iteration
        if time.time() < deadline - 25.0:
            try:
                opt_pts, opt_cr = _lloyd_iteration(best_pts, n_iter=5, deadline=deadline - 20.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # 3. L-BFGS with moderate beta
        if time.time() < deadline - 20.0:
            try:
                opt_pts, opt_cr = _lbfgs_refine(best_pts, beta=50.0, deadline=deadline - 15.0, maxiter=200)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # 4. Riesz energy refinement
        if time.time() < deadline - 15.0:
            try:
                opt_pts, opt_cr = _lbfgs_riesz(best_pts, s=12.0, maxiter=150, deadline=deadline - 10.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
    
    # Final polish phase
    if time.time() < deadline - 15.0:
        # Multi-beta LSE cascade
        try:
            opt_pts, opt_cr = _multi_beta_lse(best_pts, betas=[30, 60, 120], deadline=deadline - 10.0)
            if opt_cr < best_cr:
                best_pts = opt_pts
                best_cr = opt_cr
        except Exception:
            pass
        
        # Spherical GD for fine-tuning
        if time.time() < deadline - 10.0:
            try:
                opt_pts, opt_cr = _spherical_gd(best_pts, lr=0.005, n_iter=50, deadline=deadline - 5.0, beta=100.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # Simulated annealing for escape from local minima
        if time.time() < deadline - 8.0:
            try:
                opt_pts, opt_cr = _simulated_annealing(best_pts, t_start=0.1, t_end=0.005, n_iter=100, deadline=deadline - 5.0)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
        
        # Final high-precision LSE polish
        if time.time() < deadline - 5.0:
            try:
                opt_pts, opt_cr = _lbfgs_refine(best_pts, beta=200.0, deadline=deadline, maxiter=100)
                if opt_cr < best_cr:
                    best_pts = opt_pts
                    best_cr = opt_cr
            except Exception:
                pass
    
    # Ensure return is valid
    best_pts = _normalize(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
