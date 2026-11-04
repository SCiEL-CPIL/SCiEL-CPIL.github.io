import numpy as np
import scipy.optimize as opt
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _normalize(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
    
    if len(hull.simplices) == 0:
        return np.pi, None
    
    max_r = 0.0
    worst_c = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        if r > max_r:
            max_r = r
            worst_c = cc
    
    return max_r, worst_c

def _facet_circumradius(a, b, c):
    """
    Compute the angular circumradius of the spherical triangle (a, b, c).
    Returns (radius, circumcenter) where circumcenter is the unit vector
    equidistant from a, b, c (the one with smaller angular distance).
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi, np.zeros(3)
    n = n / nn
    
    best_r = np.pi
    best_c = n
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        # Check if p is equidistant from a, b, c
        if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
            if da < best_r:
                best_r = da
                best_c = p
    return best_r, best_c

def _fast_facets(pts):
    """
    Compute all facet circumradii and circumcenters efficiently.
    Returns (radii, circumcenters) arrays.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    
    if len(hull.simplices) == 0:
        return None, None
    
    nfac = len(hull.simplices)
    radii = np.zeros(nfac)
    circumcenters = np.zeros((nfac, 3))
    
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        radii[i] = r
        circumcenters[i] = cc
    
    return radii, circumcenters

def _smooth_covering_radius_grad(pts, beta=50.0):
    """
    Compute the log-sum-exp smooth approximation of the covering radius
    and its gradient.
    
    Uses the facet circumradii as the basis for the smooth max.
    """
    radii, circumcenters = _fast_facets(pts)
    if radii is None:
        return np.pi, None
    
    # Log-sum-exp for numerical stability
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(pts)
    
    # LSE: (1/beta) * log(sum(exp(beta * r_i)))
    # For the gradient, we need to know how each facet's circumradius
    # depends on the points. This is complex, so we use a simpler approach:
    # approximate the gradient by moving points toward the weighted circumcenters.
    
    weights = np.exp(beta * (radii - max_r))
    weights /= weights.sum()
    
    # Gradient approximation: for each point, sum weighted circumcenters
    # of facets it belongs to, then normalize
    grad = np.zeros_like(pts)
    for i, simplex in enumerate(_fast_facets(pts)[1] is not None and _get_hull_simplices(pts) or []):
        pass
    
    # Simpler approach: use the weighted circumcenters directly
    # For each facet, the circumcenter is the direction to move
    # We assign the gradient contribution to the three vertices of each facet
    try:
        hull = ConvexHull(pts)
    except Exception:
        return max_r, None
    
    for i, simplex in enumerate(hull.simplices):
        w = weights[i]
        cc = circumcenters[i]
        for idx in simplex:
            # The gradient of the circumradius w.r.t. point idx points
            # from the point toward the circumcenter (to reduce the radius)
            grad[idx] += w * (cc - pts[idx])
    
    # Normalize gradient
    gn = np.linalg.norm(grad, axis=1, keepdims=True)
    gn[gn < 1e-12] = 1.0
    grad = grad / gn
    
    return max_r, grad

def _get_hull_simplices(pts):
    """Get hull simplices, returning empty list on failure."""
    try:
        hull = ConvexHull(pts)
        return hull.simplices
    except Exception:
        return []

def _riesz_energy_grad(pts, s=2.0):
    """
    Compute the gradient of the Riesz s-energy on the sphere.
    E = sum_{i<j} ||p_i - p_j||^{-s}
    This encourages uniform spreading.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            # dE/dp_i = s * d / dist^{s+2}
            # dE/dp_j = -s * d / dist^{s+2}
            coeff = s / (dist ** (s + 2))
            grad[i] += coeff * d
            grad[j] -= coeff * d
    
    return grad

def _lbfgs_smooth(pts, beta=50.0, max_iter=100, deadline=None):
    """
    L-BFGS optimization using the smooth LSE covering radius surrogate.
    """
    current_pts = _normalize(pts.copy())
    current_val, _ = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    def objective_and_grad(x):
        p = _normalize(x.reshape(N, 3))
        val, grad = _smooth_covering_radius_grad(p, beta=beta)
        # Convert to flat gradient
        g = grad.flatten()
        return val, g
    
    # Use L-BFGS-B with the smooth objective
    x0 = current_pts.flatten()
    
    try:
        result = opt.minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        if result.success:
            new_pts = _normalize(result.x.reshape(N, 3))
            new_val, _ = _covering_radius(new_pts)
            if new_val < best_val:
                best_pts = new_pts
                best_val = new_val
    except Exception:
        pass
    
    return best_pts, best_val

def _lbfgs_riesz(pts, s=2.0, max_iter=100, deadline=None):
    """
    L-BFGS optimization of Riesz energy for uniform spreading.
    """
    current_pts = _normalize(pts.copy())
    
    def objective_and_grad(x):
        p = _normalize(x.reshape(N, 3))
        # Riesz energy
        energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = p[i] - p[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    energy += dist ** (-s)
        grad = _riesz_energy_grad(p, s=s)
        return energy, grad.flatten()
    
    x0 = current_pts.flatten()
    best_pts = current_pts.copy()
    best_energy, _ = _covering_radius(current_pts)
    
    try:
        result = opt.minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-5, 'ftol': 1e-7}
        )
        if result.success:
            new_pts = _normalize(result.x.reshape(N, 3))
            new_val, _ = _covering_radius(new_pts)
            if new_val < best_energy:
                best_pts = new_pts
                best_energy = new_val
    except Exception:
        pass
    
    return best_pts, best_energy

def _lloyd_iteration(pts, max_iter=80, deadline=None):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    """
    pts = _normalize(pts.copy())
    current_cr, _ = _covering_radius(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute all facet circumradii and identify the worst
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, compute a target direction based on the facets it belongs to
        # Weighted by the circumradius of those facets
        targets = np.zeros_like(pts)
        weights = np.zeros(len(pts))
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            r, cc = _facet_circumradius(a, b, c)
            
            # The circumcenter cc is the point on the sphere farthest from the facet
            # We want to move the vertices of this facet toward cc to reduce the radius
            # Weight by r^2 to emphasize the worst facets
            w = r * r
            for idx in simplex:
                targets[idx] += w * cc
                weights[idx] += w
        
        # Normalize targets and blend with current positions
        improved = False
        for idx in range(len(pts)):
            if weights[idx] > 1e-12:
                target = targets[idx] / np.linalg.norm(targets[idx])
                # Move a fraction toward the target
                new_pt = 0.7 * pts[idx] + 0.3 * target
                new_pt = _normalize(new_pt.reshape(1, 3))[0]
                pts[idx] = new_pt
                improved = True
        
        if not improved:
            break
        
        new_cr, _ = _covering_radius(pts)
        if new_cr < current_cr - 1e-10:
            current_cr = new_cr
        else:
            # Revert if no improvement
            break
    
    return pts, current_cr

def _simulated_annealing(pts, max_iter=200, T0=0.3, T_end=0.01, deadline=None):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = _normalize(pts.copy())
    current_val, _ = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.1 * direction
        new_pts[idx] = _normalize(new_pts[idx].reshape(1, 3))[0]
        
        new_val, _ = _covering_radius(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
    return best_pts, best_val

def _fibonacci_points(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _jittered_fibonacci_points(n, jitter=0.05, seed=None):
    """Generate jittered Fibonacci points for diversity."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    base = _fibonacci_points(n)
    noise = rng.randn(n, 3) * jitter
    return _normalize(base + noise)

def _local_refine(pts, max_iter=50, step0=0.05, deadline=None):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _normalize(pts.copy())
    best_cr, worst_c = _covering_radius(pts)
    step = step0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Move that point toward the circumcenter
        new_pt = pts[k] + step * worst_c
        new_pt = _normalize(new_pt.reshape(1, 3))[0]
        
        cand = pts.copy()
        cand[k] = new_pt
        cr, wc = _covering_radius(cand)
        
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            worst_c = wc
            step = min(step * 1.1, 0.2)
        else:
            step *= 0.8
            if step < 1e-5:
                break
    
    return pts, best_cr

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses convex hull-based covering radius, Lloyd's relaxation, local refinement,
    smooth LSE surrogate, Riesz energy, and simulated annealing with multi-start.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    np.random.seed(42)
    
    best_pts = None
    best_val = np.pi
    
    # Start with multiple diverse initializations
    starts = [_fibonacci_points(N)]
    
    # Add random starts with different seeds
    for seed in [12345, 67890, 24680, 13579]:
        starts.append(_random_points(N, seed=seed))
    
    # Add jittered Fibonacci starts
    for seed in [111, 222, 333]:
        starts.append(_jittered_fibonacci_points(N, jitter=0.05, seed=seed))
    
    # Add icosahedron-based start
    ico = np.array([
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [-1/np.sqrt(2), -1/np.sqrt(2), 0],
        [0, 1/np.sqrt(2), 1/np.sqrt(2)],
        [0, -1/np.sqrt(2), 1/np.sqrt(2)],
        [0, 1/np.sqrt(2), -1/np.sqrt(2)],
        [0, -1/np.sqrt(2), -1/np.sqrt(2)],
        [1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [1/np.sqrt(2), 0, -1/np.sqrt(2)],
        [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
    ])
    ico = _normalize(ico)
    # Expand to 50 points by adding more
    extra = _random_points(30, seed=999)
    starts.append(_normalize(np.vstack([ico, extra])))
    
    for st in starts:
        if time.time() > deadline - 40.0:
            break
        
        # Stage 1: Riesz energy pre-refinement for uniform spread
        riesz_pts, riesz_val = _lbfgs_riesz(st, s=2.0, max_iter=50, deadline=deadline)
        
        # Stage 2: Lloyd's relaxation
        lloyd_pts, lloyd_val = _lloyd_iteration(riesz_pts, max_iter=60, deadline=deadline)
        
        # Stage 3: Smooth LSE refinement
        smooth_pts, smooth_val = _lbfgs_smooth(lloyd_pts, beta=50.0, max_iter=80, deadline=deadline)
        
        # Stage 4: Local refinement
        ref_pts, ref_val = _local_refine(smooth_pts, max_iter=40, step0=0.05, deadline=deadline)
        
        # Stage 5: Simulated annealing
        sa_pts, sa_val = _simulated_annealing(ref_pts, max_iter=120, T0=0.2, T_end=0.005, deadline=deadline)
        
        if sa_val < best_val:
            best_val = sa_val
            best_pts = sa_pts.copy()
    
    # Final local refinement on the best result
    if best_pts is not None and time.time() < deadline - 15.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=30, step0=0.03, deadline=deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    # Final smooth LSE polish
    if best_pts is not None and time.time() < deadline - 10.0:
        polish_pts, polish_val = _lbfgs_smooth(best_pts, beta=100.0, max_iter=50, deadline=deadline)
        if polish_val < best_val:
            best_val = polish_val
            best_pts = polish_pts.copy()
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_points(N)
    
    best_pts = _normalize(best_pts)
    
    return best_pts
