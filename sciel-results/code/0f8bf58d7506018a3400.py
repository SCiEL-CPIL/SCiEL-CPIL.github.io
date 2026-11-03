import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians) and array of circumcenter unit vectors."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    n = n / np.maximum(n_norm[:, None], 1e-10)
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _covering_radius_rad(pts):
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=500, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
    pts = pts.copy()
    for it in range(n_iters):
        # Pairwise dot products
        D = pts @ pts.T
        # Force on i from j: proportional to (pts[i] - pts[j]) / |...|^2 but on sphere we use tangent component
        # Use inverse-square repulsion in chordal distance
        # chord^2 = 2 - 2*D
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        # Repulsion magnitude ~ 1/chord^2
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        # Force = sum_j W_ij * (pts[i] - pts[j]) projected to tangent
        # = pts[i] * sum_j W_ij - sum_j W_ij * pts[j]
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        # Project to tangent plane
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        forces /= (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12)
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(pts, beta=8.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max).
    Returns (surrogate_value, softmax_weights, radii, simplices, circumcenters, mask)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None
    # Use only valid facets
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    # Soft-max of squared radii
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, cc_valid, simp_valid

def _surrogate_gradient(pts, weights, simplices, circumcenters, beta=8.0):
    """Gradient of smooth surrogate w.r.t. each point (in R^3, then projected to tangent)."""
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = circumcenters[k]
        a = pts[simplices[k, 0]]
        cos_r = np.clip(np.dot(c, a), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        scale = w * (2.0 * cos_r) / sin_r
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            r_k = np.arccos(cos_r)
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad

def _lm_polish(pts, max_time=10.0):
    """Polish points using Levenberg-Marquardt to equalize facet radii."""
    start = time.time()
    x0 = pts.flatten()
    
    def residuals(x):
        if time.time() - start > max_time:
            raise TimeoutError
        pts_flat = x.reshape(N, 3)
        pts_norm = _normalize(pts_flat)
        try:
            hull = ConvexHull(pts_norm)
        except Exception:
            # Return large penalty if hull fails
            return np.ones(N * 3) * 10.0
        
        radii, _, mask = _facet_circumradii(pts_norm, hull.simplices)
        if not np.any(mask):
            return np.ones(N * 3) * 10.0
        
        # We want to minimize the variance of the radii.
        # Residuals are radii - mean(radii).
        # But the number of facets might change.
        # To handle this, we fix the number of residuals to the initial number?
        # No, that's hard.
        # Instead, we just return the radii themselves?
        # Minimizing sum(radii^2) is not ideal.
        # Let's try minimizing the max radius by using a smooth approximation?
        # No, LM needs a fixed size vector.
        
        # Strategy: If the number of facets changes, we break out of the solver?
        # `least_squares` doesn't support dynamic size.
        # So we must ensure the topology is stable or use a trick.
        
        # Trick: Pad or truncate the residual vector to a fixed size M.
        # This is hacky.
        
        # Alternative: Just use the gradient of the max radius?
        # The current code already does gradient descent on a surrogate.
        # The directive suggests LM.
        
        # Let's assume the topology is stable for a few steps.
        # If it changes, the residual size changes, and `least_squares` will fail.
        # To prevent this, I will check the hull size in the main loop and restart if it changes.
        
        # For now, return the radii.
        # If the size changes, this function will raise an error or return wrong size.
        # I will wrap the call in a try-except in the main loop.
        
        return radii[~mask] # Only valid facets? No, mask is for valid normals.
        # Actually, _facet_circumradii returns radii for all simplices in hull.simplices.
        # The mask indicates which ones have valid normals.
        # I should return all radii, but handle the invalid ones?
        # Let's just return radii.
        
    try:
        # Initial residual to check size
        res0 = residuals(x0)
        if res0.size == 0:
            return pts
        
        # Run LM
        # We want to minimize the variance, so we should subtract the mean.
        # But we can't easily do that inside residuals if we want to use LM properly?
        # We can define residuals as radii - mean(radii).
        # Let's modify residuals to do that.
        
        def residuals_var(x):
            if time.time() - start > max_time:
                raise TimeoutError
            pts_flat = x.reshape(N, 3)
            pts_norm = _normalize(pts_flat)
            try:
                hull = ConvexHull(pts_norm)
            except Exception:
                return np.ones(N * 3) * 10.0
            
            radii, _, mask = _facet_circumradii(pts_norm, hull.simplices)
            if not np.any(mask):
                return np.ones(N * 3) * 10.0
            
            # If size changes, this is problematic.
            # I will just return the radii. Minimizing sum of squares of radii
            # will tend to minimize them, but might not equalize them.
            # However, combined with the repulsion/greedy phase, it should work.
            # Actually, minimizing sum(r^2) is a good proxy for minimizing max(r)
            # if the distribution is tight.
            
            return radii

        # Use 'trf' (trust region reflective) or 'lm' (Levenberg-Marquardt)
        # 'lm' is faster but requires no bounds.
        # We don't have bounds, but points must be on sphere.
        # The normalization in residuals handles the sphere constraint.
        
        res = least_squares(residuals_var, x0, method='lm', max_nfev=100)
        
        if res.success or res.cost < 0: # cost is sum of squares
            new_pts = _normalize(res.x.reshape(N, 3))
            # Check if max radius improved
            old_rad = _covering_radius_rad(pts)
            new_rad = _covering_radius_rad(new_pts)
            if new_rad < old_rad:
                return new_pts
    except Exception:
        pass
    
    return pts

def construct():
    start_time = time.time()
    max_time = 140.0  # Safety margin before 150s kill

    def _time_left():
        return max_time - (time.time() - start_time)

    best_points = _fibonacci_spiral()
    best_rad = _covering_radius_deg(best_points)

    # Phase 0: Multiple initializations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.3))
    inits.append(_fibonacci_spiral(0.7))
    # Random
    rng = np.random.RandomState(42)
    for _ in range(2):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    # Electrostatic-refined from Fibonacci
    inits.append(_electrostatic_refine(_fibonacci_spiral(0.0), n_iters=300, lr=0.005))

    # Evaluate all initializations
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_deg(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()

    # Phase 1: For each good initialization, do greedy Lloyd + LM Polish + polish
    init_list = []
    for init in inits:
        rad = _covering_radius_deg(init)
        init_list.append((rad, init.copy()))
    init_list.sort(key=lambda x: x[0])
    # Take top 3 initializations
    top_inits = [x[1] for x in init_list[:3]]

    for init_idx, init_pts in enumerate(top_inits):
        if _time_left() < 10.0:
            break
        pts = init_pts.copy()
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

        # Phase 1a: Greedy Lloyd (move worst facet vertices toward circumcenter)
        step = 0.02
        for it in range(400):
            if _time_left() < 5.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            # Find worst facet
            valid_idx = np.where(mask)[0]
            worst_local = valid_idx[np.argmax(radii[valid_idx])]
            worst_rad = radii[worst_local]
            if worst_rad < 0.1:  # ~5.7 degrees, good enough
                break
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            # Move vertices toward circumcenter
            for j in range(3):
                idx = simp[j]
                v = pts[idx]
                new_v = v + step * (c - v)
                new_v = new_v / np.linalg.norm(new_v)
                pts[idx] = new_v
            # Small random perturbation every 30 iters to escape local minima
            if it % 30 == 0:
                for _ in range(2):
                    idx = rng.randint(N)
                    tangent = rng.randn(3)
                    tangent -= np.dot(tangent, pts[idx]) * pts[idx]
                    tn = np.linalg.norm(tangent)
                    if tn > 1e-10:
                        tangent /= tn
                        new_v = pts[idx] + 0.01 * tangent
                        pts[idx] = new_v / np.linalg.norm(new_v)
            cur_rad = _covering_radius_deg(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
            # Decay step
            step *= 0.995
            if step < 0.001:
                step = 0.001

        # Phase 1b: Levenberg-Marquardt Polish
        if _time_left() < 5.0:
            continue
        pts = _lm_polish(pts, max_time=_time_left() - 2.0)
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

        # Phase 1c: Nelder-Mead style local polish on top-k worst facets
        if _time_left() < 5.0:
            continue
        for polish_iter in range(80):
            if _time_left() < 3.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            valid_idx = np.where(mask)[0]
            # Sort by radius descending, take top 3
            order = valid_idx[np.argsort(-radii[valid_idx])][:3]
            improved = False
            for worst_local in order:
                if _time_left() < 2.0:
                    break
                simp = hull.simplices[worst_local]
                c = cc[worst_local]
                for j in range(3):
                    idx = simp[j]
                    v = pts[idx]
                    # Try moving toward circumcenter with decreasing steps
                    for s in [0.02, 0.01, 0.005, 0.002]:
                        new_v = v + s * (c - v)
                        new_v = new_v / np.linalg.norm(new_v)
                        old_v = pts[idx].copy()
                        pts[idx] = new_v
                        new_rad = _covering_radius_deg(pts)
                        if new_rad < best_rad:
                            best_rad = new_rad
                            best_points = pts.copy()
                            improved = True
                        else:
                            pts[idx] = old_v
                    # Try random tangent perturbations
                    for _ in range(4):
                        tangent = rng.randn(3)
                        tangent -= np.dot(tangent, v) * v
                        tn = np.linalg.norm(tangent)
                        if tn < 1e-10:
                            continue
                        tangent /= tn
                        for s in [0.01, 0.005]:
                            new_v = v + s * tangent
                            new_v = new_v / np.linalg.norm(new_v)
                            old_v = pts[idx].copy()
                            pts[idx] = new_v
                            new_rad = _covering_radius_deg(pts)
                            if new_rad < best_rad:
                                best_rad = new_rad
                                best_points = pts.copy()
                                improved = True
                            else:
                                pts[idx] = old_v
            if not improved and polish_iter > 40:
                break

    # Final high-precision polish on best_points
    if _time_left() > 3.0:
        pts = best_points.copy()
        for it in range(100):
            if _time_left() < 2.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            valid_idx = np.where(mask)[0]
            worst_local = valid_idx[np.argmax(radii[valid_idx])]
            worst_rad = radii[worst_local]
            if worst_rad < 0.19:  # ~10.9 degrees
                break
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            for j in range(3):
                idx = simp[j]
                v = pts[idx]
                for s in [0.005, 0.002, 0.001]:
                    new_v = v + s * (c - v)
                    new_v = new_v / np.linalg.norm(new_v)
                    old_v = pts[idx].copy()
                    pts[idx] = new_v
                    new_rad = _covering_radius_deg(pts)
                    if new_rad < best_rad:
                        best_rad = new_rad
                        best_points = pts.copy()
                    else:
                        pts[idx] = old_v

    return _normalize(best_points)
