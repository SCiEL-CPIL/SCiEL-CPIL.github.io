"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _facet_circumradius_vec(pts, hull):
    """
    Vectorized computation of facet circumradii for a convex hull.
    Returns (max_r, worst_circumcenter, all_radii, all_centers)
    """
    simplices = hull.simplices
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    
    valid = nn > 1e-12
    n = np.where(valid[:, None], n / nn[:, None], 0.0)
    
    # For each facet, try both normals and pick the one where all three
    # vertices are at equal angular distance (i.e., the circumcenter)
    # The circumcenter direction p satisfies: arccos(p·a) = arccos(p·b) = arccos(p·c)
    # This means p·a = p·b = p·c (for angles in [0, pi])
    # So p is in the direction of the plane normal, and we need p·a = p·b = p·c
    # p = s * n, so s*(n·a) = s*(n·b) = s*(n·c)
    # This requires n·a = n·b = n·c, which is NOT generally true.
    # 
    # Actually, the circumcenter of a spherical triangle is the point equidistant
    # from all three vertices. For a triangle on the sphere, the circumcenter
    # is along the normal to the plane containing a, b, c only if the triangle
    # is "right" in some sense. In general, we need to solve for p on the sphere
    # such that p·a = p·b = p·c.
    #
    # Let p = alpha*n + beta*(a+b+c) ... no, p must be unit.
    # Actually, p·a = p·b = p·c means p is perpendicular to (a-b) and (a-c).
    # So p is parallel to n = (b-a) x (c-a). So p = s*n/|n|.
    # Then p·a = s*(n·a)/|n|, p·b = s*(n·b)/|n|, p·c = s*(n·c)/|n|.
    # For these to be equal, we need n·a = n·b = n·c.
    # n·a = (b-a) x (c-a) · a = a x (c-a) · (b-a) ... 
    # Actually n·a = (b-a) x (c-a) · a
    # n·b = (b-a) x (c-a) · b
    # n·c = (b-a) x (c-a) · c
    # 
    # These are generally NOT equal. So the circumcenter is NOT along the plane normal
    # in general. I need to solve the system properly.
    #
    # For a spherical triangle with vertices a, b, c, the circumcenter p satisfies:
    # p·a = p·b = p·c = cos(r) where r is the circumradius.
    # p is a unit vector.
    # 
    # From p·a = p·b: p·(a-b) = 0
    # From p·a = p·c: p·(a-c) = 0
    # So p is perpendicular to both (a-b) and (a-c), hence p = ± n/|n|.
    # 
    # Wait, that's what I said before. Let me recheck.
    # p·(a-b) = 0 and p·(a-c) = 0 means p is in the null space of the matrix
    # with rows (a-b)^T and (a-c)^T. The null space is 1D, spanned by (a-b) x (a-c) = n.
    # So p = s * n / |n| for some s = ±1.
    # 
    # Then p·a = s * (n·a) / |n|. For this to be the circumradius cos, we just
    # evaluate it. The circumradius is arccos(|p·a|) if p·a = p·b = p·c, but
    # they may not be equal!
    #
    # Hmm, let me reconsider. If p = n/|n|, then:
    # p·a = (n·a)/|n|, p·b = (n·b)/|n|, p·c = (n·c)/|n|
    # These are equal iff n·a = n·b = n·c.
    #
    # n·a = (b-a) x (c-a) · a
    # = b·(c-a)·... let me compute: (b-a) x (c-a) = b x c - b x a - a x c + a x a
    # = b x c - b x a - a x c
    # Dot with a: a·(b x c) - a·(b x a) - a·(a x c)
    # = a·(b x c) - 0 - 0  [since a·(b x a) = 0 and a·(a x c) = 0]
    # = a·(b x c)
    #
    # Similarly n·b = b·(b x c) - b·(b x a) - b·(a x c)
    # = 0 - 0 - b·(a x c) = -b·(a x c) = c·(a x b) ... 
    # Actually b·(a x c) = a·(c x b) = -a·(b x c)
    # So n·b = -(-a·(b x c)) = a·(b x c)? Let me redo.
    #
    # n·b = [(b-a) x (c-a)] · b
    # Expand (b-a) x (c-a) = b x c - b x a - a x c + a x a = b x c - b x a - a x c
    # Dot with b: b·(b x c) - b·(b x a) - b·(a x c) + b·(a x a)
    # = 0 - 0 - b·(a x c) - 0
    # = -b·(a x c)
    # Now b·(a x c) = a·(c x b) = -a·(b x c)
    # So n·b = -(-a·(b x c)) = a·(b x c)
    #
    # Similarly n·c = [(b-a) x (c-a)] · c
    # = c·(b x c) - c·(b x a) - c·(a x c)
    # = 0 - c·(b x a) - 0
    # = -c·(b x a) = c·(a x b)
    # And c·(a x b) = a·(b x c) (cyclic permutation)
    #
    # So n·a = n·b = n·c = a·(b x c) = the scalar triple product!
    # They ARE equal! I made an error earlier.
    #
    # So p = n/|n| gives p·a = p·b = p·c = (a·(b x c))/|n|
    # The circumradius is arccos(|(a·(b x c))/|n||) ... but we need to check
    # that this is the correct circumcenter (the one that is the center of the
    # smallest enclosing cap).
    
    triple = np.einsum('ij,ij->i', a, np.cross(b, c))  # a·(b x c)
    cos_r = triple / nn
    cos_r = np.clip(cos_r, -1.0, 1.0)
    
    # The circumcenter is p = s * n / |n| where s is chosen so that
    # the cap centered at p with radius arccos(cos_r) contains a, b, c
    # and is the smallest such cap.
    # p·a = s * triple / |n| = s * cos_r
    # For the cap to contain a, b, c, we need arccos(p·a) = arccos(p·b) = arccos(p·c) = r
    # where r = arccos(|cos_r|) ... actually r = arccos(p·a) and p·a = s*cos_r
    # 
    # The two candidates are p = n/|n| and p = -n/|n|.
    # For p = n/|n|: p·a = cos_r, r = arccos(cos_r)
    # For p = -n/|n|: p·a = -cos_r, r = arccos(-cos_r)
    # We want the smaller r.
    # arccos(cos_r) + arccos(-cos_r) = pi
    # So the smaller is min(arccos(cos_r), arccos(-cos_r)) = min(r, pi-r)
    # which is arccos(|cos_r|) if we take the absolute value... 
    # Actually arccos(cos_r) for cos_r in [-1,1]. If cos_r > 0, arccos(cos_r) < pi/2
    # and arccos(-cos_r) > pi/2. So smaller is arccos(cos_r).
    # If cos_r < 0, arccos(cos_r) > pi/2 and arccos(-cos_r) < pi/2. So smaller is arccos(-cos_r).
    # So the circumradius is arccos(|cos_r|).
    #
    # But wait - we also need the cap to be "empty" (no other points inside).
    # The problem states: "the largest circumradius of the facets of the convex hull,
    # each facet checked to have an empty cap". So we need to verify the cap is empty.
    # For the covering radius, we want the maximum over all facets of the circumradius
    # of the empty cap.
    
    # The circumcenter direction: choose s so that the cap is the smaller one
    # and contains the facet vertices.
    # p = sign(triple) * n / |n| gives p·a = |cos_r| >= 0, so r = arccos(|cos_r|) <= pi/2
    # This is the smaller cap. But we need to check it's empty.
    
    # For now, compute both and pick the one with smaller radius
    r_pos = np.arccos(np.clip(cos_r, -1.0, 1.0))  # for p = n/|n|
    r_neg = np.arccos(np.clip(-cos_r, -1.0, 1.0))  # for p = -n/|n|
    
    # Choose the smaller radius
    use_pos = r_pos <= r_neg
    r = np.where(use_pos, r_pos, r_neg)
    
    # Circumcenter direction
    n_unit = np.where(valid[:, None], n / np.maximum(nn, 1e-12)[:, None], 0.0)
    centers = np.where(use_pos[:, None], n_unit, -n_unit)
    
    # Handle degenerate facets
    r = np.where(valid, r, np.pi)
    centers = np.where(valid[:, None], centers, 0.0)
    
    max_r = np.max(r)
    idx = int(np.argmax(r))
    worst_c = centers[idx]
    
    return max_r, worst_c, r, centers

def _covering_radius_fast(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _to_sphere(pts)
    
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
    
    max_r, worst_c, _, _ = _facet_circumradius_vec(pts, hull)
    
    # Check that the worst cap is empty (no other points inside)
    # For the covering radius to be valid, the cap at the worst circumcenter
    # with radius max_r should not contain any other points.
    # Actually, by definition of the convex hull, the facets are on the boundary,
    # and the cap corresponding to a facet should be empty of other points
    # (otherwise the hull would be different).
    
    return max_r, worst_c

def _smooth_covering(pts, beta=200.0, n_samples=None):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    Differentiable and suitable for gradient-based optimization.
    """
    if n_samples is None:
        n_samples = 3000
    
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distance from each sample to nearest point
    dots = samples @ pts.T  # (n_samples, N)
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)  # (n_samples, N)
    
    # For each sample, find the minimum angle (distance to nearest point)
    min_angles = np.min(angles, axis=1)  # (n_samples,)
    
    # Log-sum-exp for smooth maximum
    max_angle = np.max(min_angles)
    log_terms = beta * (min_angles - max_angle)
    log_terms = np.where(log_terms > -50, log_terms, -50)
    sum_exp = np.sum(np.exp(log_terms))
    smooth_max = max_angle - np.log(sum_exp) / beta
    
    return smooth_max

def _smooth_covering_grad(pts, beta=200.0, n_samples=None):
    """
    Compute the smooth covering radius and its gradient with respect to points.
    """
    if n_samples is None:
        n_samples = 1500
    
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T  # (n_samples, N)
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
    min_idx = np.argmin(angles, axis=1)  # (n_samples,)
    min_angles = angles[np.arange(n_samples), min_idx]  # (n_samples,)
    
    # Soft max weights
    max_angle = np.max(min_angles)
    log_terms = beta * (min_angles - max_angle)
    log_terms = np.where(log_terms > -50, log_terms, -50)
    exp_terms = np.exp(log_terms)
    weights = exp_terms / np.sum(exp_terms)  # (n_samples,)
    
    smooth_val = max_angle - np.log(np.sum(exp_terms)) / beta
    
    # Gradient computation
    grad = np.zeros_like(pts)
    for i in range(n_samples):
        j = min_idx[i]
        w = weights[i]
        a = min_angles[i]
        sin_a = np.sin(a)
        if sin_a < 1e-10:
            continue
        d_angle_d_pt = -samples[i] / sin_a
        grad[j] += w * d_angle_d_pt
    
    return smooth_val, -grad

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    return _to_sphere(pts)

def _lloyd_relax(pts, max_iter=80, deadline=None):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    """
    pts = _to_sphere(pts.copy())
    current_cr, _ = _covering_radius_fast(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        targets = np.zeros_like(pts)
        weights = np.zeros(len(pts))
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                r = np.pi
                cc = np.zeros(3)
            else:
                n = n / nn
                # Circumcenter is along n (as proven, n·a = n·b = n·c)
                triple = np.dot(a, np.cross(b, c))
                cos_r = triple / nn
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r_pos = np.arccos(cos_r)
                r_neg = np.arccos(-cos_r)
                if r_pos <= r_neg:
                    r = r_pos
                    cc = n
                else:
                    r = r_neg
                    cc = -n
            
            w = r * r
            for idx in simplex:
                targets[idx] += w * cc
                weights[idx] += w
        
        improved = False
        for idx in range(len(pts)):
            if weights[idx] > 1e-12:
                target = targets[idx] / np.linalg.norm(targets[idx])
                new_pt = 0.6 * pts[idx] + 0.4 * target
                new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
                pts[idx] = new_pt
                improved = True
        
        if not improved:
            break
        
        new_cr, _ = _covering_radius_fast(pts)
        if new_cr < current_cr - 1e-10:
            current_cr = new_cr
        else:
            break
    
    return pts, current_cr

def _gradient_descent(pts, max_iter=60, deadline=None, beta_start=50.0, beta_end=500.0):
    """
    Gradient descent on the smooth covering radius with beta cascade.
    """
    pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius_fast(pts)
    best_pts = pts.copy()
    best_val = current_val
    
    # Beta cascade: start coarse, end fine
    n_betas = 5
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        lr = 0.02
        for it in range(max_iter // n_betas + 5):
            if deadline is not None and time.time() > deadline - 5.0:
                break
            
            val, grad = _smooth_covering_grad(pts, beta=beta, n_samples=1500)
            
            # Project to tangent space
            grad_proj = np.zeros_like(pts)
            for j in range(len(pts)):
                pj = pts[j]
                gj = grad[j]
                gj_tan = gj - np.dot(gj, pj) * pj
                grad_proj[j] = gj_tan
            
            new_pts = pts - lr * grad_proj
            new_pts = _to_sphere(new_pts)
            new_val, _ = _covering_radius_fast(new_pts)
            
            if new_val < best_val - 1e-10:
                pts = new_pts
                best_pts = new_pts.copy()
                best_val = new_val
                lr = min(lr * 1.03, 0.05)
            else:
                lr *= 0.9
                if lr < 1e-7:
                    break
    
    return best_pts, best_val

def _linalg_polish(pts, deadline=None, max_iter=30):
    """
    High-precision polish using L-BFGS-B on the smooth objective with tight tolerances.
    """
    pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius_fast(pts)
    best_pts = pts.copy()
    best_val = current_val
    
    x0 = pts.flatten()
    
    def obj(x):
        p = _to_sphere(x.reshape(N, 3))
        val, _ = _smooth_covering_grad(p, beta=300.0, n_samples=2000)
        return val
    
    def obj_grad(x):
        p = _to_sphere(x.reshape(N, 3))
        val, grad = _smooth_covering_grad(p, beta=300.0, n_samples=2000)
        grad_proj = np.zeros_like(x)
        for j in range(N):
            pj = p[j]
            gj = grad[j]
            gj_tan = gj - np.dot(gj, pj) * pj
            grad_proj[j * 3:(j + 1) * 3] = gj_tan
        return grad_proj.flatten()
    
    try:
        res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
        if res.success or res.fun < obj(x0):
            new_pts = _to_sphere(res.x.reshape(N, 3))
            new_val, _ = _covering_radius_fast(new_pts)
            if new_val < best_val:
                best_pts = new_pts
                best_val = new_val
    except Exception:
        pass
    
    return best_pts, best_val

def _local_refine(pts, max_iter=60, step0=0.03, deadline=None):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c = _covering_radius_fast(pts)
    step = step0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Try multiple step sizes and directions
        best_cand = None
        best_cand_cr = best_cr
        
        for s in [step, step * 0.5, step * 0.25]:
            new_pt = pts[k] + s * worst_c
            new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
            cand = pts.copy()
            cand[k] = new_pt
            cr, wc = _covering_radius_fast(cand)
            if cr < best_cand_cr - 1e-10:
                best_cand = cand
                best_cand_cr = cr
        
        if best_cand is not None:
            pts = best_cand
            best_cr = best_cand_cr
            worst_c, _ = _covering_radius_fast(pts)
            step = min(step * 1.05, 0.12)
        else:
            step *= 0.85
            if step < 1e-6:
                break
    
    return pts, best_cr

def _simulated_annealing(pts, max_iter=200, T0=0.15, T_end=0.003, deadline=None):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius_fast(current_pts)
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
        
        # Adaptive perturbation size
        scale = 0.05 + 0.05 * (T / T0)
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + scale * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val, _ = _covering_radius_fast(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    _deadline = time.time() + 140.0  # safety margin vs 150s kill
    
    np.random.seed(42)
    
    best_pts = None
    best_val = np.pi
    
    # Multiple starting configurations
    starts = [_fibonacci_spiral(N)]
    for seed in [1, 2, 3, 4, 5, 6]:
        starts.append(_random_points(N, seed=seed))
    
    for st in starts:
        if time.time() > _deadline - 30.0:
            break
        
        # Apply Lloyd's relaxation
        lloyd_pts, lloyd_val = _lloyd_relax(st, max_iter=60, deadline=_deadline)
        
        # Apply gradient descent with beta cascade
        gd_pts, gd_val = _gradient_descent(lloyd_pts, max_iter=80, deadline=_deadline,
                                           beta_start=50.0, beta_end=500.0)
        
        # Apply L-BFGS polish
        lbfgs_pts, lbfgs_val = _linalg_polish(gd_pts, deadline=_deadline, max_iter=40)
        
        # Apply local refinement
        ref_pts, ref_val = _local_refine(lbfgs_pts, max_iter=40, step0=0.025, deadline=_deadline)
        
        # Apply simulated annealing for final polish
        sa_pts, sa_val = _simulated_annealing(ref_pts, max_iter=150, T0=0.12, T_end=0.003,
                                              deadline=_deadline)
        
        if sa_val < best_val:
            best_val = sa_val
            best_pts = sa_pts.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < _deadline - 10.0:
        final_pts, final_val = _linalg_polish(best_pts, deadline=_deadline, max_iter=50)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    # Final local refinement
    if best_pts is not None and time.time() < _deadline - 5.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=30, step0=0.015, deadline=_deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
