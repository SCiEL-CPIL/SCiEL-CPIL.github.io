import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    """
    Generate points on two concentric circles.
    n1, n2: number of points on outer and inner ring.
    r1, r2: radii.
    rot1, rot2: rotations.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    """
    Generate points on three concentric circles.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _smooth_min_triangle_area(points, beta=100.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    This is differentiable and approaches the true min as beta → ∞.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # For numerical stability, shift by max
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    
    # Stable log-sum-exp: log(sum(exp(-beta * areas))) = -beta*max_area + log(sum(exp(-beta*(areas - max_area))))
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    log_sum = -beta * max_area + np.log(np.sum(exp_vals))
    smooth_min = -log_sum / beta
    return smooth_min

def _smooth_score(points, beta=100.0):
    """
    Smooth version of the score: smooth_min_triangle_area / hull_area
    """
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _smooth_min_triangle_area(points, beta)
        return smta / ha
    except Exception:
        return 0.0

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    Uses central finite differences for gradient.
    """
    points = points_flat.reshape(-1, 2)
    score = _smooth_score(points, beta)
    
    # Numerical gradient with central differences
    eps = 1e-6
    grad = np.zeros_like(points_flat)
    for i in range(len(points_flat)):
        orig = points_flat[i]
        points_flat[i] = orig + eps
        f_plus = _smooth_score(points_flat.reshape(-1, 2), beta)
        points_flat[i] = orig - eps
        f_minus = _smooth_score(points_flat.reshape(-1, 2), beta)
        points_flat[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return -score, -grad

def _polish_exact(points, maxiter=2000):
    """
    High-precision polish using exact (non-smooth) score with Nelder-Mead.
    Uses very tight tolerances for final refinement.
    """
    try:
        def _neg_exact_score(x_flat):
            return -_score(x_flat.reshape(-1, 2))
        
        res = scipy.optimize.minimize(
            _neg_exact_score,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=30.0, beta_end=800.0, maxiter=3000):
        """
        Optimize using smooth surrogate with annealed beta.
        Start with low beta (smoother) and increase to high beta (sharper).
        """
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        # Anneal beta from beta_start to beta_end with more stages
        n_stages = 7
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for stage, beta in enumerate(betas):
            if time.time() > deadline - 5:
                break
            
            # Decrease maxiter as we go to higher beta (need fewer iterations)
            iter_frac = 1.0 - 0.5 * (stage / n_stages)
            stage_maxiter = int(maxiter * iter_frac / n_stages)
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': stage_maxiter, 'gtol': 1e-9, 'ftol': 1e-13}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with exact (non-smooth) score using Nelder-Mead
        if time.time() < deadline - 2:
            try:
                polished = _polish_exact(x.reshape(-1, 2), maxiter=1500)
                _update_best(polished)
            except Exception:
                pass

    def _try_optimize_points(init_points, maxiter=3000):
        """Optimize the 2N coordinates directly using smooth surrogate."""
        _try_optimize_smooth(init_points, beta_start=30.0, beta_end=800.0, maxiter=maxiter)

    # 1. Explore the two-ring family analytically
    # Possible splits for N=14: (1,13), (2,12), (3,11), (4,10), (5,9), (6,8), (7,7)
    # (7,7) is symmetric and a very strong candidate.
    # (6,8) and (8,6) are also good.
    
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4)]
    
    for n1, n2 in splits:
        if time.time() > deadline:
            break
            
        # Optimize the parameters of the two-ring configuration
        # Params: r1 (normalized to 1), r2/r1, rot1, rot2
        # We fix r1=1 for optimization and scale at the end.
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            # Ensure r2 > 0
            if r2 <= 0.01:
                return -1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            # Normalize and score
            sc = _score(_normalize(pts))
            return -sc

        # Initial guesses for r_ratio, rot1, rot2
        # r_ratio typically between 0.4 and 0.8
        # rot1 = 0 is WLOG due to rotational symmetry of the whole set? No, relative rotation matters.
        # We can fix rot1=0 and optimize rot2 relative to it? 
        # Actually, the score is invariant under global rotation. So we can fix rot1=0.
        # But the optimizer might struggle if we don't allow it. Let's just optimize all 3.
        
        # Try a few different starting points for the ratio
        for r_init in [0.35, 0.45, 0.55, 0.65, 0.75]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2, 2*np.pi/n1, np.pi/n1 + np.pi/n2]:
                if time.time() > deadline:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    # If this configuration is good, refine it by optimizing all 2N coordinates
                    if sc > best_score * 0.93 and time.time() < deadline - 20:
                        _try_optimize_points(pts, maxiter=2500)
                        
                except Exception:
                    pass

    # 2. Explore three-ring configurations
    three_splits = [(5, 5, 4), (4, 5, 5), (3, 5, 6), (6, 5, 3), (5, 4, 5), (4, 4, 6), (6, 4, 4)]
    
    for n1, n2, n3 in three_splits:
        if time.time() > deadline:
            break
            
        def _three_ring_score(params):
            r2_ratio = params[0]
            r3_ratio = params[1]
            rot2 = params[2]
            rot3 = params[3]
            
            r1 = 1.0
            r2 = r2_ratio * r1
            r3 = r3_ratio * r1
            
            if r2 <= 0.01 or r3 <= 0.01:
                return -1e6
                
            pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, 0.0, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        # Try several initial guesses
        for r2_init in [0.5, 0.6, 0.7]:
            for r3_init in [0.3, 0.4, 0.5]:
                for rot2_init in [0.0, np.pi/n2]:
                    for rot3_init in [0.0, np.pi/n3]:
                        if time.time() > deadline:
                            continue
                        x0 = np.array([r2_init, r3_init, rot2_init, rot3_init])
                        try:
                            res = scipy.optimize.minimize(
                                _three_ring_score,
                                x0,
                                method='Nelder-Mead',
                                options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-11}
                            )
                            r2_ratio, r3_ratio, rot2, rot3 = res.x
                            pts = _generate_three_ring_config(n1, n2, n3, 1.0, r2_ratio, r3_ratio, 0.0, rot2, rot3)
                            sc = _update_best(pts)
                            
                            if sc > best_score * 0.93 and time.time() < deadline - 25:
                                _try_optimize_points(pts, maxiter=2000)
                                
                        except Exception:
                            pass

    # 3. Explore regular polygon and slight perturbations
    # Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Perturb regular polygon and optimize with smooth surrogate
    rng = np.random.RandomState(42)
    for i in range(12):
        if time.time() > deadline - 15:
            break
        noise = rng.uniform(-0.15, 0.15, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_points(init, maxiter=2000)

    # 4. Random restarts with optimization
    # Generate random points and optimize
    for i in range(6):
        if time.time() > deadline - 15:
            break
        # Generate points on a disk, then optimize
        r = np.sqrt(rng.uniform(0.2, 1.0, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_points(init, maxiter=1500)

    # 5. Final high-precision polish on the best solution found
    if best_points is not None and time.time() < deadline - 1:
        try:
            polished = _polish_exact(best_points, maxiter=3000)
            _update_best(polished)
        except Exception:
            pass

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
