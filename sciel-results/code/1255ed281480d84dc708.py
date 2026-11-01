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
    Uses analytic gradient of the log-sum-exp soft-min.
    """
    points = points_flat.reshape(-1, 2)
    
    # Compute areas and weights
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Handle degenerate case
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    # Stable log-sum-exp for weights
    # We want p_k proportional to exp(-beta * A_k)
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    # Smooth min value
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    # Hull area for the score
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    # Gradient of smooth_min with respect to points
    # d(smooth_min)/d(P_i) = sum_k weights_k * d(A_k)/d(P_i)
    # A_k = 0.5 * | cross_k |
    # d(A_k)/d(P_i) = 0.5 * sign(cross_k) * d(cross_k)/d(P_i)
    
    # d(cross)/d(P_i): cross = (Pj-ix)(Pk-iy) - (Pj-iy)(Pk-ix)
    # Let v1 = Pj-Pi, v2 = Pk-Pi
    # cross = v1x v2y - v1y v2x
    # d(cross)/d(Pix) = -v2y - v1y = -(v1y + v2y)
    # d(cross)/d(Piy) = v2x + v1x = v1x + v2x
    # d(cross)/d(Pjx) = v2y
    # d(cross)/d(Pjy) = -v2x
    # d(cross)/d(Pkx) = -v1y
    # d(cross)/d(Pky) = v1x
    
    signs = np.sign(cross)
    # If cross is 0, sign is 0, which is fine (gradient 0)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # d(A_k) = 0.5 * sign(cross_k) * d(cross_k)
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    # Accumulate gradients for P_i
    # grad_points[i] += sum_k w_k * [d_cross_di_x, d_cross_di_y]
    # We need to sum over all triangles where i is the first index
    # Using np.add.at is efficient for this
    
    # Vectorized accumulation
    # dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    # np.add.at(grad_points, _I, dA_di)
    
    # To avoid creating large temporary arrays, we can do it in chunks or just use add.at
    # Since M=364 and N=14, the arrays are small.
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Gradient of score = smooth_min / ha
    # d(score)/d(P) = (1/ha) * d(smooth_min)/d(P)
    # Note: ha depends on points too, but for the purpose of finding local optima of the ratio,
    # ignoring the gradient of the denominator is a common approximation if the hull is stable.
    # However, for better precision, we should include it.
    # But calculating gradient of hull area is complex.
    # Given the directive to use smooth surrogate, and the fact that the hull area is relatively constant
    # in the neighborhood of the optimum (points stay in convex position), the approximation
    # grad_score ≈ grad_smooth_min / ha is often sufficient to drive the optimizer.
    # Let's stick to this for speed and simplicity, as the hull area changes slowly compared to triangle areas.
    
    grad_score = grad_points / ha
    
    # We are minimizing -score
    return -score, -grad_score.flatten()

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

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=2000):
        """
        Optimize using smooth surrogate with annealed beta.
        Start with low beta (smoother) and increase to high beta (sharper).
        """
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        # Anneal beta from beta_start to beta_end
        n_stages = 5
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                # Increased maxiter per stage due to faster analytic gradient
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with exact (non-smooth) score using Nelder-Mead
        # Or use a very high beta L-BFGS step
        if time.time() < deadline - 3:
            try:
                # High beta polish
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 1000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                # Final Nelder-Mead polish on exact score for fine-tuning
                if time.time() < deadline - 1:
                    def _neg_exact_score(x_flat):
                        return -_score(x_flat.reshape(-1, 2))
                    
                    res = scipy.optimize.minimize(
                        _neg_exact_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    def _try_optimize_points(init_points, maxiter=3000):
        """Optimize the 2N coordinates directly using smooth surrogate."""
        _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=maxiter)

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
        for r_init in [0.4, 0.5, 0.6, 0.7]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2]:
                if time.time() > deadline:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    # If this configuration is good, refine it by optimizing all 2N coordinates
                    if sc > best_score * 0.95 and time.time() < deadline - 15:
                        _try_optimize_points(pts, maxiter=2000)
                        
                except Exception:
                    pass

    # 2. Explore regular polygon and slight perturbations
    # Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Perturb regular polygon and optimize with smooth surrogate
    rng = np.random.RandomState(42)
    for i in range(15):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_points(init, maxiter=1500)

    # 3. Random restarts with optimization
    # Generate random points and optimize
    for i in range(8):
        if time.time() > deadline - 10:
            break
        # Generate points on a disk, then optimize
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_points(init, maxiter=1000)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
