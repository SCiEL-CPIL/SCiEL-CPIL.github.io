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

def _points_from_gaps(gaps):
    """
    Convert angular gaps to points on unit circle.
    gaps: array of length N-1. The last gap is 2*pi - sum(gaps).
    """
    if len(gaps) != N - 1:
        return None
    if np.any(gaps <= 0):
        return None
    last_gap = 2 * np.pi - np.sum(gaps)
    if last_gap <= 0:
        return None
    all_gaps = np.append(gaps, last_gap)
    angles = np.cumsum(all_gaps)[:-1] # Start at 0
    # Sort angles to ensure consistent indexing? 
    # The gaps are in order, so angles are increasing.
    # We need to map these to the fixed triangle indices which assume sorted order?
    # No, triangle indices are just combinations of indices 0..N-1.
    # The geometry is invariant to rotation, so starting at 0 is fine.
    # The points are at angles[0], angles[1], ...
    # We return points in the order of the gaps.
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _score_from_gaps(gaps, beta=100.0):
    """
    Calculate smooth score from angular gaps.
    """
    pts = _points_from_gaps(gaps)
    if pts is None:
        return 0.0
    # Hull area of points on unit circle is the area of the polygon formed by them.
    # For points on a circle, the convex hull is the polygon itself if they are in circular order.
    # Area of polygon with vertices on unit circle:
    # A = 0.5 * sum( sin(theta_{i+1} - theta_i) )
    # Since gaps are positive and sum to 2pi, this is the area.
    all_gaps = np.append(gaps, 2 * np.pi - np.sum(gaps))
    ha = 0.5 * np.sum(np.sin(all_gaps))
    
    if ha < 1e-12:
        return 0.0
        
    smta = _smooth_min_triangle_area(pts, beta)
    return smta / ha

def _neg_smooth_score_gaps_and_grad(gaps_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for optimizing gaps.
    Uses numerical gradient for the gap-space optimization to avoid complex derivative chaining.
    """
    gaps = gaps_flat.copy()
    # Ensure gaps are positive and sum < 2pi
    if np.any(gaps <= 0) or np.sum(gaps) >= 2 * np.pi:
        return 1e6, np.zeros_like(gaps_flat)
        
    score = _score_from_gaps(gaps, beta)
    if score == 0.0:
        return 1e6, np.zeros_like(gaps_flat)
        
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(gaps_flat)
    for i in range(len(gaps_flat)):
        g_plus = gaps_flat.copy()
        g_plus[i] += eps
        if g_plus[i] <= 0 or np.sum(g_plus) >= 2 * np.pi:
            grad[i] = 0
            continue
        s_plus = _score_from_gaps(g_plus, beta)
        
        g_minus = gaps_flat.copy()
        g_minus[i] -= eps
        if g_minus[i] <= 0 or np.sum(g_minus) >= 2 * np.pi:
            grad[i] = 0
            continue
        s_minus = _score_from_gaps(g_minus, beta)
        
        grad[i] = (s_plus - s_minus) / (2 * eps)
        
    return -score, -grad

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

    def _optimize_gaps(init_gaps, beta=100.0, maxiter=1000):
        """
        Optimize the angular gaps.
        """
        if time.time() > deadline:
            return
            
        x0 = init_gaps.copy()
        
        # Use L-BFGS-B with numerical gradient
        # Bounds: gaps must be > 0. Sum < 2pi.
        # L-BFGS-B doesn't handle linear constraints like sum < 2pi directly well with only bounds.
        # We can penalize violations in the objective.
        
        def _obj_gaps(x):
            if np.any(x <= 0) or np.sum(x) >= 2 * np.pi:
                return 1e6
            return -_score_from_gaps(x, beta)
            
        def _grad_gaps(x):
            if np.any(x <= 0) or np.sum(x) >= 2 * np.pi:
                return np.zeros_like(x)
            _, grad = _neg_smooth_score_gaps_and_grad(x, beta)
            return grad

        try:
            res = scipy.optimize.minimize(
                _obj_gaps,
                x0,
                jac=_grad_gaps,
                method='L-BFGS-B',
                bounds=[(1e-4, 2*np.pi)] * (N-1),
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = res.x
            if np.any(x_opt <= 0) or np.sum(x_opt) >= 2 * np.pi:
                return
                
            pts = _points_from_gaps(x_opt)
            if pts is not None:
                _update_best(pts)
                
        except Exception:
            pass

    # 1. Start with regular polygon gaps
    # Regular N-gon has gaps of 2*pi/N
    reg_gaps = np.full(N-1, 2 * np.pi / N)
    _optimize_gaps(reg_gaps, beta=50.0, maxiter=500)
    _optimize_gaps(reg_gaps, beta=100.0, maxiter=500)
    _optimize_gaps(reg_gaps, beta=200.0, maxiter=500)
    _optimize_gaps(reg_gaps, beta=500.0, maxiter=500)

    # 2. Random restarts in gap space
    rng = np.random.RandomState(42)
    for i in range(20):
        if time.time() > deadline - 5:
            break
            
        # Generate random gaps that sum to less than 2pi
        # Use Dirichlet distribution scaled to 2pi
        # Dirichlet generates vectors summing to 1
        # We need N gaps summing to 2pi.
        # Generate N-1 gaps from Dirichlet(alpha) * 2pi?
        # No, Dirichlet(alpha) of length N gives N values summing to 1.
        # Let's generate N-1 values.
        # Simpler: generate N-1 random positive numbers, normalize them to sum to 2pi * (1 - epsilon).
        
        raw_gaps = rng.uniform(0.1, 1.0, N-1)
        scale = (2 * np.pi * 0.9) / np.sum(raw_gaps)
        init_gaps = raw_gaps * scale
        
        _optimize_gaps(init_gaps, beta=100.0, maxiter=300)

    # 3. Fallback to regular polygon if gap optimization failed or didn't find anything
    reg_pts = _regular_polygon(N, 1.0)
    _update_best(reg_pts)

    if best_points is None:
        best_points = _normalize(reg_pts)
    else:
        best_points = _normalize(best_points)
        
    return best_points
