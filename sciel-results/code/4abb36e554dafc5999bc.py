"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 135  # seconds, safety margin from 150s limit

# Precompute triangle indices for vectorized area computation
_TRIS = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRIS = len(_TRIS)


def _compute_all_areas(points):
    """Vectorized computation of all triangle areas (normalized)."""
    x = points[:, 0]
    y = points[:, 1]
    # areas[t] = 0.5 * |cross(p_i, p_j, p_k)| for triangle t
    pi = points[_TRIS[:, 0]]
    pj = points[_TRIS[:, 1]]
    pk = points[_TRIS[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas / TRI_AREA


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    areas = _compute_all_areas(points)
    return float(np.min(areas)) if len(areas) > 0 else 0.0


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle with small margin."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True


def _soft_min_objective(x, beta):
    """
    Differentiable soft-min of negative triangle areas + boundary penalty.
    Uses log-sum-exp for smooth approximation of the min.
    """
    points = x.reshape(-1, 2)
    areas = _compute_all_areas(points)
    
    # Soft-min: -softmin(a) ≈ -(max(a) - (1/beta)*log(sum(exp(-beta*(a - max_a)))))
    # We want to maximize min(areas), so minimize -softmin(areas)
    a_max = np.max(areas)
    # Numerically stable log-sum-exp of -beta*(a - a_max)
    shifted = -beta * (areas - a_max)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = a_max - log_sum / beta
    
    # Boundary penalty
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        d_bottom = max(0.0, -py)
        d_top = max(0.0, py - S3H)
        d_left = max(0.0, -px)
        d_right = max(0.0, px - 1)
        d_slant_right = max(0.0, py - S3H * (1 - px))
        d_slant_left = max(0.0, py - S3H * px)
        penalty += d_bottom**2 + d_top**2 + d_left**2 + d_right**2 + d_slant_right**2 + d_slant_left**2
    
    return -soft_min_val + 50.0 * penalty


def _soft_min_gradient(x, beta):
    """Numerical gradient of the soft-min objective."""
    eps = 1e-7
    f0 = _soft_min_objective(x, beta)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_soft_min_objective(xp, beta) - f0) / eps
    return grad


def _anneal_optimize(x0, betas, maxiter_per_stage=3000):
    """
    Multi-scale beta annealing with L-BFGS-B.
    Start with low beta (smooth landscape) and increase for sharpness.
    """
    x_curr = x0.copy()
    for beta in betas:
        try:
            res = minimize(
                _soft_min_objective,
                x_curr,
                args=(beta,),
                method='L-BFGS-B',
                jac=_soft_min_gradient,
                options={'maxiter': maxiter_per_stage, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_curr = res.x
        except Exception:
            pass
    return x_curr


def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(123)
    
    def random_point_in_triangle():
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                if 0.005 < x < 0.995 and 0.005 < y < S3H * 0.995:
                    if y < S3H * (1 - x) * 0.995 and y < S3H * x * 0.995:
                        return [x, y]
    
    # Beta schedule: start smooth, end sharp
    # Low beta = smooth (find basin), high beta = sharp (refine)
    beta_schedule = [10, 30, 80, 200]
    
    n_restarts = 15
    
    for r in range(n_restarts):
        if elapsed() > DEADLINE - 15:
            break
        
        # Generate initial configuration
        if best_points is not None and r % 3 == 0:
            x0 = best_points.flatten() + rng.randn(N * 2) * 0.04
            for i in range(N):
                x0[2*i] = np.clip(x0[2*i], 0.01, 0.99)
                x0[2*i+1] = np.clip(x0[2*i+1], 0.01, S3H * 0.99)
        else:
            x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
        
        # Run annealed optimization
        try:
            x_opt = _anneal_optimize(x0, beta_schedule, maxiter_per_stage=2000)
        except Exception:
            continue
        
        if elapsed() > DEADLINE - 10:
            break
        
        pts = x_opt.reshape(-1, 2)
        
        # Check validity
        valid = all(_is_inside_triangle(p) for p in pts)
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    if best_points is None:
        best_points = np.array([random_point_in_triangle() for _ in range(N)])
        best_score = _min_area_normalized(best_points)
    
    # Final high-beta polish
    if elapsed() < DEADLINE - 3:
        try:
            x_final = _anneal_optimize(best_points.flatten(), [50, 200, 500], maxiter_per_stage=3000)
            pts_final = x_final.reshape(-1, 2)
            if all(_is_inside_triangle(p) for p in pts_final):
                score_final = _min_area_normalized(pts_final)
                if score_final > best_score:
                    best_points = pts_final
                    best_score = score_final
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
