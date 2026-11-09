import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0

    # Softmax for positive edge lengths (angular gaps) summing to 2*pi
    def softmax(u):
        exp_u = np.exp(u - np.max(u))
        return exp_u / np.sum(exp_u) * (2 * np.pi)

    def get_angles(u):
        # u has shape (7,)
        # Append 0 for the 8th component to fix the gauge
        u_full = np.append(u, 0.0)
        g = softmax(u_full)
        # Cumulative sum to get angles
        theta = np.cumsum(np.append(0.0, g))
        # theta[0] is 0, theta[7] is 2*pi (not used as point, theta[0..7] are 8 points? No)
        # Points are at angles theta[0], theta[1], ..., theta[7]?
        # We have 8 gaps g[0..7].
        # Angles:
        # a_0 = 0
        # a_1 = g_0
        # a_2 = g_0 + g_1
        # ...
        # a_7 = g_0 + ... + g_6
        # The 8th point is at a_7. The gap g_7 closes the circle.
        # So we need 8 angles: 0, g_0, g_0+g_1, ..., sum(g_0..g_6).
        angles = np.cumsum(np.insert(g, 0, 0.0))[:8]
        return angles

    def compute_score_log(u, alpha=500.0):
        angles = get_angles(u)
        # Compute pairwise angular differences
        # diff[i,j] = angle[i] - angle[j]
        diff = angles[:, np.newaxis] - angles[np.newaxis, :]
        # We want absolute difference in [0, 2*pi)
        # Actually, distance depends on sin(delta/2). sin is symmetric and periodic.
        # sin((a-b)/2) = sin((b-a)/2) for absolute?
        # sin(x) is not even. But sin(-x) = -sin(x).
        # Distance is 2R |sin((a-b)/2)|.
        # So we need |sin(diff/2)|.
        
        # To avoid numerical issues with max, we compute sines
        sines = np.abs(np.sin(diff / 2.0))
        
        # Mask diagonal
        mask = ~np.eye(8, dtype=bool)
        sines_masked = sines[mask]
        
        # Log product of sines
        # If any sine is 0, log is -inf
        if np.any(sines_masked < 1e-12):
            return -np.inf
        
        log_prod_sines = np.sum(np.log(sines_masked))
        
        # Max sine (Diameter term)
        # Use soft-max for smoothness
        max_sine = np.max(sines_masked)
        # Soft max: log(sum(exp(alpha * (s - max)))) / alpha + max
        # Actually we want log(max(sines)).
        # Approximation: log( (sum s^alpha)^(1/alpha) ) = (1/alpha) log(sum s^alpha)
        # For large alpha, this approaches log(max).
        # Let's use a high alpha.
        alpha_eff = 500.0
        # To avoid overflow in exp, shift by max
        s_shifted = sines_masked - max_sine
        exp_terms = np.exp(alpha_eff * s_shifted)
        sum_exp = np.sum(exp_terms)
        log_sum_exp = np.log(sum_exp)
        log_max_sine = (log_sum_exp + alpha_eff * max_sine) / alpha_eff
        
        # Score log = log_prod_sines - 28 * log_max_sine
        # Note: The (2R)^28 term is constant after scaling, so we ignore it.
        # The scaling R = 1 / max_sine (since D=2, 2R*max_sine = 2 => R=1/max_sine)
        # Product of distances = (2R)^28 * prod(sines)
        # Log Product = 28 log(2R) + log_prod_sines
        # = 28 (log 2 - log(max_sine)) + log_prod_sines
        # = 28 log 2 - 28 log(max_sine) + log_prod_sines
        # Constant 28 log 2 doesn't matter for optimization.
        
        score_log = log_prod_sines - 28.0 * log_max_sine
        return score_log

    def objective(u):
        score_log = compute_score_log(u)
        if np.isinf(score_log):
            return 1e10
        return -score_log

    def gradient(u):
        # Numerical gradient is safer for this complex function initially,
        # but analytical is better. Let's try numerical first for robustness,
        # or just use L-BFGS-B with finite differences if jac is not provided.
        # L-BFGS-B default is finite difference. It's slow but robust.
        # Given 7 variables, it's fine.
        pass

    # Initial guess: Regular octagon
    # Angles: 0, pi/4, ..., 7pi/4
    # Gaps: all pi/4
    # u_i such that softmax(u) = 1/8
    # u_i = 0 for all i gives uniform distribution.
    u0 = np.zeros(7)
    
    best_points = None
    best_score = -np.inf
    
    # Run optimization
    try:
        res = minimize(objective, u0, method='L-BFGS-B', options={'maxiter': 10000, 'ftol': 1e-14})
        if res.success or res.fun < 1e9:
            u_opt = res.x
            score_log = compute_score_log(u_opt)
            if score_log > best_score:
                best_score = score_log
                best_points = u_opt
    except Exception as e:
        pass

    # Try a few random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        u_rand = rng.normal(0, 1, 7)
        try:
            res = minimize(objective, u_rand, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success or res.fun < 1e9:
                u_opt = res.x
                score_log = compute_score_log(u_opt)
                if score_log > best_score:
                    best_score = score_log
                    best_points = u_opt
        except Exception:
            pass

    if best_points is None:
        best_points = u0

    # Convert back to points
    angles = get_angles(best_points)
    # Compute R
    diff = angles[:, np.newaxis] - angles[np.newaxis, :]
    sines = np.abs(np.sin(diff / 2.0))
    mask = ~np.eye(8, dtype=bool)
    max_sine = np.max(sines[mask])
    R = 1.0 / max_sine
    
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    points = np.column_stack((x, y))
    
    return points.tolist()
