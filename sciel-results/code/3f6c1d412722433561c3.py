"""Rewritten program with coordinate descent exploiting separable structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Precompute moments of Gaussian e^{-2*pi*x^2}
def _get_moments(max_power=26):
    moments = {}
    a = 2 * np.pi
    sqrt_pi_over_a = np.sqrt(np.pi / a)
    for m in range(max_power // 2 + 1):
        k = 2 * m
        if m == 0:
            val = sqrt_pi_over_a
        else:
            double_fact = 1
            for i in range(1, m + 1):
                double_fact *= (2 * i - 1)
            val = double_fact / ((2 * a) ** m) * sqrt_pi_over_a
        moments[k] = val
    return moments

MOMENTS = _get_moments(26)

# Coefficients for Hermite polynomials in monomial basis (even powers only)
# H_0: [1]
# H_4: [12, -48, 16]
# H_8: [3024, -20160, 20160, -4608, 256]
# H_12: [665280, -7983360, 8316000, -7096320, 59400, -135168, 4096]
H_COEFFS = {
    0: np.array([1.0]),
    4: np.array([12.0, -48.0, 16.0]),
    8: np.array([3024.0, -20160.0, 20160.0, -4608.0, 256.0]),
    12: np.array([665280.0, -7983360.0, 8316000.0, -7096320.0, 59400.0, -135168.0, 4096.0])
}

H_VALS_0 = {0: 1.0, 4: 12.0, 8: 3024.0, 12: 665280.0}

def _poly_to_monomial(coeffs_dict):
    max_deg = max(coeffs_dict.keys())
    mono_coeffs = np.zeros(max_deg // 2 + 1)
    for deg, c in coeffs_dict.items():
        h_coeffs = H_COEFFS[deg]
        for i, hc in enumerate(h_coeffs):
            mono_coeffs[i] += c * hc
    return mono_coeffs

def _calc_variance(mono_coeffs):
    L = len(mono_coeffs)
    square_coeffs = np.zeros(2 * L - 1)
    for j in range(2 * L - 1):
        s = 0.0
        for i in range(max(0, j - L + 1), min(L, j + 1)):
            k = j - i
            s += mono_coeffs[i] * mono_coeffs[k]
        square_coeffs[j] = s
    norm = 0.0
    second_moment = 0.0
    for j, c_sq in enumerate(square_coeffs):
        if c_sq == 0.0:
            continue
        power = 2 * j
        if power in MOMENTS:
            norm += c_sq * MOMENTS[power]
            power2 = 2 * j + 2
            if power2 in MOMENTS:
                second_moment += c_sq * MOMENTS[power2]
    return norm, second_moment

def _compute_score(c0, c4, c8):
    c12 = -(c0 * H_VALS_0[0] + c4 * H_VALS_0[4] + c8 * H_VALS_0[8]) / H_VALS_0[12]
    coeffs_pos = {0: c0, 4: c4, 8: c8, 12: c12}
    mono_pos = _poly_to_monomial(coeffs_pos)
    coeffs_fourier = {0: c0, 4: -c4, 8: c8, 12: -c12}
    mono_fourier = _poly_to_monomial(coeffs_fourier)
    norm_x, second_moment_x = _calc_variance(mono_pos)
    norm_xi, second_moment_xi = _calc_variance(mono_fourier)
    if norm_x <= 0 or norm_xi <= 0:
        return 1e10
    var_x = second_moment_x / norm_x
    var_xi = second_moment_xi / norm_xi
    return np.sqrt(var_x * var_xi)

def _coordinate_descent(x0, max_iters=5000, tol=1e-14, deadline=None):
    """
    Coordinate descent: for each coordinate, solve the 1D subproblem exactly
    using the rational structure of the objective.
    """
    x = x0.copy()
    best_score = _compute_score(*x)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for coord in range(3):
            # Fix other two coordinates, optimize over x[coord]
            # The objective as a function of one variable (with others fixed)
            # is sqrt(V_x(t) * V_xi(t)) where V_x and V_xi are ratios of
            # quadratics in t. So we need to minimize V_x(t)*V_xi(t).
            
            # Sample the 1D function to find a good bracket, then use golden section
            # or scipy's minimize_scalar
            x_fixed = x.copy()
            base_val = x[coord]
            
            def f_1d(t):
                x_fixed[coord] = t
                return _compute_score(*x_fixed)
            
            # Try a range of values
            # Based on seed values: c0 ~ -1.13, c4 ~ 0.04, c8 ~ 0.0003
            if coord == 0:
                t_min, t_max = -3.0, 0.5
            elif coord == 1:
                t_min, t_max = -0.2, 0.2
            else:
                t_min, t_max = -0.005, 0.005
            
            # Use scipy minimize_scalar for the 1D subproblem
            try:
                res = opt.minimize_scalar(f_1d, bounds=(t_min, t_max), method='bounded',
                                         options={'xatol': 1e-16, 'maxiter': 200})
                if res.fun < best_score:
                    x[coord] = res.x
                    best_score = res.fun
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin for 180s limit
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = _compute_score(*x0)
    best_x = x0.copy()
    
    # 1. Coordinate descent from seed
    x_cd, score_cd = _coordinate_descent(x0, max_iters=10000, tol=1e-15, deadline=deadline - 30.0)
    if score_cd < best_score:
        best_score = score_cd
        best_x = x_cd.copy()
    
    # 2. L-BFGS-B from the coordinate descent result
    if time.time() < deadline - 20.0:
        try:
            res = opt.minimize(_compute_score, best_x, method='L-BFGS-B',
                               options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # 3. More coordinate descent from the L-BFGS result
    if time.time() < deadline - 10.0:
        x_cd2, score_cd2 = _coordinate_descent(best_x, max_iters=10000, tol=1e-16, deadline=deadline - 5.0)
        if score_cd2 < best_score:
            best_score = score_cd2
            best_x = x_cd2.copy()
    
    # 4. Try multiple starting points with coordinate descent
    if time.time() < deadline - 10.0:
        start_points = [
            np.array([-1.0, 0.0, 0.0]),
            np.array([-1.2, 0.05, 0.0]),
            np.array([-1.1, 0.03, 0.001]),
            np.array([-0.9, 0.02, 0.0005]),
            np.array([-1.3, 0.04, 0.0002]),
        ]
        for sp in start_points:
            if time.time() > deadline - 5.0:
                break
            x_cd, score_cd = _coordinate_descent(sp, max_iters=5000, tol=1e-14, deadline=deadline - 2.0)
            if score_cd < best_score:
                best_score = score_cd
                best_x = x_cd.copy()
    
    # 5. Final high-precision polish
    if time.time() < deadline - 2.0 and best_score < 1e10:
        try:
            res_final = opt.minimize(_compute_score, best_x, method='Powell',
                                     options={'maxiter': 5000, 'ftol': 1e-20, 'xtol': 1e-18})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
        
        # Final coordinate descent polish
        if time.time() < deadline - 1.0:
            x_cd3, score_cd3 = _coordinate_descent(best_x, max_iters=10000, tol=1e-16, deadline=deadline - 0.5)
            if score_cd3 < best_score:
                best_score = score_cd3
                best_x = x_cd3.copy()
    
    return best_x

# EVOLVE-BLOCK-END
