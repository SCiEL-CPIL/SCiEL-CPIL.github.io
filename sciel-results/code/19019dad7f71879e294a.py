"""
Combined program: Exact uncertainty bound with robust multi-stage optimization.
Uses the full moment calculation from the inspiration and the DE+L-BFGS-B structure from the current program.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Precompute constants for speed
# I[n] = integral x^(2n) e^(-x^2) dx = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 12 (since max power in P(x)^2 is x^24, so 2n=24 => n=12)
def _precompute_integrals():
    I = np.zeros(13)
    sp = math.sqrt(math.pi)
    for n in range(13):
        # (2n)! / (4^n n!)
        # Use log to avoid overflow in factorials for larger n, though n<=12 is safe for float64
        # (2n)! is approx 10^17 for n=12, which is fine.
        num = math.factorial(2 * n)
        den = (4.0 ** n) * math.factorial(n)
        I[n] = (num / den) * sp
    return I

_I_TABLE = _precompute_integrals()
_TWO_PI = 2.0 * math.pi

def _bound_and_grad(params):
    """
    Compute the exact uncertainty bound A(f)*A(F) and its gradient.
    params = [c0, c4, c8]
    c12 is determined by g(0)=0.
    
    g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^(-x^2/2)
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    g(0) = c0 + 3c4 + 105c8 + 945c12 = 0
    => c12 = -(c0 + 3c4 + 105c8) / 945
    """
    c0, c4, c8 = params[0], params[1], params[2]
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^6 p_k x^{2k}
    # H0: 1
    # H4: x^4 - 6x^2 + 3  -> coeffs: [3, -6, 1, 0, 0, 0, 0]
    # H8: x^8 - 28x^6 + 210x^4 - 420x^2 + 105 -> coeffs: [105, -420, 210, -28, 1, 0, 0]
    # H12: x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945 -> coeffs: [945, -7560, 13860, -6930, 1155, -66, 1]
    
    p = np.zeros(7)
    # p[0] (x^0)
    p[0] = c0 * 1.0 + c4 * 3.0 + c8 * 105.0 + c12 * 945.0
    # p[1] (x^2)
    p[1] = c4 * (-6.0) + c8 * (-420.0) + c12 * (-7560.0)
    # p[2] (x^4)
    p[2] = c4 * 1.0 + c8 * 210.0 + c12 * 13860.0
    # p[3] (x^6)
    p[3] = c8 * (-28.0) + c12 * (-6930.0)
    # p[4] (x^8)
    p[4] = c8 * 1.0 + c12 * 1155.0
    # p[5] (x^10)
    p[5] = c12 * (-66.0)
    # p[6] (x^12)
    p[6] = c12 * 1.0
    
    # Compute moments m_k = integral x^k |P(x)|^2 e^(-x^2) dx
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    # We need m_0 and m_2.
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            idx0 = i + j
            idx2 = i + j + 1
            if idx0 < 13:
                m0 += pij * _I_TABLE[idx0]
            if idx2 < 13:
                m2 += pij * _I_TABLE[idx2]
                
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    # Fourier side:
    # G(xi) = Fourier[g](xi). 
    # For g(x) = P(x)e^{-x^2/2}, G(xi) is proportional to Q(xi)e^{-xi^2/2} where Q is related to P.
    # Specifically, Fourier[x^n e^{-x^2/2}] involves derivatives of delta or Hermite functions.
    # A simpler identity: If g(x) = sum a_k H_k(x) e^{-x^2/2}, then \hat{g}(\xi) = sum a_k (-i)^k H_k(\xi) e^{-xi^2/2} (up to normalization).
    # Wait, standard Fourier transform convention: \int f(x) e^{-i x \xi} dx.
    # \hat{H_n(x) e^{-x^2/2}} = \sqrt{2\pi} (-i)^n H_n(\xi) e^{-\xi^2/2}.
    # So |G(\xi)|^2 = 2\pi |Q(\xi)|^2 e^{-\xi^2} where Q(\xi) = sum a_k (-1)^k H_k(\xi) (since (-i)^2n = (-1)^n for even n).
    # Actually, we only have even degrees 0,4,8,12. So (-i)^k = (-1)^{k/2}.
    # H0 -> 1
    # H4 -> (-1)^2 = 1
    # H8 -> (-1)^4 = 1
    # H12 -> (-1)^6 = 1
    # So Q(\xi) has the same polynomial coefficients as P(x) if we just map H_k(x) to H_k(\xi)?
    # Let's check: H_4(x) = x^4 - 6x^2 + 3.
    # Fourier of H_4(x)e^{-x^2/2} is \sqrt{2\pi} (1) H_4(\xi) e^{-\xi^2/2}.
    # So yes, the polynomial part Q(\xi) is identical to P(x) in terms of coefficients relative to the Hermite basis.
    # Thus, the moments n_k for the Fourier side are just 2\pi * m_k.
    
    n0 = _TWO_PI * m0
    n2 = _TWO_PI * m2
    
    if n0 <= 0 or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    var_x = m2 / m0
    var_xi = n2 / n0
    
    if var_x <= 0 or var_xi <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = math.sqrt(var_x * var_xi)
    
    # Gradient Calculation
    # bound = (var_x * var_xi)^0.5
    # dbound/dp = 0.5 * (var_x * var_xi)^-0.5 * (var_xi * dvar_x + var_x * dvar_xi)
    # dvar_x = (dm2/m0 - m2 dm0/m0^2) = (dm2 m0 - m2 dm0) / m0^2
    # dvar_xi = 2\pi * dvar_x (since n = 2\pi m, var_xi = var_x)
    # Wait, var_xi = n2/n0 = (2\pi m2)/(2\pi m0) = m2/m0 = var_x.
    # So var_x == var_xi.
    # bound = var_x.
    # dbound/dp = dvar_x/dp.
    
    # Let's re-verify: A(f) = sqrt(<x^2>/<1>). A(F) = sqrt(<xi^2>/<1>).
    # <xi^2> = 2\pi <x^2> (by Plancherel and scaling? No, \int xi^2 |G|^2 dxi = \int x^2 |g|^2 dx * 2\pi? 
    # \int |G|^2 dxi = 2\pi \int |g|^2 dx.
    # \int xi^2 |G|^2 dxi = \int x^2 |g|^2 dx * 2\pi?
    # Let's check dimensions. G(xi) = \int g(x) e^{-ix\xi} dx.
    # \xi G' ... 
    # Standard result: \langle \xi^2 \rangle = \langle x^2 \rangle for normalized functions?
    # No, \hat{g}(\xi) = \sqrt{2\pi} H_n(\xi) e^{-\xi^2/2}.
    # \int \xi^2 H_n(\xi)^2 e^{-\xi^2} d\xi = \int x^2 H_n(x)^2 e^{-x^2} dx.
    # So yes, the integrals are identical. The factor 2\pi cancels in the ratio.
    # So var_x = var_xi.
    # bound = var_x.
    
    # We need dvar_x / dparams.
    # var_x = m2/m0.
    # dvar_x = (dm2 m0 - m2 dm0) / m0^2.
    
    # We need dm0/dp_k and dm2/dp_k.
    # m_k = sum_{i,j} p_i p_j I[k+i+j].
    # dm_k/dp_a = sum_{i,j} (delta_{ia} p_j + p_i delta_{ja}) I[k+i+j] = 2 sum_j p_j I[k+a+j].
    
    # params are c0, c4, c8.
    # p is linear in c0, c4, c8 (and c12 which is linear in them).
    # dp/dc_a = sum_c (dp/dc_c) * (dc_c/dc_a).
    
    # Let's compute dp/dc for c0, c4, c8, c12.
    # p[0] = c0 + 3c4 + 105c8 + 945c12
    # p[1] = -6c4 - 420c8 - 7560c12
    # p[2] = c4 + 210c8 + 13860c12
    # p[3] = -28c8 - 6930c12
    # p[4] = c8 + 1155c12
    # p[5] = -66c12
    # p[6] = c12
    
    # dc12/dc0 = -1/945, dc12/dc4 = -3/945, dc12/dc8 = -105/945.
    
    dp_dc = np.zeros((7, 3)) # dp[polynomial_index] / dc[parameter_index]
    
    # Derivatives of c12
    dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    for a in range(3):
        # dp0/dc_a
        dp_dc[0, a] = np.array([1.0, 3.0, 105.0])[a] + 945.0 * dc12[a]
        # dp1/dc_a
        dp_dc[1, a] = np.array([0.0, -6.0, -420.0])[a] - 7560.0 * dc12[a]
        # dp2/dc_a
        dp_dc[2, a] = np.array([0.0, 1.0, 210.0])[a] + 13860.0 * dc12[a]
        # dp3/dc_a
        dp_dc[3, a] = np.array([0.0, 0.0, -28.0])[a] - 6930.0 * dc12[a]
        # dp4/dc_a
        dp_dc[4, a] = np.array([0.0, 0.0, 1.0])[a] + 1155.0 * dc12[a]
        # dp5/dc_a
        dp_dc[5, a] = 0.0 - 66.0 * dc12[a]
        # dp6/dc_a
        dp_dc[6, a] = 0.0 + 1.0 * dc12[a]
        
    # Precompute sums S_k[a] = sum_j p_j I[k+a+j]
    # dm_k/dp_a = 2 * S_k[a]
    # dm_k/dc_c = sum_a (dm_k/dp_a) * (dp_a/dc_c) = 2 * sum_a S_k[a] * dp_dc[a, c]
    
    grad = np.zeros(3)
    for c in range(3):
        dm0_dc = 0.0
        dm2_dc = 0.0
        for a in range(7):
            if dp_dc[a, c] == 0.0:
                continue
            # S_0[a] = sum_j p_j I[a+j]
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                if p[j] == 0.0:
                    continue
                idx0 = a + j
                idx2 = a + j + 1
                if idx0 < 13:
                    s0 += p[j] * _I_TABLE[idx0]
                if idx2 < 13:
                    s2 += p[j] * _I_TABLE[idx2]
            dm0_dc += 2.0 * s0 * dp_dc[a, c]
            dm2_dc += 2.0 * s2 * dp_dc[a, c]
            
        dvar_x = (dm2_dc * m0 - m2 * dm0_dc) / (m0 * m0)
        grad[c] = dvar_x
        
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # 1. Global search using Differential Evolution
    # Bounds based on expected scale
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    try:
        res_de = minimize(
            _objective, 
            x0, 
            bounds=bounds, 
            method='differential_evolution',
            options={
                'maxiter': 100, 
                'polish': True, 
                'init': 'sobol',
                'popsize': 15,
                'seed': 42,
                'tol': 1e-8
            }
        )
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_x = res_de.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 40:
        return best_x

    # 2. Local polish with L-BFGS-B from the best DE result
    try:
        res_local = minimize(_objective_grad, best_x, jac=True, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res_local.fun < best_score:
            best_score = res_local.fun
            best_x = res_local.x.copy()
    except Exception:
        pass
        
    # 3. Additional local polish from random points to ensure robustness
    np.random.seed(123)
    # Use a few random restarts
    for i in range(15):
        if time.time() > deadline - 10:
            break
        # Random start in a reasonable range
        x_start = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.05, 0.05)
        ])
        try:
            res_rand = minimize(_objective_grad, x_start, jac=True, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11})
            if res_rand.fun < best_score:
                best_score = res_rand.fun
                best_x = res_rand.x.copy()
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
