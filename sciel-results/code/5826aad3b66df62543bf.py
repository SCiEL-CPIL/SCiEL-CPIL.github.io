import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _gaussian_moment(m):
    """G[m] = integral_{-inf}^{inf} x^{2m} exp(-x^2) dx = sqrt(pi) * (2m)!/(4^m m!)"""
    if m == 0:
        return np.sqrt(np.pi)
    result = np.sqrt(np.pi)
    for i in range(1, m + 1):
        result *= (2 * i - 1) / 2.0
    return result


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product for g(x) = P(x) * exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12, with a12 chosen so P(0)=0.
    Returns the uncertainty product A(f)*A(F) = sqrt(M_2/M_0) * sqrt(N_2/M_0)
    where M_0 = ∫|g|², M_2 = ∫x²|g|², N_2 = ∫|g'|².
    """
    if time.time() > _DEADLINE:
        return np.inf
    
    # H_n(0) values
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 945
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 945.0
    
    # Build P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0: [1]
    # H4: [3, -6, 1] for x^0, x^2, x^4
    # H8: [105, -420, 210, -28, 1] for x^0, x^2, x^4, x^6, x^8
    # H12: [945, -7560, 13860, -6930, 1155, -66, 1] for x^0,...,x^12
    
    p = np.zeros(7)
    # k=0: a0*1 + a4*3 + a8*105 + a12*945 = 0 (by constraint)
    p[0] = 0.0
    p[1] = a4 * (-6.0) + a8 * (-420.0) + a12 * (-7560.0)
    p[2] = a4 * 1.0 + a8 * 210.0 + a12 * 13860.0
    p[3] = a8 * (-28.0) + a12 * (-6930.0)
    p[4] = a8 * 1.0 + a12 * 1155.0
    p[5] = a12 * (-66.0)
    p[6] = a12 * 1.0
    
    # M_0 = ∫ P(x)² exp(-x²) dx = sum_{j,k} p_j p_k G[j+k]
    # M_2 = ∫ x² P(x)² exp(-x²) dx = sum_{j,k} p_j p_k G[j+k+1]
    n_p = len(p)
    M_0 = 0.0
    M_2 = 0.0
    for j in range(n_p):
        if p[j] == 0.0:
            continue
        for k in range(n_p):
            if p[k] == 0.0:
                continue
            val = p[j] * p[k]
            M_0 += val * _gaussian_moment(j + k)
            M_2 += val * _gaussian_moment(j + k + 1)
    
    if M_0 <= 0:
        return np.inf
    
    # N_2 = ∫ |g'(x)|² dx where g(x) = P(x) exp(-x²/2)
    # g'(x) = [P'(x) - x P(x)] exp(-x²/2)
    # |g'|² = [P'(x) - x P(x)]² exp(-x²)
    # Q(x) = P'(x) - x P(x)
    # P(x) = sum p_k x^{2k}
    # P'(x) = sum_{k=1}^{6} 2k p_k x^{2k-1}
    # x P(x) = sum_{k=0}^{6} p_k x^{2k+1}
    # Q(x) = -p_0 x + sum_{k=1}^{6} (2k p_k) x^{2k-1} - sum_{k=0}^{6} p_k x^{2k+1}
    # Q(x) is an odd polynomial: Q(x) = sum_{m=0}^{5} q_m x^{2m+1}
    # Let's find q_m:
    # Term x^1: from P': 2*1*p_1, from -xP: -p_0 => q_0 = 2*p_1 - p_0
    # Term x^3: from P': 4*p_2, from -xP: -p_1 => q_1 = 4*p_2 - p_1
    # Term x^5: from P': 6*p_3, from -xP: -p_2 => q_2 = 6*p_3 - p_2
    # Term x^7: from P': 8*p_4, from -xP: -p_3 => q_3 = 8*p_4 - p_3
    # Term x^9: from P': 10*p_5, from -xP: -p_4 => q_4 = 10*p_5 - p_4
    # Term x^11: from P': 12*p_6, from -xP: -p_5 => q_5 = 12*p_6 - p_5
    # (No x^13 term since P' has no x^13 and xP has no x^13 for k=6 gives x^13? Wait)
    # Actually x*P(x) has terms up to x^{2*6+1} = x^{13}, and P'(x) has terms up to x^{2*6-1} = x^{11}
    # So we also have: Term x^13: from -xP: -p_6 => q_6 = -p_6
    
    q = np.zeros(7)  # q_m for x^{2m+1}, m=0..6
    q[0] = 2.0 * p[1] - p[0]
    q[1] = 4.0 * p[2] - p[1]
    q[2] = 6.0 * p[3] - p[2]
    q[3] = 8.0 * p[4] - p[3]
    q[4] = 10.0 * p[5] - p[4]
    q[5] = 12.0 * p[6] - p[5]
    q[6] = -p[6]
    
    # N_2 = ∫ Q(x)² exp(-x²) dx
    # Q(x) = sum_{m=0}^{6} q_m x^{2m+1}
    # Q(x)² = sum_{m,n} q_m q_n x^{2(m+n+1)}
    # ∫ x^{2(m+n+1)} exp(-x²) dx = G[m+n+1]
    n_q = len(q)
    N_2 = 0.0
    for m in range(n_q):
        if q[m] == 0.0:
            continue
        for n in range(n_q):
            if q[n] == 0.0:
                continue
            N_2 += q[m] * q[n] * _gaussian_moment(m + n + 1)
    
    if N_2 < 0:
        N_2 = 0.0
    
    dx = np.sqrt(M_2 / M_0)
    dp = np.sqrt(N_2 / M_0)
    return dx * dp


def _decode_params(params):
    """Decode log-magnitude + sign parameters to (a0, a4, a8)."""
    # params = [s0, m0, s4, m4, s8, m8] where s_i in {-1,1} approx, m_i real
    # Use soft sign: a_i = tanh(s_i) * exp(m_i)
    s0, m0, s4, m4, s8, m8 = params
    a0 = np.tanh(s0) * np.exp(m0)
    a4 = np.tanh(s4) * np.exp(m4)
    a8 = np.tanh(s8) * np.exp(m8)
    return a0, a4, a8


def construct():
    """Find optimal coefficients to minimize the uncertainty product."""
    start_time = time.time()
    
    # Initial guess from the seed (known SOTA)
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_bound(*best_coeffs)
    
    # Encode initial guess into log-magnitude + sign space
    def encode(a):
        if abs(a) < 1e-15:
            return 0.0, 0.0
        return np.sign(a) * 2.0, np.log(abs(a))
    
    s0, m0 = encode(best_coeffs[0])
    s4, m4 = encode(best_coeffs[1])
    s8, m8 = encode(best_coeffs[2])
    best_params = np.array([s0, m0, s4, m4, s8, m8])
    
    # Objective in parameter space
    def objective(params):
        if time.time() > _DEADLINE - 5:
            return np.inf
        a0, a4, a8 = _decode_params(params)
        return _compute_bound(a0, a4, a8)
    
    # Multi-start optimization in parameter space
    rng = np.random.default_rng(12345)
    
    # Try several random restarts in parameter space
    for restart in range(20):
        if time.time() > _DEADLINE - 30:
            break
        
        if restart == 0:
            x0 = best_params.copy()
        else:
            # Random perturbation in parameter space
            x0 = best_params + rng.normal(0, 0.5, 6)
            # Also try some structured initializations
            if restart == 1:
                x0 = np.array([0.0, np.log(0.5), 0.0, np.log(0.1), 0.0, np.log(0.01)])
            elif restart == 2:
                x0 = np.array([-2.0, np.log(1.0), 2.0, np.log(0.5), -2.0, np.log(0.1)])
            elif restart == 3:
                x0 = np.array([2.0, np.log(1.0), -2.0, np.log(0.5), 2.0, np.log(0.1)])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
                a0, a4, a8 = _decode_params(best_params)
                best_coeffs = np.array([a0, a4, a8])
        except Exception:
            pass
    
    # Also try Nelder-Mead on the original coefficients for refinement
    if time.time() < _DEADLINE - 20:
        def obj_direct(x):
            if time.time() > _DEADLINE - 5:
                return np.inf
            return _compute_bound(x[0], x[1], x[2])
        
        try:
            result = minimize(obj_direct, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
