"""Fast spectral moment optimization using Gauss-Hermite quadrature."""

import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = time.time() + 160

# Pre-compute Gauss-Hermite quadrature nodes and weights
# We need to integrate up to x^24 roughly, so high order is good.
# N=100 is usually sufficient for double precision accuracy for smooth functions.
_N_QUAD = 100
_x_nodes, _w_weights = np.polynomial.hermite.hermgauss(_N_QUAD)

# Pre-compute powers of x for the polynomial basis
# P(x) = sum_{k=0}^6 p_k x^{2k}
# We need x^{2(i+j)} for i,j in 0..6. Max power is 24.
# Let's pre-compute X_powers[k] = x_i^{2k} for k=0..12 (to cover products up to 12+12=24? No, P is deg 12, so P^2 is deg 24. 
# Wait, P(x) is sum_{k=0}^6 p_k x^{2k}. The highest power in P is 12.
# So P(x)^2 has highest power 24.
# We need moments m_k = \int x^{2k} P(x)^2 e^{-x^2} dx.
# Max power in integrand for m_2 is 2 + 24 = 26.
# So we need x^{2k} up to 26? 
# Actually, we can just compute P(x_i)^2 * x_i^{2k} * w_i.
# Let's pre-compute X_even_powers[k] = x_i^{2k} for k=0..13 (26/2=13).

_X_powers = np.zeros((_N_QUAD, 14)) # indices 0..13
for k in range(14):
    _X_powers[:, k] = _x_nodes ** (2 * k)

# Pre-compute weight times x^{2k} for m_k
# m_k = sum_i w_i x_i^{2k} P(x_i)^2
# Let W_k[i] = w_i * x_i^{2k}
_W_moments = np.zeros((_N_QUAD, 3)) # k=0,1,2
for k in range(3):
    _W_moments[:, k] = _w_weights * _x_nodes ** (2 * k)

# For Fourier side:
# Q(xi) = sum_{k=0}^6 p_k (-1)^k xi^{2k}
# n_k = 2*pi * sum_i w_i xi_i^{2k} Q(xi_i)^2
# Same nodes and weights apply.
_W_fourier_moments = np.zeros((_N_QUAD, 3)) # k=0,1,2
for k in range(3):
    _W_fourier_moments[:, k] = 2.0 * np.pi * _w_weights * _x_nodes ** (2 * k)


def _make_p(a0, a4, a8):
    """
    Compute polynomial coefficients p[0..6] for P(x) = sum p_k x^{2k}.
    a12 is fixed such that g(0)=P(0)=0.
    g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) e^{-x^2/2}
    H0 = 1
    H4 = 16x^4 - 48x^2 + 12
    H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112
    H12 = 4096x^12 - 32256x^10 + 90720x^8 - 107520x^6 + 51840x^4 - 10368x^2 + 945
    
    P(x) = a0*1 + a4*(16x^4 - 48x^2 + 12) + a8*(256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112) + a12*(4096x^12 - 32256x^10 + 90720x^8 - 107520x^6 + 51840x^4 - 10368x^2 + 945)
    
    Coefficients p_k for x^{2k}:
    p0 (x^0): a0 + 12a4 + 112a8 + 945a12
    p1 (x^2): -48a4 - 1344a8 - 10368a12
    p2 (x^4): 16a4 + 2688a8 + 51840a12
    p3 (x^6): -1344a8 - 107520a12
    p4 (x^8): 256a8 + 90720a12
    p5 (x^10): -32256a12
    p6 (x^12): 4096a12
    
    Condition g(0)=0 => P(0)=p0=0.
    a0 + 12a4 + 112a8 + 945a12 = 0
    a12 = -(a0 + 12a4 + 112a8) / 945
    """
    a12 = -(a0 + 12.0 * a4 + 112.0 * a8) / 945.0
    
    p = np.zeros(7)
    p[0] = 0.0 # Enforced
    p[1] = -48.0 * a4 - 1344.0 * a8 - 10368.0 * a12
    p[2] = 16.0 * a4 + 2688.0 * a8 + 51840.0 * a12
    p[3] = -1344.0 * a8 - 107520.0 * a12
    p[4] = 256.0 * a8 + 90720.0 * a12
    p[5] = -32256.0 * a12
    p[6] = 4096.0 * a12
    
    return p


def _bound_value_fast(params):
    """
    Compute the uncertainty bound using pre-computed quadrature.
    """
    a0, a4, a8 = params
    p = _make_p(a0, a4, a8)
    
    # Evaluate P(x) at nodes
    # P(x) = sum_{k=0}^6 p_k x^{2k}
    # We have _X_powers[:, k] = x_i^{2k}
    P_vals = _X_powers @ p
    
    # Compute m_k = sum w_i x_i^{2k} P(x_i)^2
    P_sq = P_vals ** 2
    m0 = np.dot(P_sq, _w_weights)
    m1 = np.dot(P_sq, _w_weights * _x_nodes**2)
    m2 = np.dot(P_sq, _w_weights * _x_nodes**4)
    
    if m0 <= 1e-15 or not np.isfinite(m0):
        return 1e18
        
    vf = m1 / m0
    
    # Evaluate Q(xi) at nodes
    # Q(xi) = sum_{k=0}^6 p_k (-1)^k xi^{2k}
    # Let q_k = p_k * (-1)^k
    q_coeffs = p * ((-1.0) ** np.arange(7))
    Q_vals = _X_powers @ q_coeffs
    
    Q_sq = Q_vals ** 2
    n0 = np.dot(Q_sq, _w_weights)
    n1 = np.dot(Q_sq, _w_weights * _x_nodes**2)
    n2 = np.dot(Q_sq, _w_weights * _x_nodes**4)
    
    # Note: The factor 2*pi is in the definition of n_k in the prompt?
    # "n_k = 2*pi * integral ..."
    # In the previous code, _integrate_fourier_moments included 2*pi inside the integral result.
    # So n0 here should be multiplied by 2*pi?
    # Let's check the previous code:
    # val, err = quad(integrand_k...) where integrand has 2.0 * np.pi
    # So yes, n0, n1, n2 need to be scaled by 2*pi.
    n0 *= 2.0 * np.pi
    n1 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if n0 <= 1e-15 or not np.isfinite(n0):
        return 1e18
        
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)


def construct():
    """
    Find optimal Hermite coefficients to minimize the uncertainty bound.
    Returns array of three coefficients [a0, a4, a8].
    """
    # Seed starting point from previous best
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [seed]
    
    # Structured perturbations
    for scale in [0.01, 0.1, 0.5, 1.0]:
        for _ in range(10):
            starts.append(seed + rng.normal(0, scale, 3))
    
    # Random starts in reasonable range
    for _ in range(20):
        starts.append(rng.uniform(-5, 5, 3))
    
    best_val = _bound_value_fast(seed)
    best_params = seed.copy()
    
    # Optimization loop
    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - 20:
            break
            
        # Use Nelder-Mead as it doesn't require gradients and is robust for this small dimension
        # With fast evaluation, we can afford many iterations
        try:
            res = minimize(
                _bound_value_fast, start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'initial_simplex': 0.1}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Occasionally try L-BFGS-B from the best found so far
        if i % 5 == 0 and time.time() < _DEADLINE - 30:
            try:
                res2 = minimize(
                    _bound_value_fast, best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res2.fun < best_val and np.isfinite(res2.fun):
                    best_val = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                _bound_value_fast, best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None or not np.all(np.isfinite(best_params)):
        return seed
        
    return np.array(best_params)
