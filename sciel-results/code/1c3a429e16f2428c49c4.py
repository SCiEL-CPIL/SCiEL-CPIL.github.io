"""
Optimized approach using Chebyshev basis expansion and L-BFGS-B with soft-max surrogate.
The objective is value = abs(2 * N * max(convolve(h, h)) / sum(h)^2).
We parameterize h as a linear combination of Chebyshev polynomials to enforce smoothness
and reduce dimensionality, then optimize the coefficients.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 150.0
START_TIME = time.time()

# Precompute Chebyshev basis for efficiency
_x = np.linspace(-1, 1, N)
_cheb_basis = np.array([eval_chebyt(k, _x) for k in range(30)])  # up to degree 29

def _value(h):
    s = np.sum(h)
    if np.isclose(s, 0):
        return 1e10
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _soft_max(x, beta=100.0):
    m = np.max(x)
    return m + np.log(np.sum(np.exp((x - m) * beta)) / len(x)) / beta

def _coeffs_to_h(coeffs):
    """Convert Chebyshev coefficients to step heights h."""
    h = np.zeros(N)
    for k, c in enumerate(coeffs):
        if c != 0:
            h += c * _cheb_basis[k]
    return h

def _objective_and_grad(coeffs):
    """Compute objective and gradient using soft-max surrogate."""
    h = _coeffs_to_h(coeffs)
    s = np.sum(h)
    if np.isclose(s, 0):
        return 1e10, np.zeros_like(coeffs)
    
    c = np.convolve(h, h)
    beta = 50.0
    m = np.max(c)
    # Soft max
    exp_c = np.exp((c - m) * beta)
    sum_exp = np.sum(exp_c)
    soft_m = m + np.log(sum_exp) / beta
    
    # Objective
    obj = abs(2.0 * N * soft_m / (s * s))
    
    # Gradient with respect to h
    # d(soft_m)/d(c_i) = exp((c_i - m)*beta) / sum_exp
    dc_dh = np.zeros_like(h)
    # c = convolve(h, h), so c[k] = sum_i h[i] * h[k-i]
    # For full convolution, c has length 2N-1
    # dc[k]/dh[j] = h[k-j] if 0 <= k-j < N, else 0
    # We need d(soft_m)/dh[j] = sum_k (d soft_m / dc[k]) * (dc[k]/dh[j])
    
    # Efficient computation using correlation
    weights = exp_c / sum_exp  # d(soft_m)/dc[k]
    # d(soft_m)/dh[j] = sum_k weights[k] * dc[k]/dh[j]
    # dc[k]/dh[j] = h[k-j] (for valid indices)
    # This is essentially a cross-correlation
    
    # Use FFT for efficiency
    weights_padded = np.zeros(2 * N - 1)
    weights_padded[:] = weights
    
    # d(soft_m)/dh = ifft(fft(h) * fft(weights_padded)) -- but need to be careful
    # Actually, let's compute directly since N=400 is small
    dm_dh = np.zeros(N)
    for j in range(N):
        # dc[k]/dh[j] = h[k-j] for k-j in [0, N-1]
        # k ranges from j to j+N-1
        k_start = j
        k_end = j + N
        if k_end > len(weights):
            k_end = len(weights)
        if k_start < len(weights):
            dm_dh[j] = np.sum(weights[k_start:k_end] * h[:k_end - k_start])
    
    # d(obj)/dh = d(obj)/d(soft_m) * d(soft_m)/dh + d(obj)/d(s) * d(s)/dh
    # obj = 2*N*soft_m/s^2 (assuming positive)
    d_obj_d_soft_m = 2.0 * N / (s * s)
    d_obj_d_s = -4.0 * N * soft_m / (s * s * s)
    
    grad_h = d_obj_d_soft_m * dm_dh + d_obj_d_s * 1.0
    
    # Chain rule: grad_coeffs[k] = sum_j grad_h[j] * d(h[j])/d(coeffs[k])
    # h[j] = sum_k coeffs[k] * T_k(x_j)
    # d(h[j])/d(coeffs[k]) = T_k(x_j)
    grad_coeffs = np.zeros_like(coeffs)
    for k in range(len(coeffs)):
        if coeffs[k] != 0:
            grad_coeffs[k] = np.sum(grad_h * _cheb_basis[k])
    
    return obj, grad_coeffs

def _objective_soft(coeffs):
    obj, _ = _objective_and_grad(coeffs)
    return obj

def _make_initial_conditions(rng, num_terms):
    """Create diverse initial conditions."""
    inits = []
    
    # 1. Constant offset
    c1 = np.zeros(num_terms)
    c1[0] = 1.0
    inits.append(c1)
    
    # 2. Small random perturbation
    c2 = np.zeros(num_terms)
    c2[0] = 1.0
    c2[1:] = rng.normal(0, 0.1, num_terms - 1)
    inits.append(c2)
    
    # 3. Bump-like (low frequency dominant)
    c3 = np.zeros(num_terms)
    c3[0] = 1.0
    c3[1] = 0.3
    c3[2] = 0.1
    inits.append(c3)
    
    # 4. Fourier-like (mix of frequencies)
    c4 = np.zeros(num_terms)
    c4[0] = 1.0
    for k in range(1, min(5, num_terms)):
        c4[k] = 0.05 * (-1)**k
    inits.append(c4)
    
    # 5. Block-like (high offset, specific mode)
    c5 = np.zeros(num_terms)
    c5[0] = 1.0
    c5[3] = 0.2
    inits.append(c5)
    
    return inits

def construct():
    if time.time() - START_TIME > DEADLINE - 5:
        return np.ones(N)
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    num_terms = 10
    rng = np.random.default_rng(42)
    
    inits = _make_initial_conditions(rng, num_terms)
    
    for init in inits:
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        try:
            res = minimize(_objective_soft, init, jac=_objective_and_grad,
                          method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            
            h_candidate = _coeffs_to_h(res.x)
            val_candidate = _value(h_candidate)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate.copy()
                
        except Exception:
            continue
    
    # Final polish: use the best h found and do a local optimization in h-space
    if time.time() - START_TIME > DEADLINE - 10:
        return best_h
    
    # Try a few more random restarts if time permits
    for _ in range(3):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        coeffs0 = np.zeros(num_terms)
        coeffs0[0] = 1.0
        coeffs0[1:] = rng.normal(0, 0.05, num_terms - 1)
        
        try:
            res = minimize(_objective_soft, coeffs0, jac=_objective_and_grad,
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            h_candidate = _coeffs_to_h(res.x)
            val_candidate = _value(h_candidate)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate.copy()
                
        except Exception:
            continue
    
    return best_h
