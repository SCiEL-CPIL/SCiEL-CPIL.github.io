"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200  # 10s margin before 210s kill

# Physicist's Hermite polynomials evaluated at 0
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 945.0

# Seed coefficients (Goncalves et al. 2017)
_SEED = np.array([-1.13, 0.04, 1.0 / 3240.0])

# Precompute coefficients for Hermite polynomials H_0, H_4, H_8, H_12
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
# We represent P(x) = sum p_k x^k. Only even k are non-zero.
# p is an array of length 13 (indices 0..12, step 2)
_H_COEFFS = np.zeros((4, 13))
_H_COEFFS[0, 0] = 1.0
_H_COEFFS[1, 4] = 1.0; _H_COEFFS[1, 2] = -6.0; _H_COEFFS[1, 0] = 3.0
_H_COEFFS[2, 8] = 1.0; _H_COEFFS[2, 6] = -28.0; _H_COEFFS[2, 4] = 210.0; _H_COEFFS[2, 2] = -420.0; _H_COEFFS[2, 0] = 105.0
_H_COEFFS[3, 12] = 1.0; _H_COEFFS[3, 10] = -66.0; _H_COEFFS[3, 8] = 1155.0; _H_COEFFS[3, 6] = -6930.0; _H_COEFFS[3, 4] = 13860.0; _H_COEFFS[3, 2] = -7560.0; _H_COEFFS[3, 0] = 945.0

# Precompute integrals I_k = int x^(2k) exp(-x^2) dx for k=0..13
# I_k = (2k-1)!! / 2^k * sqrt(pi)
def _factorial_double(n):
    if n <= 0:
        return 1.0
    res = 1.0
    for i in range(n, 0, -2):
        res *= i
    return res

_I = np.zeros(14)
for k in range(14):
    if k == 0:
        _I[k] = np.sqrt(np.pi)
    else:
        _I[k] = _I[k-1] * (2*k - 1) / 2.0

def _compute_delta_x_squared(a0, a4, a8):
    """
    Compute (Δx)² for g(x) = P(x)exp(-x²/2) where
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with g(0)=0.
    """
    if time.time() > _DEADLINE:
        return None

    # Constraint g(0) = 0 => P(0) = 0
    # P(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    coeffs = np.array([a0, a4, a8, a12])
    
    # Construct p_k coefficients for P(x)
    # p is length 13, indices 0, 2, 4, ..., 12
    p = np.sum(coeffs[:, np.newaxis] * _H_COEFFS, axis=0)
    
    # Compute q_m = sum p_i p_j for i+j=m
    # We only need even m. q[2k] = sum_{i=0..k} p[2i] p[2(k-i)]
    # p_even = p[0::2] has length 7 (indices 0..6 corresponding to powers 0..12)
    p_even = p[0::2]
    
    q_even = np.convolve(p_even, p_even)
    
    # Norm = sum q[2k] * I[k]
    # q_even[k] corresponds to x^(2k)
    # I[k] corresponds to int x^(2k) exp(-x^2)
    norm = np.dot(q_even, _I[:len(q_even)])
    
    # X2Norm = sum q[2k] * int x^(2k+2) exp(-x^2) = sum q[2k] * I[k+1]
    x2_norm = np.dot(q_even, _I[1:len(q_even)+1])
    
    if norm <= 0 or not np.isfinite(norm):
        return None
        
    delta_x_sq = x2_norm / norm
    if not np.isfinite(delta_x_sq) or delta_x_sq < 0:
        return None
        
    return delta_x_sq

def _objective(x):
    """Objective function for optimization."""
    if time.time() > _DEADLINE - 5:
        return np.inf
    v = _compute_delta_x_squared(x[0], x[1], x[2])
    if v is None:
        return np.inf
    return v

def construct():
    """Find optimal coefficients to minimize the uncertainty product bound."""
    start_time = time.time()

    # Always evaluate seed first
    seed_val = _compute_delta_x_squared(*_SEED)
    if seed_val is None:
        seed_val = np.inf
    best_val = seed_val
    best_coeffs = _SEED.copy()

    def time_ok(margin=10):
        return (time.time() - start_time) < (200 - margin)

    # Strategy 1: Differential Evolution for global search
    # Bounds: based on seed and reasonable ranges
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    if time_ok(50):
        try:
            res = differential_evolution(_objective, bounds, seed=42, maxiter=1000, popsize=15, tol=1e-12, polish=True)
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Strategy 2: L-BFGS-B refinement
    if time_ok(20):
        try:
            result = minimize(_objective, best_coeffs, method='L-BFGS-B',
                              options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 3: Multi-start local refinement
    if time_ok(30):
        rng = np.random.default_rng(42)
        # Try some structured starts
        starts = [
            _SEED,
            np.array([0, 0, 0]),
            np.array([-1, 0.1, 0]),
            np.array([-0.5, 0.05, 0.001]),
            np.array([1, -0.1, 0]),
            np.array([-2, 0.2, -0.01]),
        ]
        for i in range(10):
            if not time_ok(10):
                break
            if i < len(starts):
                coeffs = starts[i]
            else:
                coeffs = best_coeffs + rng.normal(0, 0.5, 3)
            
            val = _compute_delta_x_squared(*coeffs)
            if val is not None and val < best_val:
                best_val = val
                best_coeffs = coeffs.copy()
            
            # Local polish
            if time_ok(5):
                try:
                    res = minimize(_objective, best_coeffs, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                            'adaptive': True})
                    if np.isfinite(res.fun) and res.fun < best_val:
                        best_val = res.fun
                        best_coeffs = res.x.copy()
                except Exception:
                    pass

    # Ensure we never return worse than seed
    if best_val > seed_val:
        best_coeffs = _SEED.copy()
        best_val = seed_val

    return best_coeffs
# EVOLVE-BLOCK-END
