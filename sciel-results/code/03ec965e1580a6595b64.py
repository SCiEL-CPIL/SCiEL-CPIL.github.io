"""
Optimized uncertainty bound using vectorized moment computation and multi-start optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Safety margin: evaluator kills at 210s. We aim to finish by 195s.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0

def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    Represented as vector p = [p0, p1, p2, p3, p4, p5, p6] for x^0, x^2, ..., x^12.
    """
    H = np.zeros((4, 7))
    # H_0 = 1
    H[0, 0] = 1.0
    # H_4 = x^4 - 6x^2 + 3
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    # H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    # H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H

def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..24.
    I[n] = (2n-1)!! * sqrt(pi) / 2^n
    """
    I = np.zeros(25)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 25):
        I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I

_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute matrix for moments: M[i,j] = I[i+j]
_M = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M[i, j] = _I[i + j]

# Precompute matrix for second moments: M2[i,j] = I[1+i+j]
_M2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M2[i, j] = _I[1 + i + j]

def _calc_bound_and_grad(params):
    """
    Compute the uncertainty bound A(f)*A(F) and its gradient.
    params = [a0, a4, a8]
    a12 is determined by P(0)=0 constraint: a0 + 3a4 + 105a8 + 945a12 = 0
    """
    a0, a4, a8 = params[0], params[1], params[2]
    # Constraint: sum c_i H_i(0) = 0 => a0*1 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    # Moments for position space: m0 = p^T M p, m2 = p^T M2 p
    m0 = p @ _M @ p
    m2 = p @ _M2 @ p
    
    # Moments for frequency space
    q = p * _SIGN
    
    n0 = 2.0 * np.pi * (q @ _M @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient calculation
    # d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2 * m0 - m2 * dm0) / m0^2
    # dm0 = 2 * p^T M dp, dm2 = 2 * p^T M2 dp
    
    # Derivatives of a12 w.r.t a0, a4, a8
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        # dp/da_c = H[c] + da12[c]*H[3]
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        # dm0 = 2 * p^T M dp
        # dm2 = 2 * p^T M2 dp
        dm0 = 2.0 * p @ _M @ dp
        dm2 = 2.0 * p @ _M2 @ dp
        
        dq = dp * _SIGN
        
        # dn0 = 4*pi * q^T M dq
        # dn2 = 4*pi * q^T M2 dq
        dn0 = 4.0 * np.pi * (q @ _M @ dq)
        dn2 = 4.0 * np.pi * (q @ _M2 @ dq)
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _calc_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _calc_bound_and_grad(params)
    return val, grad

def construct():
    """
    Multi-start L-BFGS-B optimization with vectorized moment computation.
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = [best_params.copy()]
    
    # Perturbed starts around the seed
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        for _ in range(5):
            pert = rng.randn(3) * scale * np.array([1.0, 0.1, 0.01])
            starts.append(best_params + pert)
    
    # Random starts in a reasonable box
    for _ in range(20):
        x0 = np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.02, 0.02)
        ])
        starts.append(x0)
    
    # Multi-start L-BFGS-B
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(
                _objective_grad, 
                start, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 2:
        try:
            res_final = minimize(
                _objective_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res_final.fun < best_val and np.isfinite(res_final.fun):
                best_val = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
