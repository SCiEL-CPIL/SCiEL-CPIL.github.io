"""Uncertainty inequality optimization: analytic Hermite-basis quadratic forms + multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces P(0)=0)
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # should be ~0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _build_moment_matrices():
    """
    Build the quadratic-form matrices for m_k and n_k.
    m_k = sum_{i,j} p_i p_j * I[k+i+j]
    n_k = 2*pi * sum_{i,j} q_i q_j * I[k+i+j]  where q_i = (-1)^i p_i
    
    We express m_k and n_k as quadratic forms in p:
    m_k = p^T M_k p  where (M_k)_{ij} = I[k+i+j]
    n_k = 2*pi * p^T D M_k D p  where D = diag((-1)^i)
    """
    D = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 25:
                    M[k, i, j] = _I[idx]
    # n matrices: D M_k D
    N = np.zeros((3, 7, 7))
    for k in range(3):
        N[k] = M[k] * np.outer(D, D)
    return M, N


_M, _N = _build_moment_matrices()


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product sqrt(vf * vF)."""
    p = _coeffs_from_params(a0, a4, a8)
    
    m0 = p @ _M[0] @ p
    m2 = p @ _M[2] @ p
    n0 = 2.0 * np.pi * (p @ _N[0] @ p)
    n2 = 2.0 * np.pi * (p @ _N[2] @ p)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18
    return bound


def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the bound and its analytic gradient via chain rule.
    
    p = A @ [a0, a4, a8] + b  (affine in parameters, since a12 is linear)
    bound = sqrt( (p^T M2 p / (p^T M0 p)) * (2pi p^T N2 p / (2pi p^T N0 p)) )
    
    Let vf = p^T M2 p / (p^T M0 p), vF = p^T N2 p / (p^T N0 p)
    bound = sqrt(vf * vF)
    
    d(bound)/da = (1/(2*bound)) * (vF * d(vf)/da + vf * d(vF)/da)
    d(vf)/da = (2 p^T M2 dp/da * m0 - 2 p^T M0 dp/da * m2) / m0^2
    d(vF)/da = (2 p^T N2 dp/da * n0 - 2 p^T N0 dp/da * n2) / n0^2
    """
    p = _coeffs_from_params(a0, a4, a8)
    
    m0 = p @ _M[0] @ p
    m2 = p @ _M[2] @ p
    n0 = 2.0 * np.pi * (p @ _N[0] @ p)
    n2 = 2.0 * np.pi * (p @ _N[2] @ p)
    
    if m0 <= 0 or n0 <= 0 or m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # dp/da: p is affine in (a0, a4, a8)
    # p[0] = 0 (always, by constraint)
    # p[1] = -6*a4 - 420*a8 - 7560*a12, a12 = -(a0+3a4+105a8)/945
    #       = -6a4 - 420a8 + (7560/945)(a0+3a4+105a8)
    #       = (7560/945)a0 + (-6 + 24)a4 + (-420 + 840)a8
    #       = 8a0 + 18a4 + 420a8
    # p[2] = a4 + 210a8 + 13860*a12
    #       = a4 + 210a8 - (13860/945)(a0+3a4+105a8)
    #       = -(13860/945)a0 + (1 - 44)a4 + (210 - 1540)a8
    #       = -(13860/945)a0 - 43a4 - 1330a8
    # p[3] = -28a8 - 6930*a12
    #       = -28a8 + (6930/945)(a0+3a4+105a8)
    #       = (6930/945)a0 + (21)a4 + (-28+770)a8
    #       = (6930/945)a0 + 21a4 + 742a8
    # p[4] = a8 + 1155*a12
    #       = a8 - (1155/945)(a0+3a4+105a8)
    #       = -(1155/945)a0 - (3.5)a4 + (1 - 128.333...)a8
    #       = -(1155/945)a0 - (7/2)a4 - (1145/9)a8
    # p[5] = -66*a12 = (66/945)(a0+3a4+105a8)
    #       = (66/945)a0 + (198/945)a4 + (6930/945)a8
    # p[6] = a12 = -(1/945)(a0+3a4+105a8)
    #       = -(1/945)a0 - (3/945)a4 - (105/945)a8
    
    # Build dp/da matrix (7x3): rows are p[0..6], cols are da0, da4, da8
    dp_da = np.zeros((7, 3))
    dp_da[0] = [0.0, 0.0, 0.0]
    dp_da[1] = [7560.0/945.0, -6.0 + 24.0, -420.0 + 840.0]  # [8, 18, 420]
    dp_da[2] = [-13860.0/945.0, 1.0 - 44.0, 210.0 - 1540.0]  # [-14.667, -43, -1330]
    dp_da[3] = [6930.0/945.0, 3.0*6930.0/945.0, -28.0 + 770.0]  # [7.333, 22, 742]
    dp_da[4] = [-1155.0/945.0, -3.0*1155.0/945.0, 1.0 - 105.0*1155.0/945.0]  # [-1.222, -3.667, -127.333]
    dp_da[5] = [66.0/945.0, 3.0*66.0/945.0, 105.0*66.0/945.0]  # [0.0698, 0.2095, 7.333]
    dp_da[6] = [-1.0/945.0, -3.0/945.0, -105.0/945.0]  # [-0.001058, -0.003175, -0.1111]
    
    # Gradient computation
    # d(vf)/da_i = 2 * (p^T M2 dp_i * m0 - p^T M0 dp_i * m2) / m0^2
    # where dp_i = dp_da[:, i]
    
    grad = np.zeros(3)
    for i in range(3):
        dp_i = dp_da[:, i]
        M2_dp = _M[2] @ dp_i
        M0_dp = _M[0] @ dp_i
        N2_dp = _N[2] @ dp_i
        N0_dp = _N[0] @ dp_i
        
        d_vf = 2.0 * (p @ M2_dp * m0 - p @ M0_dp * m2) / (m0 * m0)
        d_vF = 2.0 * (p @ N2_dp * n0 - p @ N0_dp * n2) / (n0 * n0)
        
        grad[i] = (vF * d_vf + vf * d_vF) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    return _compute_bound(params[0], params[1], params[2])


def _objective_grad(params):
    bound, grad = _compute_bound_and_grad(params[0], params[1], params[2])
    return bound, grad


def construct():
    best_val = np.inf
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Initial evaluation
    try:
        v = _compute_bound(best_params[0], best_params[1], best_params[2])
        if np.isfinite(v) and v < best_val:
            best_val = v
    except Exception:
        pass
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start candidate generation
    starts = [best_params.copy()]
    
    # Local perturbations around seed
    for _ in range(30):
        s = best_params + rng.normal(0, 0.1, 3)
        starts.append(s)
    
    # Wider exploration
    for _ in range(80):
        s = np.array([
            rng.uniform(-3.5, 2.5),
            rng.uniform(-0.6, 0.6),
            rng.uniform(-0.015, 0.015)
        ])
        starts.append(s)
    
    # Structured grid
    for a0 in np.linspace(-3.0, 1.5, 10):
        for a4 in np.linspace(-0.4, 0.4, 6):
            a8 = 0.001 * rng.normal()
            starts.append(np.array([a0, a4, a8]))
    
    # Evaluate all starts, keep top ones
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            val = _compute_bound(s[0], s[1], s[2])
            if np.isfinite(val):
                scored.append((val, s))
                if val < best_val:
                    best_val = val
                    best_params = s.copy()
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Phase 2: L-BFGS-B from top starts
    top_n = min(20, len(scored))
    top_starts = [s for _, s in scored[:top_n]]
    if best_params not in top_starts:
        top_starts.append(best_params.copy())
    
    for s in top_starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead ultra-fine polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
