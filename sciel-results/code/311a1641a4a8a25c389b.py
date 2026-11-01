"""Analytic uncertainty bound with exact gradient + multi-method optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, least_squares, fsolve

_T0 = time.time()
_DEADLINE = _T0 + 195  # 15s safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
_I_TABLE = np.zeros(14)
def _precompute_I():
    sp = math.sqrt(math.pi)
    I = np.zeros(14)
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        num = 1.0
        for m in range(1, 2*n + 1):
            num *= m
        den = (4.0 ** n)
        for m in range(1, n + 1):
            den *= m
        I[n] = num / den * sp
    return I
_I_TABLE = _precompute_I()

# Precompute the coefficient matrix for p in terms of (a0, a4, a8, a12)
# p0 = 1*a0 + 3*a4 + 105*a8 + 945*a12
# p1 = 0*a0 + (-6)*a4 + (-420)*a8 + (-7560)*a12
# p2 = 0*a0 + 1*a4 + 210*a8 + 13860*a12
# p3 = 0*a0 + 0*a4 + (-28)*a8 + (-6930)*a12
# p4 = 0*a0 + 0*a4 + 1*a8 + 1155*a12
# p5 = 0*a0 + 0*a4 + 0*a8 + (-66)*a12
# p6 = 0*a0 + 0*a4 + 0*a8 + 1*a12
_P_COEFFS = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0],
])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
_A12_COEFFS = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Precompute M matrices for efficient moment computation
# m_k = sum_{i,j} p_i p_j I[k+i+j]
# We can write this as: for each k, m_k = p^T M_k p where (M_k)_{ij} = I[k+i+j]
_M_MATRICES = []
for k in range(3):
    Mk = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                Mk[i, j] = _I_TABLE[idx]
    _M_MATRICES.append(Mk)

# Q coefficients: q_k = p_k * (-1)^k
_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _A12_COEFFS[0]*a0 + _A12_COEFFS[1]*a4 + _A12_COEFFS[2]*a8
    
    # Compute p vector
    p = _P_COEFFS[0]*a0 + _P_COEFFS[1]*a4 + _P_COEFFS[2]*a8 + _P_COEFFS[3]*a12
    
    # Compute q vector (Fourier domain)
    q = p * _SIGNS
    
    # Compute moments using precomputed matrices
    m0 = p @ _M_MATRICES[0] @ p
    m2 = p @ _M_MATRICES[1] @ p
    n0 = 2.0*math.pi * (q @ _M_MATRICES[0] @ q)
    n2 = 2.0*math.pi * (q @ _M_MATRICES[1] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # dp/da_c for c = 0,1,2 (a0, a4, a8)
    # dp/da_c = P_COEFFS[:, c] + P_COEFFS[:, 3] * A12_COEFFS[c]
    DP = np.zeros((7, 3))
    for c in range(3):
        DP[:, c] = _P_COEFFS[:, c] + _P_COEFFS[:, 3] * _A12_COEFFS[c]
    
    grad = np.zeros(3)
    for c in range(3):
        dp = DP[:, c]
        dq = dp * _SIGNS
        
        # dm_k/da_c = 2 * dp^T M_k p
        dm0 = 2.0 * (dp @ _M_MATRICES[0] @ p)
        dm2 = 2.0 * (dp @ _M_MATRICES[1] @ p)
        
        # dn_k/da_c = 2 * 2*pi * dq^T M_k q
        dn0 = 2.0 * 2.0*math.pi * (dq @ _M_MATRICES[0] @ q)
        dn2 = 2.0 * 2.0*math.pi * (dq @ _M_MATRICES[1] @ q)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _lsq_residuals(params):
    """
    Residuals for least_squares formulation.
    params = [a0, a4, a8, t] where t is the target uncertainty.
    We want to minimize ||r||^2 where r = [bound(a) - t, a0, a4, a8]
    This effectively minimizes (bound - t)^2 while keeping a free.
    But that's not quite right. Let's think differently.
    
    Actually, the standard trick is: minimize (bound(a) - t)^2 + penalty
    But we want bound to be small. So we set t=0 and minimize bound(a)^2.
    That's just the original objective.
    
    Better approach: use the fact that at optimum, gradient of bound = 0.
    So we solve the system: grad_bound(a) = 0 (3 equations, 3 unknowns).
    This is a nonlinear system we can solve with fsolve.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    _, grad = _bound_and_grad(np.array([a0, a4, a8]))
    return grad


def _lsq_residuals_jac(params):
    """
    Jacobian of _lsq_residuals = Hessian of bound.
    Compute numerically for simplicity (3x3 matrix).
    """
    a0, a4, a8 = params[0], params[1], params[2]
    p = np.array([a0, a4, a8])
    eps = 1e-7
    J = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pp = p.copy()
            pp[j] += eps
            _, grad_plus = _bound_and_grad(pp)
            pp = p.copy()
            pp[j] -= eps
            _, grad_minus = _bound_and_grad(pp)
            J[i, j] = (grad_plus[i] - grad_minus[i]) / (2 * eps)
    return J


def construct():
    # Multiple starting points - expanded set
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.0, 0.06, 0.0005]),
        np.array([-0.6, 0.025, 0.00025]),
        np.array([-1.8, 0.07, 0.0007]),
        np.array([-2.5, 0.12, 0.0012]),
        np.array([0.5, -0.03, 0.0003]),
        np.array([0.3, -0.02, 0.0002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: SLSQP from best and nearby points
    if best_params is not None and time.time() < _DEADLINE - 40:
        slsqp_starts = [best_params.copy()]
        rng2 = np.random.default_rng(456)
        for i in range(10):
            slsqp_starts.append(best_params + rng2.normal(0, 0.1, 3))
        
        for start in slsqp_starts:
            if time.time() > _DEADLINE - 30:
                break
            try:
                res = minimize(_objective_grad, start, method='SLSQP',
                              jac=True, options={'maxiter': 2000, 'ftol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: least_squares to solve grad=0 (KKT conditions)
    if best_params is not None and time.time() < _DEADLINE - 25:
        try:
            # Solve the nonlinear system grad_bound(a) = 0
            sol, info, ier, msg = fsolve(_lsq_residuals, best_params, full_output=True,
                                          xtol=1e-15, maxfev=1000)
            if ier == 1:  # Solution found
                val, _ = _bound_and_grad(sol)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = sol.copy()
        except Exception:
            pass
    
    # Phase 4: least_squares with LM from multiple nearby starts
    if best_params is not None and time.time() < _DEADLINE - 15:
        rng3 = np.random.default_rng(789)
        for i in range(5):
            if time.time() > _DEADLINE - 10:
                break
            try:
                start = best_params + rng3.normal(0, 0.05, 3)
                res = least_squares(_lsq_residuals, start, method='lm', 
                                     xtol=1e-15, gtol=1e-15, ftol=1e-15, max_nfev=1000)
                if res.success:
                    val, _ = _bound_and_grad(res.x)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 5: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final L-BFGS-B polish with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 7: fsolve polish on KKT conditions from refined point
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            sol, info, ier, msg = fsolve(_lsq_residuals, best_params, full_output=True,
                                          xtol=1e-16, maxfev=500)
            if ier == 1:
                val, _ = _bound_and_grad(sol)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = sol.copy()
        except Exception:
            pass
    
    # Phase 8: Very tight final L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 1:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
