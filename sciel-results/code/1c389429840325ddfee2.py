"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START_TIME = time.time()
DEADLINE = 210.0
SAFETY_MARGIN = 15.0

def _check_deadline():
    return (time.time() - _START_TIME) < (DEADLINE - SAFETY_MARGIN)

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
# Using recurrence: I_0 = sqrt(pi), I_n = (2n-1)/2 * I_{n-1}
_I_max = 25  # We need up to index 24 (x^24) for m4 of degree 12 poly
_I = np.zeros(_I_max + 1)
_I[0] = np.sqrt(np.pi)
for n in range(1, _I_max + 1):
    _I[n] = (2.0 * n - 1.0) / 2.0 * _I[n - 1]

def _get_poly_coeffs(a0, a4, a8):
    """
    Return coefficients p[0..6] for P(x) = sum_{k=0}^6 p[k] x^{2k}
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    p = np.zeros(7)
    # p0 (x^0)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    # p1 (x^2)
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    # p2 (x^4)
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    # p3 (x^6)
    p[3] = -28.0 * a8 - 6930.0 * a12
    # p4 (x^8)
    p[4] = a8 + 1155.0 * a12
    # p5 (x^10)
    p[5] = -66.0 * a12
    # p6 (x^12)
    p[6] = a12
    
    return p

def _compute_moments(p):
    """
    Compute moments m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx for k=0,1,2
    and n_k for the Fourier side.
    """
    m = np.zeros(3)
    # Vectorized computation: m_k = sum_{i,j} p_i p_j I[k+i+j]
    for k in range(3):
        s = 0.0
        for i in range(7):
            pi = p[i]
            if pi == 0.0: continue
            for j in range(7):
                pj = p[j]
                if pj == 0.0: continue
                idx = k + i + j
                if idx <= _I_max:
                    s += pi * pj * _I[idx]
        m[k] = s
        
    # Fourier side: Q(x) = P(-ix) = sum p_k (-1)^k x^{2k}
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
        
    n = np.zeros(3)
    # Note: The norm of Fourier transform depends on convention.
    # Standard: F(f)(xi) = int f(x) e^{-ixi} dx.
    # f = P e^{-x^2/2}. F(f) = sqrt(2pi) P(-i xi) e^{-xi^2/2}.
    # |F(f)|^2 = 2pi |P(-i xi)|^2 e^{-xi^2} = 2pi |Q(xi)|^2 e^{-xi^2}.
    # So n_k = 2pi * integral xi^{2k} |Q|^2 e^{-xi^2} dxi.
    for k in range(3):
        s = 0.0
        for i in range(7):
            qi = q[i]
            if qi == 0.0: continue
            for j in range(7):
                qj = q[j]
                if qj == 0.0: continue
                idx = k + i + j
                if idx <= _I_max:
                    s += qi * qj * _I[idx]
        n[k] = 2.0 * np.pi * s
        
    return m, n

def _objective(params):
    """
    Compute the uncertainty product bound.
    Returns (value, gradient) if jac is requested, else value.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    p = _get_poly_coeffs(a0, a4, a8)
    m, n = _compute_moments(p)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # da12/da = [-1/945, -3/945, -105/945]
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    
    for c in range(3):
        # dp/da_c
        dp = np.zeros(7)
        # Direct derivatives of p_k wrt a_c (c=0,1,2 correspond to a0, a4, a8)
        # p0 = a0 + 3a4 + 105a8 + 945a12
        # p1 = -6a4 - 420a8 - 7560a12
        # p2 = a4 + 210a8 + 13860a12
        # p3 = -28a8 - 6930a12
        # p4 = a8 + 1155a12
        # p5 = -66a12
        # p6 = a12
        
        # Coefficients for a0, a4, a8 directly
        direct = np.array([
            [1.0, 0.0, 0.0],  # p0
            [0.0, -6.0, 0.0], # p1
            [0.0, 1.0, 0.0],  # p2
            [0.0, 0.0, -28.0],# p3
            [0.0, 0.0, 1.0],  # p4
            [0.0, 0.0, 0.0],  # p5
            [0.0, 0.0, 0.0]   # p6
        ])
        
        # Coefficients for a12
        a12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
        
        dp = direct[:, c] + a12_coeff * da12[c]
        
        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dpi = dp[i]
                if dpi == 0.0: continue
                inner = 0.0
                for j in range(7):
                    pj = p[j]
                    if pj == 0.0: continue
                    idx = k + i + j
                    if idx <= _I_max:
                        inner += pj * _I[idx]
                s += dpi * inner
            dm[k] = 2.0 * s
            
        # dq/da_c
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0) ** k)
            
        dn = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dqi = dq[i]
                if dqi == 0.0: continue
                inner = 0.0
                for j in range(7):
                    qj = q[j]
                    if qj == 0.0: continue
                    idx = k + i + j
                    if idx <= _I_max:
                        inner += qj * _I[idx]
                s += dqi * inner
            dn[k] = 2.0 * 2.0 * np.pi * s
            
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def _objective_val(params):
    val, _ = _objective(params)
    return val

def construct():
    if not _check_deadline():
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = None
    best_score = np.inf
    
    # Seed point
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # 1. Local optimization from seed with analytic gradient
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
    
    # 2. Global search with Differential Evolution
    # Use a wider range to explore, but keep it reasonable
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    try:
        de_result = differential_evolution(_objective_val, bounds, 
                                           maxiter=300, popsize=30,
                                           seed=42, tol=1e-12,
                                           polish=True, 
                                           init='sobol')
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
        
    # 3. Multi-start local refinement with structured starts
    rng = np.random.default_rng(42)
    base = best_result if best_result is not None else x0
    
    # Structured starts: perturbations at multiple scales
    starts = [x0, base.copy()]
    
    # Radial perturbations at multiple scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for _ in range(5):
            direction = rng.normal(0, 1, 3)
            direction /= (np.linalg.norm(direction) + 1e-15)
            starts.append(base + scale * direction)
    
    # Axis-aligned perturbations
    for dim in range(3):
        for delta in [-0.1, -0.05, 0.05, 0.1]:
            pert = base.copy()
            pert[dim] += delta
            starts.append(pert)
    
    # Scaled seed points
    for scale in [0.5, 0.75, 1.25, 1.5, 2.0]:
        starts.append(x0 * scale)
    
    # Additional diverse starts
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-1.0, 0.1, 0.001]))
    starts.append(np.array([-2.0, 0.05, 0.0005]))
    starts.append(np.array([-0.5, 0.02, 0.0002]))
    starts.append(np.array([-1.5, 0.08, 0.0008]))
    starts.append(np.array([1.0, -0.05, 0.0005]))
    starts.append(np.array([-3.0, 0.15, 0.002]))
    
    for _ in range(10):
        starts.append(rng.normal(0, 0.5, 3))
    
    for start in starts:
        if not _check_deadline():
            break
        try:
            res = minimize(_objective, start, method='L-BFGS-B', jac=True,
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # 4. Final high-precision polish with Nelder-Mead (derivative-free)
    if best_result is not None and _check_deadline():
        try:
            nm_result = minimize(_objective_val, best_result, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14,
                                        'adaptive': True})
            if nm_result.fun < best_score and np.isfinite(nm_result.fun):
                best_score = nm_result.fun
                best_result = nm_result.x
        except Exception:
            pass
        
        # 5. Another L-BFGS-B polish from the Nelder-Mead result
        if _check_deadline():
            try:
                final_result = minimize(_objective, best_result, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-16})
                if final_result.fun < best_score and np.isfinite(final_result.fun):
                    best_score = final_result.fun
                    best_result = final_result.x
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    return x0

# EVOLVE-BLOCK-END
