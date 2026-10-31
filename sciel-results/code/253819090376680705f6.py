"""Uncertainty bound minimization via exact moment computation with analytic gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# Precompute I[n] = integral x^{2n} e^{-x^2} dx for n = 0..24
# I[n] = (2n-1)!! / 2^n * sqrt(pi) = (2n)! / (4^n * n!) * sqrt(pi)
_I = np.zeros(25)
_I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    _I[n] = _I[n-1] * (2*n - 1) / 2.0

def _bound_and_grad(params):
    """
    Compute uncertainty product sqrt(vf * vF) and its analytic gradient.
    g(x) = P(x) * exp(-x^2/2) where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    a12 = -(a0 + 3*a4 + 105*a8) / 945  (enforces g(0)=0)
    
    Position variance: vf = m2/m0 where m_k = integral x^k |P|^2 e^{-x^2} dx
    Frequency variance: vF = n2/n0 where n_k = integral xi^k |Q|^2 e^{-xi^2} dx * 2*pi
    Q(xi) = sum p_k (-1)^k xi^{2k}  (from P(-i*xi))
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12  -> in terms of x^{2k}: p0=12, p1=-48, p2=16 (times a4)
    # Wait, let me use the standard Hermite polynomials:
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 2520
    # H_12(x) = 4096x^12 - 92160x^10 + 645120x^8 - 1445760x^6 + 1290240x^4 - 399168x^2 + 27720
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.zeros(7)
    # H0 contributes: 1 to x^0
    p[0] += a0
    # H4 contributes: 12 to x^0, -48 to x^2, 16 to x^4
    p[0] += 12.0 * a4
    p[1] += -48.0 * a4
    p[2] += 16.0 * a4
    # H8 contributes: 2520 to x^0, -13440 to x^2, 11520 to x^4, -3072 to x^6, 256 to x^8
    p[0] += 2520.0 * a8
    p[1] += -13440.0 * a8
    p[2] += 11520.0 * a8
    p[3] += -3072.0 * a8
    p[4] += 256.0 * a8
    # H12 contributes: 27720 to x^0, -399168 to x^2, 1290240 to x^4, -1445760 to x^6, 645120 to x^8, -92160 to x^10, 4096 to x^12
    p[0] += 27720.0 * a12
    p[1] += -399168.0 * a12
    p[2] += 1290240.0 * a12
    p[3] += -1445760.0 * a12
    p[4] += 645120.0 * a12
    p[5] += -92160.0 * a12
    p[6] += 4096.0 * a12
    
    # Q(xi) coefficients: q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    # Compute moments m_k = integral x^{2k} |P|^2 e^{-x^2} dx for k=0,1,2
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    m[k] += pp * _I[idx]
    
    # Compute moments n_k = 2*pi * integral xi^{2k} |Q|^2 e^{-xi^2} dxi for k=0,1,2
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    n[k] += 2.0 * np.pi * qq * _I[idx]
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Analytic gradient
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c:
    # dp0/da_c = [1, 12, 2520][c] + 27720*da12[c]
    # dp1/da_c = [0, -48, -13440][c] + (-399168)*da12[c]
    # dp2/da_c = [0, 16, 11520][c] + 1290240*da12[c]
    # dp3/da_c = [0, 0, -3072][c] + (-1445760)*da12[c]
    # dp4/da_c = [0, 0, 256][c] + 645120*da12[c]
    # dp5/da_c = [0, 0, 0][c] + (-92160)*da12[c]
    # dp6/da_c = [0, 0, 0][c] + 4096*da12[c]
    
    dp_da = np.zeros((3, 7))
    dp_da[0, 0] = 1.0 + 27720.0 * da12[0]
    dp_da[0, 1] = 0.0 + (-399168.0) * da12[0]
    dp_da[0, 2] = 0.0 + 1290240.0 * da12[0]
    dp_da[0, 3] = 0.0 + (-1445760.0) * da12[0]
    dp_da[0, 4] = 0.0 + 645120.0 * da12[0]
    dp_da[0, 5] = 0.0 + (-92160.0) * da12[0]
    dp_da[0, 6] = 0.0 + 4096.0 * da12[0]
    
    dp_da[1, 0] = 12.0 + 27720.0 * da12[1]
    dp_da[1, 1] = -48.0 + (-399168.0) * da12[1]
    dp_da[1, 2] = 16.0 + 1290240.0 * da12[1]
    dp_da[1, 3] = 0.0 + (-1445760.0) * da12[1]
    dp_da[1, 4] = 0.0 + 645120.0 * da12[1]
    dp_da[1, 5] = 0.0 + (-92160.0) * da12[1]
    dp_da[1, 6] = 0.0 + 4096.0 * da12[1]
    
    dp_da[2, 0] = 2520.0 + 27720.0 * da12[2]
    dp_da[2, 1] = -13440.0 + (-399168.0) * da12[2]
    dp_da[2, 2] = 11520.0 + 1290240.0 * da12[2]
    dp_da[2, 3] = -3072.0 + (-1445760.0) * da12[2]
    dp_da[2, 4] = 256.0 + 645120.0 * da12[2]
    dp_da[2, 5] = 0.0 + (-92160.0) * da12[2]
    dp_da[2, 6] = 0.0 + 4096.0 * da12[2]
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dp_da[c]
        
        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if abs(dpi) < 1e-30:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 25:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0) ** k)
        
        # dn_k/da_c = 2*2*pi * sum_i dq_i * sum_j q_j I[k+i+j]
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if abs(dqi) < 1e-30:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 25:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * 2.0 * np.pi * dqi * s
        
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    start_time = time.time()
    
    best_val = np.inf
    best_params = None
    
    # Diverse starting points
    starts = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([-0.5, 0.0, 0.0]),
        np.array([-1.5, 0.0, 0.0]),
        np.array([-0.8, 0.0, 0.0]),
        np.array([-1.2, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([-1.0, -0.1, 0.0]),
        np.array([-1.0, 0.0, 0.01]),
        np.array([-1.0, 0.0, -0.01]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-1.5, 0.08, 0.0008]),
    ]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    for start in starts:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 25:
            break
        
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() - start_time < _DEADLINE - 15:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B from Nelder-Mead result
    if best_params is not None and time.time() - start_time < _DEADLINE - 5:
        try:
            final_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                               jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
            if final_res.fun < best_val and np.isfinite(final_res.fun):
                best_val = final_res.fun
                best_params = final_res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.0, 0.0, 0.0])
    
    return best_params
# EVOLVE-BLOCK-END
