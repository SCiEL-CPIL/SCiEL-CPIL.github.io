import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx
    I = np.zeros(14)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 14):
        I[n] = (2*n - 1) / 2.0 * I[n-1]
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            if pp == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # Fourier transform part
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
        
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            if qq == 0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0*da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0*da12[c]
        
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0)**k)
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * I[idx]
                dn[k] += 2.0 * 2.0*np.pi * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    # 1. Grid search on the unit sphere
    # a0 = sin(theta) * cos(phi)
    # a4 = sin(theta) * sin(phi)
    # a8 = cos(theta)
    
    best_val = np.inf
    best_params = np.array([0.0, 0.0, 0.0])
    
    # Use a moderately fine grid
    n_theta = 100
    n_phi = 200
    
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    # We can vectorize the evaluation partially, but the function is scalar.
    # 100 * 200 = 20000 evaluations.
    # Each evaluation is fast.
    
    # To be safe with time, we do a coarser grid first, then refine.
    # But 20k evaluations of a simple analytic function should be < 1s.
    
    candidates = []
    
    for th in thetas:
        for ph in phis:
            a0 = np.sin(th) * np.cos(ph)
            a4 = np.sin(th) * np.sin(ph)
            a8 = np.cos(th)
            params = np.array([a0, a4, a8])
            try:
                val, _ = _bound_and_grad(params)
                if val < best_val:
                    best_val = val
                    best_params = params.copy()
                # Keep top candidates for polishing
                candidates.append((val, params))
            except:
                pass
            
            if time.time() > _DEADLINE - 60:
                # If grid search takes too long, break and polish what we have
                break
        if time.time() > _DEADLINE - 60:
            break
            
    # Sort candidates by value
    candidates.sort(key=lambda x: x[0])
    
    # Take top 20 for polishing
    top_candidates = [c[1] for c in candidates[:20]]
    
    # 2. Polish with L-BFGS-B
    for start in top_candidates:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # 3. Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    return best_params
