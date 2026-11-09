import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _entropy_grad_P(q):
    # Returns gradient of entropy w.r.t probabilities q
    # H = -sum(q * log2(q))
    # dH/dq = -log2(q) - 1/ln(2)
    g = np.zeros_like(q)
    m = q > 1e-300
    g[m] = -(np.log2(q[m]) + 1.0 / np.log(2.0))
    return g

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, beta):
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1

    def fun_and_grad(z):
        P = _softmax(z)
        qa = np.bincount(idx_a, weights=P, minlength=la)
        qb = np.bincount(idx_b, weights=P, minlength=lb)
        qc = np.bincount(idx_c, weights=P, minlength=lc)
        qd = np.bincount(idx_d, weights=P, minlength=ld)
        
        # Entropies
        Ha = _entropy(qa)
        Hb = _entropy(qb)
        Hc = _entropy(qc)
        Hd = _entropy(qd)
        
        # Smooth Max
        Hs = np.array([Ha, Hb, Hd])
        max_H = np.max(Hs)
        exps = np.exp(beta * (Hs - max_H))
        sum_exps = np.sum(exps)
        M_smooth = (max_H + np.log(sum_exps)) / beta
        
        if M_smooth < 1e-12:
            return -1e9, np.zeros_like(z)
            
        value = Hc / M_smooth
        
        # Gradients w.r.t P
        ga = _entropy_grad_P(qa)
        gb = _entropy_grad_P(qb)
        gc = _entropy_grad_P(qc)
        gd = _entropy_grad_P(qd)
        
        # Map gradients from q-space to P-space
        # grad_P H_k = P * (g_k[idx] - dot(g_k[idx], P))
        # Note: g_k is defined on the bins. We need to expand it to P size.
        # g_k_expanded[i] = g_k[idx_k[i]]
        
        ga_exp = ga[idx_a]
        gb_exp = gb[idx_b]
        gc_exp = gc[idx_c]
        gd_exp = gd[idx_d]
        
        grad_Ha = P * (ga_exp - np.dot(ga_exp, P))
        grad_Hb = P * (gb_exp - np.dot(gb_exp, P))
        grad_Hc = P * (gc_exp - np.dot(gc_exp, P))
        grad_Hd = P * (gd_exp - np.dot(gd_exp, P))
        
        # Gradient of M_smooth w.r.t Hs
        # dM/dH_j = exp(beta * (H_j - max_H)) / sum_exps
        w_a = exps[0] / sum_exps
        w_b = exps[1] / sum_exps
        w_d = exps[2] / sum_exps
        
        grad_M = w_a * grad_Ha + w_b * grad_Hb + w_d * grad_Hd
        
        # Gradient of value = Hc / M_smooth
        # d(value)/dz = (1/M) * grad_Hc - (Hc/M^2) * grad_M
        
        grad_val = (1.0 / M_smooth) * grad_Hc - (Hc / (M_smooth * M_smooth)) * grad_M
        
        return -value, -grad_val

    return fun_and_grad

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_val = -1e9
    best_z = np.zeros(100)
    
    # Initializations
    inits = []
    # Uniform
    inits.append(np.zeros(100))
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.normal(0, 2.0, 100))
        
    # Heuristic: Diagonal x-y=0
    z_diag = np.full(100, -100.0)
    for y in range(10):
        x = y
        z_diag[y*10 + x] = 0.0
    inits.append(z_diag)
    
    # Heuristic: Anti-diagonal x+y=9
    z_anti = np.full(100, -100.0)
    for y in range(10):
        x = 9 - y
        z_anti[y*10 + x] = 0.0
    inits.append(z_anti)

    # Annealing schedule for beta
    # Start smooth, end sharp
    beta_schedule = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    for z0 in inits:
        if time.time() > deadline - 30.0:
            break
            
        current_z = z0.copy()
        
        for beta in beta_schedule:
            if time.time() > deadline - 10.0:
                break
                
            fun_and_grad = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, beta)
            
            try:
                res = minimize(fun_and_grad, current_z, jac=True, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000})
                current_z = res.x
                # Evaluate with hard max for true score
                P = _softmax(current_z).reshape(10, 10)
                val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
                if val > best_val:
                    best_val = val
                    best_z = current_z.copy()
            except Exception:
                pass
        
        # Final polish with very high beta
        if time.time() < deadline - 10.0:
            fun_and_grad = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, 1000.0)
            try:
                res = minimize(fun_and_grad, current_z, jac=True, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000})
                current_z = res.x
                P = _softmax(current_z).reshape(10, 10)
                val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
                if val > best_val:
                    best_val = val
                    best_z = current_z.copy()
            except Exception:
                pass

    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P
