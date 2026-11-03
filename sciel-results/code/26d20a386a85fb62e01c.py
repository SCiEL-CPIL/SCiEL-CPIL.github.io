import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    def objective_de(x):
        # x is unnormalized h
        h = np.abs(x)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h_norm = h / s
        # We want to maximize score, so minimize -score
        # But DE minimizes.
        # Score = L2^2 / (L1 * DELTA * MaxConv)
        # With L1=1, Score = L2^2 / (DELTA * MaxConv)
        # To maximize Score, we minimize MaxConv / L2^2 ? No.
        # We minimize MaxConv.
        conv = np.convolve(h_norm, h_norm, mode='full')
        return np.max(conv)
        
    # Run DE
    bounds = [(0, 1)] * N
    try:
        de_res = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-5,
            update_interval=10,
            polish=True
        )
        if de_res.success:
            h_de = np.abs(de_res.x)
            s = np.sum(h_de)
            if s > 1e-15:
                h_de /= s
                s_de = _score_raw(h_de)
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de.copy()
    except Exception:
        pass

    # SLSQP Refinement
    num_conv = 2 * N - 1
    def objective(x): return x[N]
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
    def constraint_eq_fun(x): return np.sum(x[:N]) - 1.0
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
    constraints = [
        {'type': 'eq', 'fun': constraint_eq_fun, 'jac': constraint_eq_jac},
        {'type': 'ineq', 'fun': constraint_ineq_fun, 'jac': constraint_ineq_jac}
    ]
    bounds_slsqp = [(0.0, 1.0)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter=300):
        if np.sum(h_init) < 1e-15: return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds_slsqp, constraints=constraints, jac=objective_grad, options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success:
                h_c = np.clip(res.x[:N], 0, None)
                if np.sum(h_c) > 1e-15:
                    h_c /= np.sum(h_c)
                    return h_c
        except: pass
        return None

    h_ref = run_epigraph(best_h)
    if h_ref is not None:
        s_ref = _score_raw(h_ref)
        if s_ref > best_score:
            best_score = s_ref
            best_h = h_ref.copy()

    return list(best_h)
