import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def _time_left(start, deadline):
    return deadline - time.time()

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.convolve(h, h)
    peak = np.max(c)
    if peak <= 0:
        return 0.0
    return l2_sq / (l1 * peak)

def _optimize_epigraph(h0, start, deadline, maxiter=500, ftol=1e-10):
    """
    Optimize the epigraph formulation:
    Minimize M subject to:
      sum(h) = 1
      h >= 0
      M >= c_k for all k (autoconvolution constraints)
    
    Where c = h * h (autoconvolution)
    """
    def autoconv(h):
        return np.convolve(h, h)
    
    def objective(x):
        return x[N]
    
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
    
    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0
    
    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad
    
    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = autoconv(h)
        return M - c
    
    def constraints_autoconv_grad(x):
        h = x[:N]
        grad = np.zeros((2 * N - 1, N + 1))
        for k in range(2 * N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0
        return grad
    
    bounds = [(0.0, None)] * N + [(1e-12, None)]
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]
    
    # Initial M estimate
    c0 = autoconv(h0)
    M0 = np.max(c0) + 1e-6
    x0 = np.concatenate([h0, [M0]])
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': ftol, 'iprint': 0}
        )
        if res.success:
            h_res = res.x[:N]
            h_res = np.clip(h_res, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
            else:
                return None
            return h_res
    except Exception:
        pass
    return None

def _find_sidon_support(N, K, attempts=300):
    """Find a Sidon set of size K in [0, N)"""
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums or sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def _gaussian_basis(center, width, N=50):
    """Generate a Gaussian-shaped vector of length N centered at 'center' with given width"""
    x = np.arange(N)
    g = np.exp(-0.5 * ((x - center) / width) ** 2)
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_h = np.ones(N) / N
    best_ratio = _ratio(best_h)
    
    # Strategy 1: Sidon-based initializations
    for K in range(14, 4, -1):
        if _time_left(start_time, deadline) < 25.0:
            break
        S = _find_sidon_support(N, K, attempts=200)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            r = _ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
            # Optimize from this starting point
            h_opt = _optimize_epigraph(h, start_time, deadline, maxiter=300, ftol=1e-10)
            if h_opt is not None:
                r_opt = _ratio(h_opt)
                if r_opt > best_ratio:
                    best_ratio = r_opt
                    best_h = h_opt.copy()
    
    # Strategy 2: Gaussian basis combinations
    # Try various Gaussian centers and widths
    if _time_left(start_time, deadline) > 30.0:
        np.random.seed(42)
        for trial in range(30):
            if _time_left(start_time, deadline) < 15.0:
                break
            
            # Random Gaussian mixture (1-3 components)
            n_comp = np.random.randint(1, 4)
            h = np.zeros(N)
            for _ in range(n_comp):
                center = np.random.uniform(5, N - 5)
                width = np.random.uniform(3, 12)
                amp = np.random.uniform(0.5, 2.0)
                h += amp * _gaussian_basis(center, width, N)
            
            if np.sum(h) > 0:
                h = h / np.sum(h)
                r = _ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h.copy()
                
                # Optimize from this Gaussian initial guess
                h_opt = _optimize_epigraph(h, start_time, deadline, maxiter=200, ftol=1e-9)
                if h_opt is not None:
                    r_opt = _ratio(h_opt)
                    if r_opt > best_ratio:
                        best_ratio = r_opt
                        best_h = h_opt.copy()
    
    # Strategy 3: Random sparse initializations
    if _time_left(start_time, deadline) > 20.0:
        for trial in range(20):
            if _time_left(start_time, deadline) < 10.0:
                break
            
            K_rand = np.random.randint(5, 16)
            support = np.random.choice(N, K_rand, replace=False)
            h = np.zeros(N)
            h[support] = 1.0 / K_rand
            
            r = _ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
            
            h_opt = _optimize_epigraph(h, start_time, deadline, maxiter=200, ftol=1e-9)
            if h_opt is not None:
                r_opt = _ratio(h_opt)
                if r_opt > best_ratio:
                    best_ratio = r_opt
                    best_h = h_opt.copy()
    
    # Final polish: Run SLSQP with tighter tolerance on the best solution
    if _time_left(start_time, deadline) > 5.0:
        h_final = _optimize_epigraph(best_h, start_time, deadline, maxiter=1000, ftol=1e-12)
        if h_final is not None:
            r_final = _ratio(h_final)
            if r_final > best_ratio:
                best_ratio = r_final
                best_h = h_final.copy()
    
    # Ensure valid output
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
