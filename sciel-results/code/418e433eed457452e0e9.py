"""Program for Erdos minimum overlap task using spectral model + SLSQP."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=50.0):
    """Compute the smooth value to minimize using log-sum-exp."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
    return float(log_sum / N * 2.0)

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning
    for _ in range(20):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        diff = target - s
        free = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
        else:
            break
    return h

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Divisors of 100: 1, 2, 4, 5, 10, 20, 25, 50, 100
    # We'll use frequencies that are divisors for spectral symmetry
    divisors = [1, 2, 4, 5, 10, 20, 25, 50]
    
    # Strategy 1: Spectral parameterization with SLSQP
    # h(i) = 0.5 + sum_k (a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    # Optimize a_k, b_k with constraints: 0 <= h(i) <= 1, sum(h) = N/2
    
    for n_freqs in range(1, min(len(divisors) + 1, 7)):
        if remaining() < 5:
            break
        
        freq_subset = divisors[:n_freqs]
        n_params = 2 * n_freqs
        
        # Precompute basis vectors
        basis = []
        for k in freq_subset:
            cos_k = np.cos(2 * np.pi * k * i / N)
            sin_k = np.sin(2 * np.pi * k * i / N)
            basis.append((cos_k, sin_k))
        
        def params_to_h(params):
            h = np.full(N, 0.5)
            for idx, k in enumerate(freq_subset):
                a = params[2 * idx]
                b = params[2 * idx + 1]
                cos_k, sin_k = basis[idx]
                h += a * cos_k + b * sin_k
            return h
        
        # Multiple random initializations
        np.random.seed(42 + n_freqs)
        for trial in range(3):
            if remaining() < 3:
                break
            
            params_init = np.random.randn(n_params) * 0.1
            
            def obj_spectral(params):
                h = params_to_h(params)
                h_proj = project(h)
                return objective_smooth(h_proj)
            
            # Constraints: 0 <= h(i) <= 1 for all i, and sum(h) = N/2
            # SLSQP handles inequality constraints as fun >= 0
            def ineq_constraints(params):
                h = params_to_h(params)
                # h >= 0: fun = h
                # 1-h >= 0: fun = 1-h
                return np.concatenate([h, 1.0 - h])
            
            def eq_constraint(params):
                h = params_to_h(params)
                return h.sum() - N / 2.0
            
            constraints = [
                {'type': 'ineq', 'fun': ineq_constraints},
                {'type': 'eq', 'fun': eq_constraint}
            ]
            
            try:
                result = minimize(
                    obj_spectral,
                    params_init,
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                
                h_candidate = project(params_to_h(result.x))
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 2: Multi-start with cosine patterns (direct h-space)
    for k in range(1, 12):
        if remaining() < 3:
            break
        
        for A_init in [0.2, 0.3, 0.4, 0.5]:
            if remaining() < 2:
                break
            
            h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N)
            h_init = project(h_init)
            
            def obj_func(x):
                return objective_smooth(project(x))
            
            try:
                result = minimize(
                    obj_func,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 3: Random perturbations around best solution
    if remaining() > 15:
        np.random.seed(123)
        for _ in range(15):
            if remaining() < 5:
                break
            
            h_pert = best_h + 0.03 * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_func2(x):
                return objective_smooth(project(x))
            
            try:
                result = minimize(
                    obj_func2,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 4: Binary local search
    if remaining() > 15:
        h_bin = (best_h > 0.5).astype(np.float64)
        s = h_bin.sum()
        if s != N / 2.0:
            idx = np.argsort(np.abs(best_h - 0.5))
            if s < N / 2.0:
                for i in idx:
                    if h_bin[i] == 0.0:
                        h_bin[i] = 1.0
                        s += 1
                        if s == N / 2.0:
                            break
            else:
                for i in idx:
                    if h_bin[i] == 1.0:
                        h_bin[i] = 0.0
                        s -= 1
                        if s == N / 2.0:
                            break
        
        improved = True
        while improved and remaining() > 8:
            improved = False
            current_val = objective(h_bin)
            for i in range(N):
                if remaining() < 8:
                    break
                for j in range(i + 1, N):
                    if h_bin[i] == h_bin[j]:
                        continue
                    h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
                    new_val = objective(h_bin)
                    if new_val < current_val:
                        current_val = new_val
                        improved = True
                    else:
                        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
            
            if current_val < best_val:
                best_h = h_bin.copy()
                best_val = current_val
    
    # Final polish
    if remaining() > 5:
        def obj_final(x):
            return objective_smooth(project(x))
        
        try:
            result = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    best_h = project(best_h)
    return best_h
