"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _objective_smooth(params, alpha=200.0):
    """Objective with epigraph: minimize t + penalty for violation.
    params = [h_0, ..., h_{N-1}, t]
    """
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    return t + penalty

def _objective_smooth_grad(params, alpha=200.0):
    """Objective with analytic gradient."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    val = t + penalty
    
    # Gradient
    grad = np.zeros(N + 1)
    # d/dt
    if smax > t:
        grad[N] = 1.0 - 2000.0 * (smax - t)
    else:
        grad[N] = 1.0
    
    # d/dh_i for i in 0..N-1
    # overlap_i = h_i * (1 - h_{i+1}) for i < N-1
    # overlap_{N-1} = h_{N-1} * (1 - 0) = h_{N-1}
    # d(overlap_i)/d(h_j):
    #   if j == i: (1 - h_{i+1})
    #   if j == i+1: -h_i
    #   else: 0
    # (for i < N-1; for i = N-1: d(overlap_{N-1})/d(h_{N-1}) = 1)
    
    w = _smooth_max_grad(overlap, alpha)
    # d(smax)/d(overlap_k) = w[k]
    
    # d(overlap_i)/d(h_j) matrix is sparse
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad[i] += w[i] * (1.0 - h[i+1])
            grad[i+1] += w[i] * (-h[i])
    # i = N-1: overlap = h[N-1], d/dh[N-1] = 1
    if w[N-1] > 1e-15:
        grad[N-1] += w[N-1] * 1.0
    
    # Penalty gradient
    if smax > t:
        dp = 2000.0 * (smax - t)
        # d(penalty)/d(h_j) = dp * d(smax)/d(h_j)
        # d(smax)/d(h_j) already computed above as part of grad before penalty
        # Actually we need to recompute: grad_smax_h = the h-gradient of smax
        # Let me recompute cleanly
        grad_smax_h = np.zeros(N)
        for i in range(N - 1):
            if w[i] > 1e-15:
                grad_smax_h[i] += w[i] * (1.0 - h[i+1])
                grad_smax_h[i+1] += w[i] * (-h[i])
        if w[N-1] > 1e-15:
            grad_smax_h[N-1] += w[N-1]
        
        grad[:N] += dp * grad_smax_h
        grad[N] -= dp  # d(penalty)/dt = -2000*(smax-t) when smax>t
    
    return val, grad

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(10):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _sinusoidal_h(A, phi, N=N):
    """Generate sinusoidal step heights, projected to [0,1] with sum=N/2."""
    x = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * x / N + phi)
    return _project_to_feasible(h)

def _sinusoidal_score(params):
    """Score for sinusoidal parameters [A, phi]."""
    A, phi = params
    h = _sinusoidal_h(A, phi)
    return _score(h)

def _optimize_sinusoidal():
    """Optimize sinusoidal parameters A and phi via grid search + local refinement."""
    best_score = float('inf')
    best_A = 0.0
    best_phi = 0.0
    
    # Grid search over A in [0, 0.5] and phi in [0, 2*pi)
    n_A = 40
    n_phi = 80
    for i in range(n_A):
        A = i * 0.5 / (n_A - 1)
        for j in range(n_phi):
            phi = j * 2.0 * np.pi / n_phi
            sc = _sinusoidal_score([A, phi])
            if sc < best_score:
                best_score = sc
                best_A = A
                best_phi = phi
    
    # Local refinement using Nelder-Mead
    try:
        res = minimize(_sinusoidal_score, [best_A, best_phi], method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000})
        if res.fun < best_score:
            best_score = res.fun
            best_A = res.x[0]
            best_phi = res.x[1]
    except Exception:
        pass
        
    h_best = _sinusoidal_h(best_A, best_phi)
    return h_best, best_score

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Phase 0: Low-dimensional sinusoidal optimization
    # This often finds a much better global candidate than the 2-block init
    h_sinusoidal, score_sinusoidal = _optimize_sinusoidal()
    
    h0 = _two_block_init()
    score_2block = _score(h0)
    
    # Choose the better of the two as the initial guess for high-dim optimization
    if score_sinusoidal < score_2block:
        h_final = h_sinusoidal.copy()
        current_score = score_sinusoidal
    else:
        h_final = h0.copy()
        current_score = score_2block
        
    # Phase 1: SLSQP with epigraph formulation
    try:
        def objective_epigraph(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=200.0)
            return t + 1000.0 * np.maximum(0.0, smax - t)**2
        
        x0 = np.concatenate([h_final, [current_score + 0.05]])
        bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
        
        def constraint_sum(params):
            return np.sum(params[:N]) - N/2.0
        
        def constraint_epigraph(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return t - np.max(overlap)
        
        constraints_ext = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_epigraph}
        ]
        
        result = minimize(
            objective_epigraph,
            x0,
            method='SLSQP',
            bounds=bounds_ext,
            constraints=constraints_ext,
            options={'maxiter': 1500, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.fun < 100.0:
            h_opt = result.x[:N].copy()
            h_opt = _project_to_feasible(h_opt)
            new_score = _score(h_opt)
            if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                h_final = h_opt.copy()
                current_score = new_score
    except Exception:
        pass
    
    # Phase 2: High-precision polish with tighter alpha
    try:
        if time.time() < deadline - 15.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 10000.0 * np.maximum(0.0, smax - t)**2
            
            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0
            
            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)
            
            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]
            
            result2 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
            )
            
            if result2.fun < 100.0:
                h_opt2 = result2.x[:N].copy()
                h_opt2 = _project_to_feasible(h_opt2)
                new_score2 = _score(h_opt2)
                if new_score2 < current_score and np.all(h_opt2 >= 0.0) and np.all(h_opt2 <= 1.0):
                    h_final = h_opt2.copy()
                    current_score = new_score2
    except Exception:
        pass
    
    # Phase 3: Local search refinement
    try:
        if time.time() < deadline - 10.0:
            rng = np.random.default_rng(42)
            h = h_final.copy()
            best_local = current_score
            
            for iteration in range(5000):
                if time.time() > deadline - 5.0:
                    break
                
                # Try different step sizes
                step_size = max(0.001, 0.05 * (1.0 - iteration / 5000.0))
                
                h_trial = h.copy()
                n_perturb = rng.integers(1, 5)
                idxs = rng.choice(N, size=n_perturb, replace=False)
                for i in idxs:
                    h_trial[i] = np.clip(h_trial[i] + rng.uniform(-step_size, step_size), 0.0, 1.0)
                
                h_trial = _project_to_feasible(h_trial)
                sc = _score(h_trial)
                
                if sc < best_local - 1e-12:
                    best_local = sc
                    h = h_trial.copy()
                    h_final = h_trial.copy()
                    current_score = sc
    except Exception:
        pass
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    final_score = _score(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final
