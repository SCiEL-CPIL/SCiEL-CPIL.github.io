"""Program for Erdos minimum overlap task using model-and-optimize approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 142.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_and_grad_smooth(h, beta=100.0):
    """Smooth surrogate with analytical gradient for L-BFGS-B."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    val = (c_max + log_sum / beta) / N * 2.0
    
    # Gradient of soft-max: w_k = exp(beta*(c_k - c_max)) / sum
    w = exp_shifted / sum_exp
    # dc_k/dh_i for each lag k: c_k = sum_j h[j]*(1-h[j+k]) or similar
    # np.correlate(h, 1-h, 'full') gives lags from -(N-1) to (N-1)
    # c[k] = sum_j h[j] * (1-h[j + (k - (N-1))])
    # dc_k/dh_i = (1-h[i + (k-(N-1))]) - h[i] * (if index valid for second term)
    # Actually: c_k = sum_j h[j]*(1-h[j+k-(N-1)])
    # dc_k/dh_i: from first term: (1 - h[i+k-(N-1)]) if i+k-(N-1) in [0,N)
    #            from second term: -h[i] if i in [0,N) and i+k-(N-1) in [0,N) -> -h[i+k-(N-1)] wait
    # Let me be more careful. c_k = sum_j h[j] * (1 - h[j + d]) where d = k - (N-1)
    # dc_k/dh_i: 
    #   term1: h[j] with j=i contributes (1 - h[i+d]) if i+d in [0,N)
    #   term2: -h[j]*h[j+d] with j+d=i contributes -h[i-d] if i-d in [0,N)
    #   also -h[j]*h[j+d] with j=i contributes -h[i]*h[i+d]... no
    # Actually c_k = sum_j h[j] - sum_j h[j]*h[j+d]
    # dc_k/dh_i = 1 - h[i+d] (from first sum, j=i) - h[i-d] (from second sum, j+d=i, so j=i-d)
    #            - h[i] (from second sum, j=i) ... wait
    # sum_j h[j]*h[j+d]: derivative wrt h[i] is h[i+d] (when j=i) + h[i-d] (when j+d=i, i.e., j=i-d)
    # So dc_k/dh_i = 1 - h[i+d] - h[i-d] - h[i+d]... no
    # d/dh_i [sum_j h[j]*h[j+d]] = h[i+d] + h[i-d] (two terms)
    # So dc_k/dh_i = 1 - (h[i+d] + h[i-d])
    
    d = np.arange(len(c)) - (N - 1)
    # For each i and each lag k, compute dc_k/dh_i
    # grad_i = sum_k w_k * dc_k/dh_i = sum_k w_k * (1 - h[i+d_k] - h[i-d_k])
    # where h[x] = 0 if x out of [0,N)
    
    # Vectorized: for each i, sum over k of w_k * (1 - h[i+d_k] - h[i-d_k])
    # h[i+d_k] is nonzero only when 0 <= i+d_k < N
    # h[i-d_k] is nonzero only when 0 <= i-d_k < N
    
    grad = np.zeros(N)
    for k in range(len(c)):
        if w[k] < 1e-12:
            continue
        d_k = d[k]
        # h[i + d_k] valid for i in [max(0, -d_k), min(N, N-d_k))
        i_lo = max(0, -d_k)
        i_hi = min(N, N - d_k)
        if i_lo < i_hi:
            grad[i_lo:i_hi] += w[k] * (1.0 - h[i_lo + d_k:i_hi + d_k])
        # h[i - d_k] valid for i in [max(0, d_k), min(N, N + d_k))
        i_lo2 = max(0, d_k)
        i_hi2 = min(N, N + d_k)
        if i_lo2 < i_hi2:
            grad[i_lo2:i_hi2] += w[k] * (-h[i_lo2 - d_k:i_hi2 - d_k])
        # The constant 1 term
        grad += w[k] * 1.0
    
    grad *= 2.0 / N
    return val, grad

def objective_epigraph(params, beta=50.0):
    """Epigraph formulation: minimize t subject to c_i <= t for all i."""
    h = params[:N]
    t = params[N]
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    smooth_max = np.log(np.sum(np.exp(shifted))) / beta + c_max
    penalty = np.maximum(0, smooth_max - t) ** 2 * 100.0
    return float(t / N * 2.0 + penalty)

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
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
    
    for _ in range(30):
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

def local_search_refine(h, max_iter=200, tol=1e-14):
    """Faster coordinate descent: only 4 delta sizes, fewer iterations."""
    h = project(h.copy())
    best_val = objective(h)
    deltas = [0.02, -0.02, 0.005, -0.005]
    
    for it in range(max_iter):
        if remaining() < 3:
            break
        improved = False
        for i in range(N):
            if remaining() < 1:
                break
            for delta in deltas:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = project(h_trial)
                val = objective(h_trial)
                if val < best_val - tol:
                    h = h_trial
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    
    return h, best_val

def construct():
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy 1: Direct smooth optimization with gradient (faster than epigraph)
    if remaining() > 10:
        for n_modes in range(1, 5):
            if remaining() < 5:
                break
            
            np.random.seed(42 + n_modes)
            h_init = np.full(N, 0.5)
            for k in range(n_modes):
                a = 0.1 * np.random.randn()
                b = 0.1 * np.random.randn()
                h_init += a * np.cos(2 * np.pi * k * i / N)
                h_init += b * np.sin(2 * np.pi * k * i / N)
            h_init = project(h_init)
            
            def obj_func(x, b=60.0):
                h_proj = project(x)
                val, grad = objective_and_grad_smooth(h_proj, beta=b)
                return val, grad
            
            try:
                result = minimize(
                    obj_func,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 2: Direct optimization with cosine modes (reduced search space)
    for k in range(1, 10):
        if remaining() < 3:
            break
        
        for A_init in [0.3, 0.4, 0.45]:
            if remaining() < 2:
                break
            
            for phase in [0.0, np.pi/3, 2*np.pi/3]:
                if remaining() < 1:
                    break
                
                h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N + phase)
                h_init = project(h_init)
                
                for beta in [50.0, 100.0]:
                    if remaining() < 1:
                        break
                    
                    def obj_func(x, b=beta):
                        h_proj = project(x)
                        val, grad = objective_and_grad_smooth(h_proj, beta=b)
                        return val, grad
                    
                    try:
                        result = minimize(
                            obj_func,
                            h_init,
                            method='L-BFGS-B',
                            bounds=[(0.0, 1.0)] * N,
                            jac=True,
                            options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        
                        h_candidate = project(result.x)
                        val_candidate = objective(h_candidate)
                        
                        if val_candidate < best_val:
                            best_h = h_candidate
                            best_val = val_candidate
                            h_init = h_candidate.copy()
                            
                    except Exception:
                        pass
    
    # Strategy 3: Random perturbations around best solution with smooth objective
    if remaining() > 8:
        np.random.seed(123)
        for _ in range(10):
            if remaining() < 3:
                break
            
            scale = 0.02 + 0.04 * np.random.rand()
            h_pert = best_h + scale * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_pert(x, b=80.0):
                h_proj = project(x)
                val, grad = objective_and_grad_smooth(h_proj, beta=b)
                return val, grad
            
            try:
                result = minimize(
                    obj_pert,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 4: Local search refinement (coordinate descent)
    if remaining() > 5:
        h_refined, val_refined = local_search_refine(best_h, max_iter=150)
        if val_refined < best_val:
            best_h = h_refined
            best_val = val_refined
    
    # Final polish: high-precision refinement
    if remaining() > 5:
        # Pass 1: smooth objective at high beta with gradient
        def obj_high_beta(x):
            h_proj = project(x)
            val, grad = objective_and_grad_smooth(h_proj, beta=200.0)
            return val, grad
        
        try:
            result = minimize(
                obj_high_beta,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                jac=True,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
        
        # Pass 2: local search again
        if remaining() > 3:
            h_refined, val_refined = local_search_refine(best_h, max_iter=100)
            if val_refined < best_val:
                best_h = h_refined
                best_val = val_refined
    
    # Final validation and projection
    best_h = project(best_h)
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    if abs(best_h.sum() - N/2) > 1e-9 * N:
        best_h = project(best_h)
    if not np.all((best_h >= 0) & (best_h <= 1)):
        best_h = project(best_h)
    
    return best_h
