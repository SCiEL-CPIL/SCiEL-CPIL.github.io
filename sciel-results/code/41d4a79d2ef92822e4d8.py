"""Erdos minimum overlap - Enhanced Discrete + Continuous Optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    """Compute the exact value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    # Final fine-tuning for precision
    s = h.sum()
    if abs(s - target) > 1e-10:
        diff = target - s
        free = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(corr, beta):
    """Compute smooth max of correlation terms."""
    c = corr * 2.0 / N
    c_max = np.max(c)
    c_shifted = c - c_max
    soft_max = c_max + np.log(np.sum(np.exp(beta * c_shifted))) / beta
    return soft_max, c, c_max, c_shifted

def _objective_and_grad(x, beta=100.0):
    """Objective: smooth max of correlation terms with sum constraint penalty."""
    h = x
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    
    soft_max, c, c_max, c_shifted = _smooth_max(corr, beta)
    
    # Penalty for sum constraint
    sum_h = np.sum(h)
    sum_penalty = 1000.0 * (sum_h - N / 2.0) ** 2
    
    total = soft_max + sum_penalty
    
    # Gradient computation
    weights = np.exp(beta * c_shifted)
    weights = weights / np.sum(weights)
    grad_corr = weights * 2.0 / N
    
    grad_h = np.zeros(N)
    for k in range(2 * N - 1):
        gk = grad_corr[k]
        if abs(gk) < 1e-15:
            continue
        if k < N:
            for j in range(N - 1 - k):
                grad_h[j+k] += gk * (1.0 - h[j])
                grad_h[j] += gk * (-h[j+k])
        else:
            m = k - N + 1
            for j in range(m, N):
                grad_h[j] += gk * (1.0 - h[j-m])
                grad_h[j-m] += gk * (-h[j])
                
    grad_h += 2000.0 * (sum_h - N / 2.0)
    
    return total, grad_h

def local_search(h_init, max_epochs=100, time_limit=3.0):
    """Perform local search by swapping elements."""
    h = h_init.copy()
    current_val = compute_value(h)
    start_time = time.time()
    
    improved = True
    epoch = 0
    while improved and (time.time() - start_time) < time_limit and epoch < max_epochs:
        improved = False
        epoch += 1
        indices = np.random.permutation(N)
        for i in indices:
            if (time.time() - start_time) > time_limit:
                break
            for j in indices[i+1:]:
                if i == j:
                    continue
                h[i], h[j] = h[j], h[i]
                new_val = compute_value(h)
                if new_val < current_val - 1e-12:
                    current_val = new_val
                    improved = True
                else:
                    h[i], h[j] = h[j], h[i]
                    
    return h, current_val

def continuous_polish(h_start, beta=100.0, maxiter=500):
    """Run L-BFGS-B continuous optimization."""
    h_start = project(h_start)
    
    res = minimize(
        lambda x: _objective_and_grad(x, beta=beta)[0],
        h_start,
        method='L-BFGS-B',
        jac=lambda x: _objective_and_grad(x, beta=beta)[1],
        bounds=[(0, 1)] * N,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13}
    )
    
    h_polished = np.clip(res.x, 0, 1)
    h_polished = project(h_polished)
    return h_polished

def construct():
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Discrete Search with diverse multisets
    multisets = [
        [0.0] * 50 + [1.0] * 50,
        [0.25] * 50 + [0.75] * 50,
        [0.1] * 25 + [0.5] * 50 + [0.9] * 25,
        [0.0] * 25 + [0.5] * 50 + [1.0] * 25,
        [0.2] * 25 + [0.4] * 25 + [0.6] * 25 + [0.8] * 25,
        [0.1] * 20 + [0.3] * 20 + [0.5] * 20 + [0.7] * 20 + [0.9] * 20,
        [0.0] * 10 + [0.2] * 20 + [0.5] * 40 + [0.8] * 20 + [1.0] * 10,
        [0.1] * 15 + [0.3] * 15 + [0.5] * 40 + [0.7] * 15 + [0.9] * 15,
    ]
    
    for ms in multisets:
        if remaining() < 15.0:
            break
            
        for restart in range(8):
            if remaining() < 8.0:
                break
                
            h0 = np.array(ms, dtype=float)
            rng.shuffle(h0)
            
            h_opt, val = local_search(h0, max_epochs=150, time_limit=2.5)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
    # Phase 2: Continuous Polish with multiple beta values
    if remaining() > 20.0:
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if remaining() < 10.0:
                break
            try:
                h_polished = continuous_polish(best_h, beta=beta, maxiter=300)
                val = compute_value(h_polished)
                if val < best_val:
                    best_val = val
                    best_h = h_polished.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if remaining() > 15.0:
        try:
            h_polished = continuous_polish(best_h, beta=1000.0, maxiter=1000)
            val = compute_value(h_polished)
            if val < best_val:
                best_val = val
                best_h = h_polished.copy()
        except Exception:
            pass

    # Final verification
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
        
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
