import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Exact max overlap score."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Project h to satisfy sum(target) and bounds [0,1]."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        return np.full(N, target / N)
    
    # Simple scaling
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    
    # Iterative correction for clipping issues
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-12:
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
        
    # Final exact adjustment
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        # Adjust the element with the most room
        if diff > 0:
            idx = int(np.argmin(h))
        else:
            idx = int(np.argmax(h))
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _smooth_max_and_grad(x, alpha):
    """
    Compute smooth max (log-sum-exp) and its gradient.
    x: vector of overlaps
    alpha: sharpness parameter
    """
    x_max = np.max(x)
    # Numerical stability
    shifted = x - x_max
    exps = np.exp(alpha * shifted)
    sum_exps = np.sum(exps)
    
    if sum_exps == 0:
        return x_max, np.zeros_like(x)
        
    val = x_max + (1.0 / alpha) * np.log(sum_exps)
    grad = exps / sum_exps
    return val, grad

def _objective_and_grad(h, alpha):
    """
    Objective: Smooth Max of overlaps + Penalty for sum constraint.
    Returns (value, gradient).
    """
    n = len(h)
    # Calculate overlaps: o_i = h_i * (1 - h_{i+1})
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1] # h_{N-1} * (1 - 0)
    else:
        overlap[0] = h[0]
        
    smax_val, grad_smax_overlap = _smooth_max_and_grad(overlap, alpha)
    
    # Gradient of overlap w.r.t h
    # o_i depends on h_i and h_{i+1}
    grad_h = np.zeros(n)
    if n > 1:
        # d(o_i)/d(h_i) = 1 - h_{i+1}
        grad_h[:-1] += grad_smax_overlap[:-1] * (1.0 - h[1:])
        # d(o_i)/d(h_{i+1}) = -h_i
        grad_h[1:] += grad_smax_overlap[:-1] * (-h[:-1])
        # d(o_{N-1})/d(h_{N-1}) = 1
        grad_h[-1] += grad_smax_overlap[-1]
    else:
        grad_h[0] += grad_smax_overlap[0]
        
    # Penalty for sum constraint: lambda * (sum(h) - target)^2
    # We use a soft penalty in the objective, but also project afterwards.
    # A large lambda helps keep the sum close during optimization.
    lambda_penalty = 100.0
    sum_err = h.sum() - TARGET_SUM
    penalty_val = lambda_penalty * sum_err * sum_err
    penalty_grad = 2.0 * lambda_penalty * sum_err * np.ones(n)
    
    total_val = smax_val + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad

def _continuation_optimize(h_init, alphas, deadline):
    """
    Run L-BFGS-B with increasing alpha (continuation method).
    """
    h = h_init.copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()
    
    bounds = [(0.0, 1.0)] * N
    
    for alpha in alphas:
        if time.time() > deadline - 2.0:
            break
            
        def obj(x):
            val, _ = _objective_and_grad(x, alpha)
            return val
            
        def grad(x):
            _, g = _objective_and_grad(x, alpha)
            return g
            
        try:
            # L-BFGS-B
            res = minimize(
                obj, 
                h, 
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            score_new = _score(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
            else:
                # If it got worse, keep the previous best for the next stage?
                # Usually, we want to continue from the optimized point even if score is slightly worse due to smoothing,
                # but if it's significantly worse, maybe revert?
                # Let's always use the optimized point for the next stage as it's closer to the local min of the current alpha.
                h = h_new
            # Update h to the result of this stage for the next iteration
            h = h_new
            
        except Exception:
            pass
            
    return best_h, best_score

def _local_polish(h, deadline):
    """
    Fine-grained local search to polish the discrete max.
    """
    h = h.copy()
    current_score = _score(h)
    
    # Try small perturbations on the indices contributing to the max
    # Identify active constraints (indices where overlap is close to max)
    overlap = np.zeros(N)
    overlap[:-1] = h[:-1] * (1.0 - h[1:])
    overlap[-1] = h[-1]
    
    max_val = np.max(overlap)
    # Indices where overlap is within 1% of max
    active_idx = np.where(overlap > max_val * 0.99)[0]
    
    if len(active_idx) == 0:
        active_idx = np.arange(N)
        
    # Local search on active indices and their neighbors
    steps = [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in active_idx:
            # Check i and i+1 (since overlap[i] depends on h[i] and h[i+1])
            for j in [i, (i+1) % N]:
                for delta in steps:
                    h_trial = h.copy()
                    h_trial[j] += delta
                    if h_trial[j] < 0 or h_trial[j] > 1:
                        continue
                    h_trial = _fix_sum(h_trial)
                    s = _score(h_trial)
                    if s < current_score - 1e-12:
                        h = h_trial
                        current_score = s
                        improved = True
                        # Re-calculate active indices if needed, or just break inner loops
                        break
                if improved:
                    break
            if improved:
                break
        # Update active indices
        overlap = np.zeros(N)
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
        max_val = np.max(overlap)
        active_idx = np.where(overlap > max_val * 0.99)[0]
        if len(active_idx) == 0:
            active_idx = np.arange(N)

    return h, current_score

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A                        # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return _fix_sum(h0)

def _random_init(seed):
    """Random initial guess."""
    np.random.seed(seed)
    h = np.random.rand(N)
    return _fix_sum(h)

def _sinusoidal_init(frequency, phase):
    """Sinusoidal initial guess."""
    i = np.arange(N)
    h = 0.5 + 0.3 * np.sin(2.0 * np.pi * frequency * i / N + phase)
    return _fix_sum(h)

def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    
    # List of initial guesses
    inits = []
    inits.append(_two_block_init())
    
    # Add some diverse inits
    for seed in [1, 2, 3]:
        inits.append(_random_init(seed))
    inits.append(_sinusoidal_init(1, 0))
    inits.append(_sinusoidal_init(2, np.pi/2))
    inits.append(_sinusoidal_init(3, np.pi))
    
    # Continuation alphas
    alphas = [10.0, 50.0, 200.0, 1000.0]
    
    best_h = None
    best_score = float('inf')
    
    for h_init in inits:
        if time.time() > deadline - 20.0:
            break
            
        h_opt, score_opt = _continuation_optimize(h_init, alphas, deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # Final polish on the best result
    if best_h is not None and time.time() < deadline - 5.0:
        best_h, best_score = _local_polish(best_h, deadline)
        
    # Final projection
    best_h = _fix_sum(best_h)
    
    return best_h
