"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def _overlap(h, s):
    """Compute overlap of h with shifted 1-h at shift s."""
    shifted = np.roll(h, s)
    return np.sum(h * (1.0 - shifted))

def _max_overlap(h):
    """Compute maximum overlap over all shifts."""
    overlaps = []
    for s in range(N):
        overlaps.append(_overlap(h, s))
    return max(overlaps)

def _logsumexp_overlap(h, beta=100.0):
    """Smooth approximation of max overlap using log-sum-exp."""
    overlaps = np.array([_overlap(h, s) for s in range(N)])
    max_ov = np.max(overlaps)
    if beta <= 0:
        return max_ov
    scaled = beta * (overlaps - max_ov)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -700, 700)
    return max_ov + np.log(np.sum(np.exp(scaled))) / beta

def _grad_logsumexp(h, beta=100.0):
    """Gradient of log-sum-exp approximation of max overlap."""
    overlaps = np.array([_overlap(h, s) for s in range(N)])
    max_ov = np.max(overlaps)
    scaled = beta * (overlaps - max_ov)
    scaled = np.clip(scaled, -700, 700)
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    grad = np.zeros(N)
    for s in range(N):
        shifted = np.roll(h, s)
        # d/dh[i] of sum_j h[j]*(1-h[j+s]) = (1-h[i+s]) - h[i-s]
        # where indices are mod N
        dhi = (1.0 - np.roll(h, -s)) - np.roll(h, s)
        grad += weights[s] * dhi
    return beta * grad

def _project(h, target_sum):
    """Project h onto [0,1]^N with sum = target_sum."""
    # Simple projection: clip and adjust
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-10:
        return h
    # Distribute the difference
    n_pos = np.sum(h < 1)
    n_neg = np.sum(h > 0)
    if diff > 0 and n_pos > 0:
        h[h < 1] += diff / n_pos
    elif diff < 0 and n_neg > 0:
        h[h > 0] += diff / n_neg
    return np.clip(h, 0, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    target_sum = N / 2.0
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try multiple initializations
    initializations = []
    initializations.append(np.full(N, 0.5))
    
    # Binary-like patterns
    for period in [2, 3, 4, 5, 7, 10]:
        if N % period == 0:
            pattern = []
            for k in range(period):
                if k < period // 2:
                    pattern.append(1.0)
                else:
                    pattern.append(0.0)
            h_init = np.tile(pattern, N // period)
            initializations.append(h_init)
    
    # Random initializations
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project(h_rand, target_sum)
        initializations.append(h_rand)
    
    for h_init in initializations:
        if time.time() > deadline:
            break
        
        h_init = _project(h_init, target_sum)
        
        # Phase 1: Coarse optimization with smooth surrogate
        def obj_smooth(h):
            return _logsumexp_overlap(h, beta=50.0)
        
        def grad_smooth(h):
            return _grad_logsumexp(h, beta=50.0)
        
        res = minimize(
            obj_smooth, h_init, jac=grad_smooth,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        h_opt = _project(res.x, target_sum)
        score = _max_overlap(h_opt)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # Phase 2: Polish with sharper log-sum-exp
        if time.time() > deadline:
            break
        
        def obj_sharp(h):
            return _logsumexp_overlap(h, beta=500.0)
        
        def grad_sharp(h):
            return _grad_logsumexp(h, beta=500.0)
        
        res2 = minimize(
            obj_sharp, h_opt, jac=grad_sharp,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        
        h_opt2 = _project(res2.x, target_sum)
        score2 = _max_overlap(h_opt2)
        if score2 < best_score:
            best_score = score2
            best_h = h_opt2.copy()
    
    # Final projection to ensure exact sum
    best_h = _project(best_h, target_sum)
    
    # Verify sum is exactly N/2
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Adjust one element
        idx = np.argmax(best_h < 1) if diff > 0 else np.argmin(best_h > 0)
        best_h[idx] += diff
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
