import numpy as np
import time

N = 50
DEADLINE = 120.0  # seconds

# EVOLVE-BLOCK-START
def _compute_ratio(h):
    """Compute the ratio L2^2 / (L1 * ConvInf) for a discrete sequence."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return -1.0  # invalid
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1.0
    l2_sq = np.sum(h ** 2)
    # Compute autoconvolution (linear convolution with itself)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return -1.0
    return l2_sq / (l1 * conv_inf)

def _ratio_from_top_hats(w, x):
    """
    Given weights w and positions x for top-hat basis functions,
    compute the discrete sequence h and its ratio.
    """
    t = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N, dtype=np.float64)
    for wi, xi in zip(w, x):
        if wi > 1e-15:
            mask = (t - xi >= 0) & (t - xi < 1)
            h[mask] += wi
    return _compute_ratio(h), h

def _optimize_kpoint(w_init, x_init, max_iters=500, tol=1e-12):
    """
    Optimize a k-point top-hat cluster using coordinate descent.
    Parameters: w (weights), x (positions)
    """
    k = len(w_init)
    w = np.array(w_init, dtype=np.float64)
    x = np.array(x_init, dtype=np.float64)
    
    # Ensure valid initial values
    w = np.maximum(w, 0.0)
    x = np.clip(x, -0.25, 0.25)
    
    best_val, best_h = _ratio_from_top_hats(w, x)
    best_w, best_x = w.copy(), x.copy()
    
    if best_val < 0:
        return best_h, best_val
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 20:
            break
        improved = False
        
        # Optimize weights
        for i in range(k):
            step = 0.01
            for _ in range(15):
                trial_w = best_w.copy()
                trial_w[i] += step
                trial_w = np.maximum(trial_w, 0.0)
                val, h = _ratio_from_top_hats(trial_w, best_x)
                if val > best_val + tol:
                    best_val = val
                    best_w = trial_w
                    best_h = h
                    improved = True
                    step *= 1.3
                else:
                    step *= 0.5
                    # Try negative direction
                    trial_w2 = best_w.copy()
                    trial_w2[i] -= step
                    trial_w2 = np.maximum(trial_w2, 0.0)
                    val2, h2 = _ratio_from_top_hats(trial_w2, best_x)
                    if val2 > best_val + tol:
                        best_val = val2
                        best_w = trial_w2
                        best_h = h2
                        improved = True
                        step *= 1.3
                    else:
                        break
        
        # Optimize positions
        for i in range(k):
            step = 0.005
            for _ in range(15):
                trial_x = best_x.copy()
                trial_x[i] += step
                trial_x[i] = np.clip(trial_x[i], -0.25, 0.25)
                val, h = _ratio_from_top_hats(best_w, trial_x)
                if val > best_val + tol:
                    best_val = val
                    best_x = trial_x
                    best_h = h
                    improved = True
                    step *= 1.3
                else:
                    step *= 0.5
                    # Try negative direction
                    trial_x2 = best_x.copy()
                    trial_x2[i] -= step
                    trial_x2[i] = np.clip(trial_x2[i], -0.25, 0.25)
                    val2, h2 = _ratio_from_top_hats(best_w, trial_x2)
                    if val2 > best_val + tol:
                        best_val = val2
                        best_x = trial_x2
                        best_h = h2
                        improved = True
                        step *= 1.3
                    else:
                        break
        
        if not improved:
            break
    
    return best_h, best_val

def _optimize_discrete(h_init, max_iters=200):
    """
    Directly optimize the discrete heights using coordinate descent.
    """
    h = np.array(h_init, dtype=np.float64)
    h = np.maximum(h, 0.0)
    best_val = _compute_ratio(h)
    best_h = h.copy()
    
    if best_val < 0:
        return best_h, best_val
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 15:
            break
        improved = False
        
        for i in range(N):
            step = 0.01
            for _ in range(10):
                trial = best_h.copy()
                trial[i] += step
                val = _compute_ratio(trial)
                if val > best_val + 1e-12:
                    best_val = val
                    best_h = trial
                    improved = True
                    step *= 1.2
                else:
                    step *= 0.5
                    trial2 = best_h.copy()
                    trial2[i] -= step
                    trial2[i] = max(trial2[i], 0.0)
                    val2 = _compute_ratio(trial2)
                    if val2 > best_val + 1e-12:
                        best_val = val2
                        best_h = trial2
                        improved = True
                        step *= 1.2
                    else:
                        break
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Try various top-hat configurations and discrete optimizations.
    """
    start_time = time.time()
    
    best_h = None
    best_val = -1.0
    
    # Candidate 1: 2-point cluster (from seed)
    candidates_2pt = [
        ([2.0/3, 1.0/3], [-0.25, -0.25]),
        ([1.0, 0.5], [-0.25, -0.15]),
        ([1.5, 0.5], [-0.25, -0.2]),
        ([1.0, 1.0], [-0.25, -0.1]),
        ([2.0, 1.0], [-0.2, -0.1]),
        ([1.0, 0.5], [-0.25, -0.05]),
        ([1.5, 0.5], [-0.15, -0.1]),
        ([1.0, 1.0], [-0.25, 0.0]),
    ]
    
    for w_init, x_init in candidates_2pt:
        if time.time() > DEADLINE - 25:
            break
        h, val = _optimize_kpoint(w_init, x_init, max_iters=300)
        if val > best_val:
            best_val = val
            best_h = h
    
    # Candidate 2: 3-point cluster
    candidates_3pt = [
        ([1.0, 0.5, 0.5], [-0.25, -0.15, -0.05]),
        ([1.5, 0.5, 0.3], [-0.25, -0.2, -0.1]),
        ([1.0, 1.0, 0.5], [-0.25, -0.1, 0.0]),
    ]
    
    for w_init, x_init in candidates_3pt:
        if time.time() > DEADLINE - 20:
            break
        h, val = _optimize_kpoint(w_init, x_init, max_iters=200)
        if val > best_val:
            best_val = val
            best_h = h
    
    # Candidate 3: Discrete optimization from various starts
    discrete_starts = [
        [2.0/3, 1.0/3] + [0.0] * (N - 2),
        [1.0, 0.5, 0.3] + [0.0] * (N - 3),
        [0.5] * 5 + [0.0] * (N - 5),
        [1.0, 0.8, 0.6, 0.4, 0.2] + [0.0] * (N - 5),
        [1.0, 0.0, 1.0, 0.0, 1.0] + [0.0] * (N - 5),
    ]
    
    for h_init in discrete_starts:
        if time.time() > DEADLINE - 10:
            break
        h, val = _optimize_discrete(h_init, max_iters=100)
        if val > best_val:
            best_val = val
            best_h = h
    
    # Final polish: try small perturbations on the best solution
    if best_h is not None and best_val > 0:
        if time.time() < DEADLINE - 5:
            h_polished, val_polished = _optimize_discrete(best_h, max_iters=50)
            if val_polished > best_val:
                best_val = val_polished
                best_h = h_polished
    
    # Fallback
    if best_h is None or best_val < 0:
        h = [0.0] * N
        h[0] = 2.0 / 3.0
        h[1] = 1.0 / 3.0
        return h
    
    return list(best_h)
# EVOLVE-BLOCK-END
