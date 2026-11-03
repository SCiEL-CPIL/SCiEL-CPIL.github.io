import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 145  # Safety margin

def compute_score(f):
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def gradient_smooth_max_autoconv(f, beta):
    """
    Compute the gradient of the smooth max of the autoconvolution of f.
    f: array of length N
    beta: sharpness parameter
    """
    # 1. Compute autoconvolution
    c = np.convolve(f, f, mode='full')
    
    # 2. Compute weights w_k = softmax(beta * c_k)
    # Use log-sum-exp trick for stability
    c_max = np.max(c)
    exp_c = np.exp(beta * (c - c_max))
    sum_exp = np.sum(exp_c)
    if sum_exp == 0:
        return np.zeros_like(f)
    w = exp_c / sum_exp
    
    # 3. Compute correlation of w and f
    # grad_j = sum_k w_k * (f_{k-j} + f_{j-k})
    # This is corr(w, f)[j] + corr(w, f)[-j]
    # np.correlate(w, f, 'full') computes sum_k w_k * f_{k-j}
    # The output has length len(w) + len(f) - 1 = (2N-1) + N - 1 = 3N - 2
    # The index corresponding to lag j is j + (len(f) - 1) = j + N - 1
    corr = np.correlate(w, f, mode='full')
    
    # We need grad_j for j in [0, N-1]
    # corr index for lag j is j + N - 1
    # corr index for lag -j is -j + N - 1
    
    grad = np.zeros(N)
    for j in range(N):
        idx_pos = j + N - 1
        idx_neg = -j + N - 1
        # Ensure indices are within bounds
        if 0 <= idx_pos < len(corr) and 0 <= idx_neg < len(corr):
            grad[j] = corr[idx_pos] + corr[idx_neg]
            
    return grad

def subgradient_descent(f_init, start_time, max_time, beta_schedule=None):
    """
    Perform subgradient descent to minimize the smooth max of autoconvolution.
    """
    f = f_init.copy()
    best_f = f.copy()
    best_score = compute_score(f)
    
    # Adaptive learning rate
    lr = 1e-4
    beta = 5.0
    
    # Schedule for beta: increase over time for sharper max approximation
    total_steps_est = 5000
    step = 0
    
    while time.time() - start_time < max_time:
        # Update beta
        if beta_schedule:
             beta = beta_schedule(step, total_steps_est)
        else:
             beta = 5.0 + 20.0 * (step / total_steps_est)
        
        # Compute gradient
        grad = gradient_smooth_max_autoconv(f, beta)
        
        # Normalize gradient to have unit norm for stable step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Update
        f_new = f - lr * grad
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0.0)
        
        # Normalize to keep sum constant (optional, but helps with scale invariance)
        s = np.sum(f_new)
        if s > 1e-12:
            f_new = f_new * (N / s)
        else:
            break
            
        # Check score
        score_new = compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            f = f_new
            # Increase lr slightly if improving
            lr = min(lr * 1.01, 1e-2)
        else:
            # Decrease lr
            lr *= 0.95
            # If lr is too small, break
            if lr < 1e-8:
                break
        
        step += 1
        
        # Adaptive beta: increase beta to focus more on the peak
        if step % 100 == 0:
             beta = min(beta * 1.1, 50.0)
             
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Random
    rng = np.random.RandomState(42)
    candidates.append(rng.rand(N))
    
    # 3. Cosine bumps (from previous good ideas)
    for n_c in [3, 4, 5]:
        centers = np.linspace(-0.2, 0.2, n_c)
        widths = np.full(n_c, 0.15)
        heights = np.full(n_c, 1.0)
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        f = np.zeros(N)
        for c, w, h in zip(centers, widths, heights):
            dist = np.abs(x - c)
            mask = dist <= w / 2.0
            if np.any(mask):
                f[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
        candidates.append(f)
        
    best_f = None
    best_score = np.inf
    
    # Run subgradient descent on each candidate
    # Allocate time per candidate
    time_per_candidate = (TIME_LIMIT - 10) / len(candidates)
    
    for i, f_init in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        f_opt, score_opt = subgradient_descent(
            f_init, 
            start_time, 
            time.time() + time_per_candidate,
            beta_schedule=lambda step, total: 5.0 + 20.0 * (step / total)
        )
        
        if score_opt < best_score:
            best_score = score_opt
            best_f = f_opt
    
    # Final polish with very small steps and high beta
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 2:
        # Use the best found as init for a final polish
        f_polished, score_polished = subgradient_descent(
            best_f,
            start_time,
            TIME_LIMIT - 1,
            beta_schedule=lambda step, total: 30.0 + 20.0 * (step / total)
        )
        if score_polished < best_score:
            best_score = score_polished
            best_f = f_polished
            
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
