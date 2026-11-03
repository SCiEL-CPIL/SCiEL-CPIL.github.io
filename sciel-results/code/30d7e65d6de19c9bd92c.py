"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 135.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def compute_overlap(x, shift):
    """
    Compute the overlap integral of x and its shifted version (1-x).
    x is a step function with N steps of equal width.
    shift is in [0, N) representing the number of steps to shift.
    
    The overlap is sum_i x_i * (1 - x_{(i+shift) % N}) * (width)
    But wait, the problem says "maximum overlap of h with shifted 1-h".
    
    Let h be a function on [0,2] with step heights. The domain is [0,2].
    If we have N steps, each has width 2/N.
    h(t) = h_i for t in [2i/N, 2(i+1)/N).
    
    The shifted function: h(t - s) for some shift s.
    We want to minimize the maximum over all shifts s of the overlap integral.
    
    Overlap of h and (1 - h_shifted):
    Integral_0^2 h(t) * (1 - h(t - s)) dt
    
    Since h is a step function, and we're shifting by s, the overlap depends on s.
    For discrete shifts, if we shift by k steps (s = 2k/N), then:
    h(t - s) at step i is h_{(i-k) % N}
    
    So overlap_k = sum_i h_i * (1 - h_{(i-k)%N}) * (2/N)
    
    But the shift s can be continuous. However, for step functions, the overlap
    as a function of s is piecewise linear, with breakpoints at s = 2k/N.
    The maximum will occur at one of these breakpoints or at the boundaries.
    
    Actually, let's think more carefully. If h is a step function with N steps,
    and we shift by s, the overlap integral is:
    sum over i of [integral over the region where both h and h_shifted are constant]
    
    For a shift s, let s = 2k/N + delta, where 0 <= delta < 2/N.
    Then for each step i, the overlap involves partial overlaps.
    
    However, for the purpose of optimization, we can evaluate the overlap at
    discrete shifts s = 2k/N for k = 0, 1, ..., N-1, and also consider that
    the maximum over continuous s might be at these points.
    
    Actually, for step functions, the maximum overlap with a shifted version
    occurs at discrete shifts that are multiples of the step width. This is
    because the overlap function is piecewise linear between these points,
    and the maximum of a piecewise linear function on an interval occurs at
    an endpoint.
    
    So we can just check k = 0, 1, ..., N-1.
    """
    # x has length N, values in [0,1]
    # For shift k (in steps), the shifted x is np.roll(x, k)
    # Overlap = sum_i x_i * (1 - x_{(i-k)%N}) * (2/N)
    
    max_overlap = 0.0
    width = 2.0 / N
    
    for k in range(N):
        x_shifted = np.roll(x, k)
        overlap = np.sum(x * (1.0 - x_shifted)) * width
        if overlap > max_overlap:
            max_overlap = overlap
    
    return max_overlap

def objective(x):
    """Objective function: maximize overlap to minimize (we'll negate or use as-is)."""
    return compute_overlap(x, 0)

def soft_max_overlap(x, beta=100.0):
    """
    Smooth approximation of the maximum overlap using log-sum-exp.
    This helps with gradient-based optimization.
    """
    width = 2.0 / N
    overlaps = np.zeros(N)
    for k in range(N):
        x_shifted = np.roll(x, k)
        overlaps[k] = np.sum(x * (1.0 - x_shifted)) * width
    
    # Log-sum-exp for numerical stability
    max_ov = np.max(overlaps)
    if max_ov == 0:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta + max_ov
    return log_sum

def construct():
    start_time = time.time()
    
    # Constraint: sum(x) = N/2 (since integral is 1, width is 2/N, so sum * 2/N = 1 => sum = N/2)
    # And 0 <= x_i <= 1
    
    target_sum = N / 2.0
    
    best_x = np.full(N, 0.5)
    best_score = compute_overlap(best_x, 0)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return TIME_LIMIT - 10.0 - elapsed()  # 10s safety margin
    
    # Try multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.full(N, 0.5))
    
    # 2. Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    alt[1::2] = 0.0
    # Adjust to sum = N/2
    if N % 2 == 0:
        initializations.append(alt.copy())
    else:
        alt[N//2] = 0.5
        initializations.append(alt.copy())
    
    # 3. Block: first half 1, second half 0
    block = np.zeros(N)
    block[:N//2] = 1.0
    if N % 2 != 0:
        block[N//2] = 0.5
    initializations.append(block.copy())
    
    # 4. Random with correct sum
    np.random.seed(42)
    for trial in range(5):
        x_rand = np.random.uniform(0, 1, N)
        # Adjust to sum = N/2
        current_sum = np.sum(x_rand)
        if current_sum > 0:
            x_rand = x_rand * (target_sum / current_sum)
            x_rand = np.clip(x_rand, 0, 1)
            # Re-adjust
            diff = target_sum - np.sum(x_rand)
            if abs(diff) > 1e-10:
                # Distribute the difference
                x_rand += diff / N
                x_rand = np.clip(x_rand, 0, 1)
                # Final adjustment
                diff = target_sum - np.sum(x_rand)
                if abs(diff) > 1e-10:
                    idx = np.random.randint(N)
                    x_rand[idx] += diff
                    x_rand = np.clip(x_rand, 0, 1)
        initializations.append(x_rand.copy())
    
    # 5. Frequency-based initializations
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        if freq < N:
            t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            x_freq = 0.5 + 0.5 * np.sin(t)
            x_freq = np.clip(x_freq, 0, 1)
            # Adjust sum
            current_sum = np.sum(x_freq)
            if current_sum > 0:
                x_freq = x_freq * (target_sum / current_sum)
                x_freq = np.clip(x_freq, 0, 1)
                diff = target_sum - np.sum(x_freq)
                if abs(diff) > 1e-10:
                    x_freq += diff / N
                    x_freq = np.clip(x_freq, 0, 1)
            initializations.append(x_freq.copy())
    
    # Evaluate all initializations
    for x_init in initializations:
        if time_left() < 5.0:
            break
        x_init = np.clip(x_init, 0, 1)
        # Ensure sum is correct
        diff = target_sum - np.sum(x_init)
        if abs(diff) > 1e-10:
            x_init += diff / N
            x_init = np.clip(x_init, 0, 1)
        
        score = compute_overlap(x_init, 0)
        if score < best_score:
            best_score = score
            best_x = x_init.copy()
    
    # Use scipy minimize with constraints
    # We'll use the smooth surrogate for optimization, then evaluate with exact
    
    bounds = [(0, 1)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum},
    ]
    
    # Try L-BFGS-B with soft max objective
    for x_init in initializations:
        if time_left() < 10.0:
            break
        
        x_init = np.clip(x_init, 0, 1)
        diff = target_sum - np.sum(x_init)
        if abs(diff) > 1e-10:
            x_init += diff / N
            x_init = np.clip(x_init, 0, 1)
        
        # Use soft max with increasing beta
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if time_left() < 5.0:
                break
            
            def obj(x):
                return soft_max_overlap(x, beta=beta)
            
            def grad(x):
                # Numerical gradient
                eps = 1e-7
                g = np.zeros(N)
                for i in range(N):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    # Project to satisfy sum constraint approximately
                    # For gradient, we just compute directional derivative
                    f_plus = obj(x_plus)
                    
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    f_minus = obj(x_minus)
                    
                    g[i] = (f_plus - f_minus) / (2 * eps)
                return g
            
            try:
                res = minimize(obj, x_init, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.success or res.fun < obj(x_init):
                    x_opt = np.clip(res.x, 0, 1)
                    # Enforce sum constraint
                    diff = target_sum - np.sum(x_opt)
                    if abs(diff) > 1e-10:
                        x_opt += diff / N
                        x_opt = np.clip(x_opt, 0, 1)
                    
                    score = compute_overlap(x_opt, 0)
                    if score < best_score:
                        best_score = score
                        best_x = x_opt.copy()
            except Exception:
                pass
    
    # Final polish with high-precision PGD-like steps
    if time_left() > 20.0:
        x_polish = best_x.copy()
        lr = 0.01
        for step in range(200):
            if time_left() < 10.0:
                break
            
            # Compute gradient of soft max with high beta
            beta = 500.0
            width = 2.0 / N
            overlaps = np.zeros(N)
            for k in range(N):
                x_shifted = np.roll(x_polish, k)
                overlaps[k] = np.sum(x_polish * (1.0 - x_shifted)) * width
            
            max_ov = np.max(overlaps)
            if max_ov < 1e-10:
                break
            
            exp_vals = np.exp(beta * (overlaps - max_ov))
            weights = exp_vals / np.sum(exp_vals)
            
            # Gradient of sum_k weights_k * overlap_k
            # overlap_k = sum_i x_i * (1 - x_{(i-k)%N}) * width
            # d(overlap_k)/dx_j = width * [(1 - x_{(j-k)%N}) - x_{(j+k)%N}]
            # Wait, let's be more careful.
            # overlap_k = width * sum_i x_i * (1 - x_{(i-k)%N})
            # d(overlap_k)/dx_j = width * [(1 - x_{(j-k)%N}) + sum_{i != j, i-k = j} (-x_i)]
            # The second term: when (i-k)%N = j, i.e., i = (j+k)%N
            # So d(overlap_k)/dx_j = width * [(1 - x_{(j-k)%N}) - x_{(j+k)%N}]
            
            grad = np.zeros(N)
            for k in range(N):
                w = weights[k]
                for j in range(N):
                    term = (1.0 - x_polish[(j - k) % N]) - x_polish[(j + k) % N]
                    grad[j] += w * width * term
            
            # Gradient descent step
            x_new = x_polish - lr * grad
            
            # Project to [0,1] and sum constraint
            x_new = np.clip(x_new, 0, 1)
            diff = target_sum - np.sum(x_new)
            if abs(diff) > 1e-10:
                x_new += diff / N
                x_new = np.clip(x_new, 0, 1)
            
            score_new = compute_overlap(x_new, 0)
            if score_new < best_score:
                best_score = score_new
                best_x = x_new.copy()
                x_polish = x_new.copy()
            else:
                # Reduce learning rate
                lr *= 0.9
    
    # Final enforcement of constraints
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-10:
        best_x += diff / N
        best_x = np.clip(best_x, 0, 1)
    
    # One more check
    final_score = compute_overlap(best_x, 0)
    if final_score < best_score:
        best_score = final_score
    
    return best_x

# EVOLVE-BLOCK-END
