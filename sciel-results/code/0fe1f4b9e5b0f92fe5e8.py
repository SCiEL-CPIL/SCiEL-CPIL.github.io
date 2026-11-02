import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s

K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    score = max_conv / (integral_f ** 2)
    return score

def _autoconv_and_grad(heights):
    """Compute autoconvolution max and gradient w.r.t. heights.
    
    Returns (max_conv, grad, integral_f, argmax_idx)
    """
    heights = np.maximum(heights, 0.0)
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Find the max convolution value
    max_idx = np.argmax(conv)
    max_conv = conv[max_idx]
    
    # The gradient of max_conv w.r.t. f_fine[j] is sum_k f_fine[k] * delta(conv_idx - (j-k))
    # conv[t] = sum_j f_fine[j] * f_fine[j + t - M_FFT/2] (approximately, for circular conv)
    # Actually for linear convolution via zero-padding:
    # conv[t] = sum_j f_fine[j] * f_fine[t - j] where t is in [0, M_FFT)
    # So d conv[t] / d f_fine[m] = sum_j [j == m] f_fine[t-j] + f_fine[j] [t-j == m]
    # = f_fine[t-m] + f_fine[t-m] = 2 * f_fine[t-m] if t-m is in range, but more carefully:
    # conv[t] = sum_j f_fine[j] f_fine[t-j]
    # d/d f_fine[m] conv[t] = f_fine[t-m] + f_fine[t-m] = 2 f_fine[t-m] (when 0 <= t-m < M_FFT)
    # But we need to be careful about the actual convolution definition.
    
    # For the gradient w.r.t. heights[i], we need to sum over all fine-grid positions m 
    # that correspond to interval i (i.e., m in [i*K, (i+1)*K))
    # d max_conv / d heights[i] = sum_{m in [i*K, (i+1)*K)} d max_conv / d f_fine[m]
    # = sum_{m in [i*K, (i+1)*K)} 2 * f_fine[max_idx - m] (if 0 <= max_idx - m < M_FFT)
    
    # More efficiently: d conv[t] / d f_fine[m] = f_fine[t-m] + f_fine[t-m] if we consider
    # the product rule. Actually:
    # conv[t] = sum_j f[j] f[t-j]
    # d conv[t] / d f[m] = f[t-m] (from first factor) + f[t-m] (from second factor) = 2*f[t-m]
    # when t-m is a valid index.
    
    # So the gradient contribution from interval i is:
    # grad_i = 2 * sum_{m=i*K}^{(i+1)*K-1} f_fine[max_idx - m] (where valid)
    
    # Let's compute this efficiently
    # For each m in [i*K, (i+1)*K), the contribution is f_fine[max_idx - m] if 0 <= max_idx - m < M_FFT
    
    # We can compute this as: for each i, sum f_fine[max_idx - m] for m in [i*K, (i+1)*K)
    # Let u = max_idx - m, so m = max_idx - u
    # m in [i*K, (i+1)*K) => max_idx - u in [i*K, (i+1)*K) => max_idx - (i+1)*K < u <= max_idx - i*K
    # And we need 0 <= u < M_FFT
    
    # This is a bit complex. Let's just do it directly but vectorized.
    
    # Create an array of all fine grid indices
    # For each interval i, the fine indices are i*K to (i+1)*K-1
    # grad_fine[m] = 2 * f_fine[max_idx - m] if 0 <= max_idx - m < M_FFT, else 0
    
    # Actually, let's think about this differently. The convolution is:
    # (f * f)[t] = sum_j f[j] f[t-j]
    # The gradient of (f*f)[t] w.r.t. f is 2 * f (shifted), i.e., (f*f)'[t] = 2 * f[t] * ... no.
    # d/d f[m] (f*f)[t] = f[t-m] + f[t-m] = 2 f[t-m]
    
    # So grad_fine[m] = 2 * f_fine[max_idx - m] for valid m
    # Let's compute this
    
    # max_idx - m ranges from max_idx - 0 to max_idx - (M_FFT-1)
    # We need 0 <= max_idx - m < M_FFT, i.e., max_idx - M_FFT < m <= max_idx
    # So m in [max(0, max_idx - M_FFT + 1), min(M_FFT-1, max_idx)]
    
    m_lo = max(0, max_idx - M_FFT + 1)
    m_hi = min(M_FFT - 1, max_idx)
    
    grad_fine = np.zeros(M_FFT, dtype=np.float64)
    if m_lo <= m_hi:
        # For m in [m_lo, m_hi], grad_fine[m] = 2 * f_fine[max_idx - m]
        # max_idx - m ranges from max_idx - m_hi to max_idx - m_lo
        u_lo = max_idx - m_hi
        u_hi = max_idx - m_lo
        grad_fine[m_lo:m_hi+1] = 2.0 * f_fine[u_lo:u_hi+1][::-1]
    
    # Now aggregate to interval level
    # grad[i] = sum of grad_fine[m] for m in [i*K, (i+1)*K)
    # Reshape and sum
    grad_fine_resized = grad_fine[:M_STEP].reshape(N, K)
    grad = grad_fine_resized.sum(axis=1)
    
    integral_f = np.sum(heights) * w
    
    return max_conv, grad, integral_f, max_idx

def _smooth_max_and_grad(values, temperature):
    """Compute smooth max and its gradient using log-sum-exp.
    
    Returns (smooth_max, grad) where grad has same shape as values.
    """
    # For numerical stability
    max_val = np.max(values)
    if temperature <= 1e-10:
        idx = np.argmax(values)
        grad = np.zeros_like(values)
        grad[idx] = 1.0
        return max_val, grad
    
    exp_vals = np.exp((values - max_val) / temperature)
    sum_exp = np.sum(exp_vals)
    smooth_max = max_val + temperature * np.log(sum_exp)
    grad = exp_vals / sum_exp
    return smooth_max, grad

def _pulse_objective(params, n_pulses):
    """Objective for pulse-based representation."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10
    
    heights = np.zeros(N, dtype=np.float64)
    for i in range(n_pulses):
        amp = params[3*i]
        pos = params[3*i + 1]  # center in [0,1]
        width = params[3*i + 2]  # width in [0,1]
        
        if amp < 0 or width <= 0:
            continue
        
        center_idx = pos * N
        half_width_idx = width * N / 2.0
        
        start_idx = int(np.floor(center_idx - half_width_idx))
        end_idx = int(np.ceil(center_idx + half_width_idx))
        
        start_idx = max(0, start_idx)
        end_idx = min(N, end_idx)
        
        if start_idx < end_idx:
            heights[start_idx:end_idx] += amp
    
    if np.sum(heights) <= 0:
        return 1e10
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * (0.5 / N)
    
    if integral_f <= 1e-10:
        return 1e10
    
    return max_conv / (integral_f ** 2)

def _gradient_descent_step(heights, lr, temperature):
    """One gradient descent step with L-BFGS-like line search."""
    max_conv, grad, integral_f, _ = _autoconv_and_grad(heights)
    
    # Smooth max gradient
    # We need the gradient of the smooth max of conv, not just the hard max
    # For now, use the hard max gradient as an approximation
    # Actually, let's compute the smooth max properly
    
    # Reconstruct conv for smooth max
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    smooth_max, smooth_grad_conv = _smooth_max_and_grad(conv, temperature)
    
    # Gradient of smooth_max w.r.t. f_fine
    # d smooth_max / d conv[t] = smooth_grad_conv[t]
    # d conv[t] / d f_fine[m] = 2 * f_fine[t-m] (as before)
    # So d smooth_max / d f_fine[m] = sum_t smooth_grad_conv[t] * 2 * f_fine[t-m]
    # This is a convolution of smooth_grad_conv with f_fine, times 2
    
    # d smooth_max / d f_fine = 2 * (smooth_grad_conv * f_fine) (circular convolution)
    G = np.fft.rfft(smooth_grad_conv)
    Ff = np.fft.rfft(f_fine)
    grad_fine_fft = 2.0 * np.fft.irfft(G * Ff, n=M_FFT)
    
    # Aggregate to interval level
    grad_fine_resized = grad_fine_fft[:M_STEP].reshape(N, K)
    grad_intervals = grad_fine_resized.sum(axis=1)
    
    # Objective: smooth_max / integral_f^2
    # d/d h [smooth_max / integral_f^2] = (grad_smooth_max * integral_f^2 - smooth_max * 2 * integral_f * d(integral_f)/dh) / integral_f^4
    # = (grad_smooth_max * integral_f - 2 * smooth_max * d(integral_f)/dh) / integral_f^3
    
    w = 0.5 / N
    d_integral = w  # d integral_f / d heights[i] = w for all i
    
    if integral_f <= 1e-10:
        return heights, 1e10
    
    grad_obj = (grad_intervals * integral_f - 2.0 * smooth_max * d_integral) / (integral_f ** 3)
    
    # Gradient descent step with simple line search
    h_trial = heights - lr * grad_obj
    h_trial = np.maximum(h_trial, 0.0)
    
    # Line search: try a few step sizes
    best_h = h_trial
    best_score = compute_score(h_trial)
    current_score = compute_score(heights)
    
    for factor in [0.5, 0.25, 0.125]:
        h_try = heights - lr * factor * grad_obj
        h_try = np.maximum(h_try, 0.0)
        s_try = compute_score(h_try)
        if s_try < best_score:
            best_score = s_try
            best_h = h_try
    
    return best_h, best_score

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Phase 1: Pulse-based optimization to get a good starting point
    from scipy.optimize import minimize
    rng = np.random.default_rng(42)
    
    for n_pulses in [2, 3, 4, 5]:
        if time.time() - START_TIME > TIME_LIMIT - 40:
            break
        
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - START_TIME > TIME_LIMIT - 30:
                break
            
            params = np.zeros(3 * n_pulses)
            for i in range(n_pulses):
                params[3*i] = rng.uniform(0.5, 2.0)
                params[3*i + 1] = rng.uniform(0.1, 0.9)
                params[3*i + 2] = rng.uniform(0.05, 0.4)
            
            bounds = []
            for i in range(n_pulses):
                bounds.append((0.0, 5.0))
                bounds.append((0.0, 1.0))
                bounds.append((0.01, 0.8))
            
            try:
                result = minimize(
                    _pulse_objective,
                    params,
                    args=(n_pulses,),
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
                )
                
                if result.fun < best_score:
                    final_params = result.x
                    heights = np.zeros(N, dtype=np.float64)
                    for i in range(n_pulses):
                        amp = final_params[3*i]
                        pos = final_params[3*i + 1]
                        width = final_params[3*i + 2]
                        
                        if amp < 0 or width <= 0:
                            continue
                        
                        center_idx = pos * N
                        half_width_idx = width * N / 2.0
                        start_idx = max(0, int(np.floor(center_idx - half_width_idx)))
                        end_idx = min(N, int(np.ceil(center_idx + half_width_idx)))
                        
                        if start_idx < end_idx:
                            heights[start_idx:end_idx] += amp
                    
                    best_score = result.fun
                    best_h = heights.copy()
            except Exception:
                pass
    
    # Phase 2: Gradient-based refinement with smooth max
    if time.time() - START_TIME < TIME_LIMIT - 10:
        h_work = best_h.copy()
        lr = 0.01
        temperature = 0.1
        
        for iteration in range(200):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            h_work, score = _gradient_descent_step(h_work, lr, temperature)
            
            if score < best_score:
                best_score = score
                best_h = h_work.copy()
            
            # Anneal temperature
            temperature *= 0.95
            lr *= 0.99
    
    # Phase 3: Final local search
    if time.time() - START_TIME < TIME_LIMIT - 3:
        rng2 = np.random.default_rng(123)
        for _ in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            perturbation = rng2.normal(0, 0.01, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
