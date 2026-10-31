import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def shape_score(h):
    """Scale-invariant: minimize ||h*h||_inf * ||h||_1 / ||h||_2^2"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l1 * conv_peak / l2_sq

def shape_score_smoothed(h, eps=0.01):
    """Smoothed version for gradient computation."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv = conv[:2*N-1]
    # Smooth max using softplus approximation
    conv_smooth = np.logsumexp(conv / eps, axis=0) * eps
    if conv_smooth < 1e-15:
        return 0.0
    return l1 * conv_smooth / l2_sq

def gradient_shape_score(h, eps=0.01):
    """Compute gradient of the smoothed shape score."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return np.zeros_like(h)
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return np.zeros_like(h)
    
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv = conv[:2*N-1]
    
    # Softplus smoothing
    conv_scaled = conv / eps
    max_conv = np.max(conv_scaled)
    exp_conv = np.exp(conv_scaled - max_conv)
    sum_exp = np.sum(exp_conv)
    
    # Gradient of softplus max
    weights = exp_conv / sum_exp  # These are the softmax weights
    conv_smooth = eps * (max_conv + np.log(sum_exp))
    
    if conv_smooth < 1e-15:
        return np.zeros_like(h)
    
    # The objective is f = l1 * conv_smooth / l2_sq
    # We need df/dh
    
    # Derivative of conv_smooth w.r.t. conv_k: eps * weights[k]
    # Derivative of conv_k w.r.t. h: from circular convolution
    
    # conv[k] = sum_i h[i] * h[(k-i) mod (2N-1)] approximately
    # Actually for linear convolution: conv[k] = sum_i h[i] * h[k-i] where valid
    
    # Let's compute gradient numerically for reliability
    grad = np.zeros(N)
    delta = 1e-7
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += delta
        h_minus = h.copy()
        h_minus[i] -= delta
        h_minus = np.maximum(h_minus, 0)
        
        fp = shape_score_smoothed(h_plus, eps)
        fm = shape_score_smoothed(h_minus, eps)
        grad[i] = (fp - fm) / (2 * delta)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)
    
    def try_candidate(h):
        nonlocal best_h, best_shape
        if time.time() > deadline:
            return
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            return
        h = h / s
        sc = shape_score(h)
        if sc < best_shape:
            best_shape = sc
            best_h = h.copy()
    
    # Phase 1: Generate diverse initial candidates using structured patterns
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Periodic combs
    for period in range(2, 16):
        for offset in range(period):
            for width in range(1, min(period, 4)):
                h = np.zeros(N)
                for i in range(N):
                    if (i - offset) % period < width:
                        h[i] = 1.0
                if np.sum(h) > 0:
                    h /= np.sum(h)
                    candidates.append(h.copy())
    
    # Gaussian profiles
    for center in range(0, N, 2):
        for sigma in [1.0, 2.0, 3.0, 5.0, 8.0]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h += 1e-15
            h /= np.sum(h)
            candidates.append(h.copy())
    
    # Block structures
    for start in range(0, N, 2):
        for length in range(1, min(N - start, 20), 3):
            h = np.zeros(N)
            end = min(start + length, N)
            if end > start:
                h[start:end] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h.copy())
    
    # Two peaks
    for c1 in range(0, N, 4):
        for c2 in range(c1 + 4, N, 4):
            for s1 in [2.0, 4.0, 6.0]:
                for s2 in [2.0, 4.0, 6.0]:
                    x = np.arange(N)
                    h = np.exp(-0.5 * ((x - c1) / s1) ** 2) + np.exp(-0.5 * ((x - c2) / s2) ** 2)
                    h += 1e-15
                    h /= np.sum(h)
                    candidates.append(h.copy())
    
    # Alternating weighted
    for period in range(2, 10):
        for w in [0.2, 0.4, 0.6, 0.8]:
            h = np.zeros(N)
            for i in range(N):
                if i % period == 0:
                    h[i] = w
                else:
                    h[i] = 1.0 - w
            h = np.maximum(h, 0)
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h.copy())
    
    # Mixtures of comb and Gaussian
    for period in [3, 4, 5, 6, 7, 8, 10]:
        for offset in range(period):
            comb = np.zeros(N)
            for i in range(N):
                if (i - offset) % period < 1:
                    comb[i] = 1.0
            if np.sum(comb) > 0:
                comb /= np.sum(comb)
            for center in [N//4, N//2, 3*N//4]:
                for sigma in [3.0, 5.0, 8.0]:
                    x = np.arange(N)
                    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
                    gauss += 1e-15
                    gauss /= np.sum(gauss)
                    for alpha in [0.3, 0.5, 0.7]:
                        h = alpha * comb + (1 - alpha) * gauss
                        if np.sum(h) > 0:
                            h /= np.sum(h)
                            candidates.append(h.copy())
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        try_candidate(h)
    
    # Phase 2: Gradient-based optimization from multiple starting points
    # Use L-BFGS-B with log-transformed variables to enforce nonnegativity
    
    def log_transformed_objective(log_h):
        """Objective function in log-space: h = exp(log_h), normalized."""
        h = np.exp(log_h)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        return shape_score_smoothed(h, eps=0.05)
    
    def log_transformed_gradient(log_h):
        """Gradient in log-space."""
        h = np.exp(log_h)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros_like(log_h)
        h = h / s
        
        # Numerical gradient
        grad_log = np.zeros(N)
        delta = 1e-6
        base = shape_score_smoothed(h, eps=0.05)
        for i in range(N):
            log_h_plus = log_h.copy()
            log_h_plus[i] += delta
            h_plus = np.exp(log_h_plus)
            s_plus = np.sum(h_plus)
            if s_plus > 1e-15:
                h_plus = h_plus / s_plus
                fp = shape_score_smoothed(h_plus, eps=0.05)
            else:
                fp = 1e10
            grad_log[i] = (fp - base) / delta
        
        return grad_log
    
    # Select top-K candidates for refinement
    scored_candidates = [(shape_score(h), h) for h in candidates if time.time() < deadline]
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 20 candidates for gradient optimization
    top_k = min(20, len(scored_candidates))
    
    for k in range(top_k):
        if time.time() > deadline:
            break
        
        h0 = scored_candidates[k][1].copy()
        # Ensure all values are positive for log transform
        h0 = np.maximum(h0, 1e-10)
        h0 = h0 / np.sum(h0)
        
        log_h0 = np.log(h0)
        
        try:
            result = opt.minimize(
                log_transformed_objective,
                log_h0,
                method='L-BFGS-B',
                jac=log_transformed_gradient,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < 1e10:
                h_opt = np.exp(result.x)
                s = np.sum(h_opt)
                if s > 1e-15:
                    h_opt = h_opt / s
                    try_candidate(h_opt)
        except Exception:
            pass
    
    # Phase 3: Local coordinate descent refinement
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)
        
        # Multiple rounds of coordinate descent with decreasing step sizes
        for round_num in range(5):
            if time.time() > deadline:
                break
            
            step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01]
            
            for psize in step_sizes:
                if time.time() > deadline:
                    break
                
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    
                    for i in range(N):
                        if time.time() > deadline:
                            break
                        for j in range(N):
                            if i == j:
                                continue
                            for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                delta = h[i] * psize * frac
                                if delta < 1e-12 or h[i] - delta < -1e-12:
                                    continue
                                h_new = h.copy()
                                h_new[i] = max(0, h_new[i] - delta)
                                h_new[j] += delta
                                sc = shape_score(h_new)
                                if sc < current_score:
                                    h = h_new
                                    current_score = sc
                                    improved = True
                                    if current_score < best_shape:
                                        best_shape = current_score
                                        best_h = h.copy()
                                    break
    
    # Phase 4: Final polish with random perturbations
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)
        
        for _ in range(300):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            delta = h[i] * 0.005 * np.random.uniform(0.1, 1.0)
            if delta < 1e-12 or h[i] - delta < -1e-12:
                continue
            
            h_new = h.copy()
            h_new[i] = max(0, h_new[i] - delta)
            h_new[j] += delta
            
            sc = shape_score(h_new)
            if sc < current_score:
                h = h_new
                current_score = sc
                if current_score < best_shape:
                    best_shape = current_score
                    best_h = h.copy()
    
    # Return the normalized best shape
    final_h = best_h.copy()
    final_h = np.maximum(final_h, 0)
    s = np.sum(final_h)
    if s > 0:
        final_h /= s
    
    return final_h.tolist()
