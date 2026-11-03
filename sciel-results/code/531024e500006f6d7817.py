import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def softplus(x, beta=10.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def spectral_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent in the frequency domain.
    We parameterize f by its DFT coefficients.
    We enforce non-negativity by projecting the time-domain signal onto the positive orthant
    (softplus or hard clip) after each frequency update.
    
    Actually, a simpler spectral approach:
    Minimize max(autoconv(f)) subject to sum(f)=1, f>=0.
    In frequency domain, autoconv(f) is IFFT(|FFT(f)|^2).
    The peak is at lag 0 usually, but can be elsewhere.
    
    Let's try a different structural change:
    Direct minimization of the peak autoconvolution using projected gradient descent
    in the time domain, but with a smarter step size and projection onto the simplex.
    """
    f = f_init.copy()
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Projected Gradient Descent on the objective J(f) = max(autoconv(f)) / (sum(f))^2
    # Since we normalize f to sum=1, we minimize max(autoconv(f)) subject to sum(f)=1, f>=0.
    # Gradient of max(autoconv(f)) is tricky if max is not unique.
    # We use a smooth approximation: log-sum-exp of autoconv values.
    
    beta = 10.0
    lr = 0.01
    
    for it in range(500):
        if time.time() - start_time > max_time:
            break
        
        conv = fft_autoconv(f)
        
        # Smooth max approximation
        # To avoid overflow, subtract max
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            break
        z = beta * (conv - max_conv)
        exp_z = np.exp(z)
        log_sum_exp = max_conv + np.log(np.sum(exp_z)) / beta
        
        # Gradient of log_sum_exp wrt f
        # d/d f_i log_sum_exp = (1/beta) * sum_k exp(z_k) * d conv_k / d f_i / sum_j exp(z_j)
        # p_k = exp(z_k) / sum(exp(z))
        p = exp_z / np.sum(exp_z)
        
        # Gradient of conv_k wrt f_i
        # conv_k = sum_j f_j f_{j+k} (for k>=0)
        # d conv_k / d f_i = f_{i+k} + f_{i-k} (with boundary handling)
        # This is equivalent to (f * f)[k] in autoconv sense.
        # grad_i = 2 * sum_k p_k * (f_{i+k} + f_{i-k})/2 ? No, just the derivative.
        
        # Let's compute the gradient efficiently.
        # grad_i = sum_k p_k * (d conv_k / d f_i)
        # d conv_k / d f_i = f_{(i+k) mod N} + f_{(i-k) mod N} ?
        # Wait, conv is linear convolution, not circular.
        # conv_k = sum_{j=0}^{N-1-k} f_j f_{j+k} for k>=0.
        # d conv_k / d f_i = f_{i+k} if i+k < N else 0  +  f_{i-k} if i-k >= 0 else 0.
        
        # We can compute this via correlation.
        # Let's use FFT for the gradient too?
        # grad = 2 * IFFT(FFT(p) * conj(FFT(f))) ?
        # p is defined on k=0..2N-2. f is on 0..N-1.
        # This is getting complicated. Let's stick to time domain gradient if N is small enough, or use a simpler heuristic.
        
        # Since N=600, O(N^2) gradient calculation per step is 360,000 ops.
        # 500 steps * 360,000 = 180,000,000 ops. Should be fast in NumPy.
        
        grad = np.zeros(N)
        for i in range(N):
            # d conv_k / d f_i
            # k ranges 0..2N-2
            # Term 1: f_{i+k} exists if i+k < N. k in [0, N-1-i]
            # Term 2: f_{i-k} exists if i-k >= 0. k in [0, i]
            
            # This loop is slow. Let's vectorize.
            pass
            
        # Vectorized gradient:
        # grad_i = sum_{k=0}^{2N-2} p_k * (f_{i+k} * 1_{i+k<N} + f_{i-k} * 1_{i-k>=0})
        
        # We can compute this using convolution/correlation.
        # Let's create an array G where G[k] = p_k.
        # Then grad = correlation of f and G?
        # conv_k = sum_j f_j f_{j+k}
        # grad_i = sum_k p_k (f_{i+k} + f_{i-k})
        # This is (p * f) + (p * rev(f))?
        
        # Let's just do it with loops for now to ensure correctness, then optimize if needed.
        # Actually, N=600 is small.
        
        # Optimization:
        # grad_i = sum_{k=0}^{N-1-i} p_k f_{i+k} + sum_{k=0}^{i} p_k f_{i-k}
        
        # We can use np.correlate or similar.
        # Let's use a simple loop for the gradient calculation, it's O(N^2) but N=600.
        
        for i in range(N):
            # Part 1: k from 0 to N-1-i
            if N-1-i >= 0:
                grad[i] += np.dot(p[0:N-i], f[i:N])
            # Part 2: k from 0 to i
            if i >= 0:
                grad[i] += np.dot(p[0:i+1], f[i::-1])
                
        # Apply gradient step
        f_new = f - lr * grad
        
        # Project onto simplex {x | x >= 0, sum(x) = 1}
        f_new = project_simplex(f_new, 1.0)
        
        # Check improvement
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            f = f_new
        else:
            # Reduce learning rate
            lr *= 0.5
            
    return best_f

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def exact_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent where each coordinate is optimized exactly
    by minimizing the normalized max autoconvolution over that coordinate.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s  # Normalize
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Precompute indices for efficiency?
    # For each i, the affected lags are m in [i, i+N-1] intersect [0, 2N-2]
    
    for sweep in range(100):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            # Define the objective function for v = f[i]
            # h(v) = max_m (C_m(v)) / (S - old_val + v)^2
            # C_m(v) = C_m_bg + delta_m(v)
            
            # Calculate C_m_bg (autoconvolution with f[i] = 0)
            # We can compute this by taking the current autoconv and subtracting the contribution of f[i]
            conv_current = fft_autoconv(f)
            
            # Contribution of f[i] to conv_current:
            # For each m, if 0 <= m-i < N:
            #   if m-i == i: term is f[i]^2
            #   else: term is 2 * f[i] * f[m-i]
            #   else: 0
            
            # We need to find v > 0 minimizing h(v).
            # Let's use scipy.optimize.minimize_scalar
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                # Compute new conv values for affected lags
                # Affected lags m are those where m-i is in [0, N-1]
                # m ranges from i to i+N-1
                m_start = max(0, i)
                m_end = min(2 * N - 2, i + N - 1)
                
                # Get current conv values for these lags
                conv_vals = conv_current[m_start:m_end + 1].copy()
                
                # Subtract old contribution
                # For m in [m_start, m_end]:
                # j = m - i
                # if j == i: old_contrib = old_val^2
                # else: old_contrib = 2 * old_val * f[j]
                
                js = np.arange(m_start, m_end + 1) - i
                # js are in [0, N-1] by construction of m_start/m_end
                
                old_contribs = np.zeros(len(js))
                mask_diag = (js == i)
                mask_offdiag = ~mask_diag
                old_contribs[mask_diag] = old_val * old_val
                old_contribs[mask_offdiag] = 2 * old_val * f[js[mask_offdiag]]
                
                conv_bg = conv_vals - old_contribs
                
                # Add new contribution
                new_contribs = np.zeros(len(js))
                new_contribs[mask_diag] = v * v
                new_contribs[mask_offdiag] = 2 * v * f[js[mask_offdiag]]
                
                new_conv_vals = conv_bg + new_contribs
                
                max_conv = np.max(new_conv_vals)
                return max_conv / (S_new * S_new)
            
            # We need a good upper bound for v.
            # Since sum is 1, v cannot be larger than 1 practically.
            # But to be safe, let's use a bound based on current max.
            v_upper = max(1.0, 2 * old_val + 0.1)
            
            try:
                res = opt.minimize_scalar(objective, bounds=(0, v_upper), method='bounded')
                v_opt = res.x
            except Exception:
                v_opt = old_val
            
            if v_opt != old_val:
                f[i] = v_opt
                improved = True
        
        # Renormalize to maintain sum=1 for next sweep?
        # The objective function handled normalization internally, so f is not strictly normalized after update.
        # However, the problem is homogeneous, so scaling doesn't change the score.
        # But for numerical stability and consistency, let's renormalize.
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        
        current_val = compute_score(f)
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_f = f.copy()
        
        if not improved:
            break
            
    return best_f

def construct():
    start_time = time.time()
    
    # Generate initial candidates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))

    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply exact coordinate descent to the best initial candidate
    # We can also try it on the second best to see if it escapes local minima
    for f_init, score_init in scored[:3]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        refined = exact_coordinate_descent(f_init, start_time, TIME_LIMIT - 2)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score

    # Try spectral coordinate descent as well
    if time.time() - start_time < TIME_LIMIT - 10:
        spectral_refined = spectral_coordinate_descent(best_f, start_time, TIME_LIMIT - 2)
        spectral_score = compute_score(spectral_refined)
        if spectral_score < best_score:
            best_f = spectral_refined
            best_score = spectral_score

    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
