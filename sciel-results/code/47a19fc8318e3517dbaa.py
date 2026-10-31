import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 142  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max autoconvolution using FFT."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def metric(h):
        """Exact normalized max autoconvolution."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        c = np.convolve(h, h, mode='full')
        return np.max(c) / (s * s)
    
    # Log-sum-exp smooth approximation of max
    def smooth_max(c, temp=0.01):
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
    
    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)
    
    # Phase 1: Evaluate structured candidates
    i = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    candidates.append(0.5 * (1 + np.cos(np.pi * i / (N - 1))))
    
    # Gaussian
    xg = (i - N/2) / (N * 0.3)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau: high at edges, low in center
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse plateau
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Binary-like: alternating high/low to reduce contiguous blocks
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        binary = np.zeros(N)
        for j in range(N):
            if (j % period) < period // 2:
                binary[j] = 1.0
        candidates.append(binary)
    
    # Sparse random patterns
    np.random.seed(42)
    for density in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sparse = (np.random.rand(N) < density).astype(float)
        candidates.append(sparse)
    
    # Low-frequency smooth patterns
    for freq in [1, 2, 3, 4, 5]:
        smooth_pat = 0.5 + 0.5 * np.cos(2 * np.pi * freq * i / N)
        candidates.append(smooth_pat)
    
    # Multi-scale: combine different frequencies
    for f1, f2 in [(1, 3), (1, 5), (2, 5), (1, 7), (2, 7)]:
        multi = 0.5 + 0.3 * np.cos(2 * np.pi * f1 * i / N) + 0.2 * np.cos(2 * np.pi * f2 * i / N)
        candidates.append(np.clip(multi, 0, None))
    
    best_val = 1e10
    best_h = np.ones(N)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization with smooth max (log-sum-exp surrogate)
    # This directly optimizes the continuous landscape via projected gradient on log-heights
    def objective(x):
        if elapsed() > DEADLINE - 20:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=0.02)
    
    def objective_grad(x):
        """Compute gradient of smooth metric w.r.t. log-heights."""
        if elapsed() > DEADLINE - 20:
            return 1e10, np.zeros(N)
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        
        temp = 0.02
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        
        # Soft-max gradient: softmax weights
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        exp_c = np.exp(c_shifted)
        w = exp_c / np.sum(exp_c)  # softmax weights over convolution
        
        # Gradient of max(c)/s^2 w.r.t. h: (w * dh/dh_i) / s^2 - 2*max_c*h_i/s^3
        # dc/dh_i = h * e_i convolved, i.e., c[k] = sum_j h[j]h[j+k]
        # dc/dh_i = sum_k w[k] * (h[i-k] + h[i+k]) (with bounds)
        # Use FFT for efficiency: gradient of c w.r.t. h is circular convolution of h with w
        # Actually dc/dh_i = (w * h)_i in circular sense for the FFT-based version
        
        # For the smooth max: d(smooth_max)/dh_i = sum_k w[k] * dc[k]/dh_i
        # dc[k]/dh_i = h[i-k] + h[i+k] (circular, for the linear part we approximate)
        
        # Use FFT: gradient = ifft(fft(w) * fft(h)) approximately
        W = np.fft.rfft(w, n=nfft)
        grad_h = np.fft.irfft(W * H, n=nfft)[:N]
        
        # Scale by 1/s^2 and subtract the normalization term
        smax = smooth_max(c, temp)
        grad_h = grad_h / (s * s) - 2 * smax * h / (s * s * s)
        
        # Chain rule: x = log(h), so h = exp(x_shifted)
        # d/dx_i = d/dh_i * h_i
        grad_x = grad_h * h
        # Subtract mean to maintain the shift invariance
        grad_x = grad_x - np.mean(grad_x)
        
        return smax / (s * s), grad_x
    
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 800}
        )
        if result.success or result.fun < best_val:
            h_opt = np.exp(result.x - np.max(result.x))
            v = fast_metric(h_opt)
            if v < best_val:
                best_val = v
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Phase 2b: Projected gradient descent with adaptive step sizes on log-heights
    # using the log-sum-exp surrogate for the max
    if elapsed() < DEADLINE - 40:
        x = np.log(best_h + 1e-15)
        x = x - np.max(x)
        current_x_val = objective(x)
        
        step_size = 0.1
        no_improve_pg = 0
        max_no_improve_pg = 200
        
        for pg_iter in range(500):
            if elapsed() > DEADLINE - 30 or no_improve_pg > max_no_improve_pg:
                break
            
            val, grad = objective_grad(x)
            if val == 1e10:
                break
            
            # Adaptive step size: backtracking line search
            alpha = step_size
            new_val = val
            new_x = x
            for ls_iter in range(10):
                new_x = x - alpha * grad
                # Project: maintain nonnegativity via softplus or just clip log
                new_x = new_x - np.max(new_x)  # shift invariance
                new_h = np.exp(new_x)
                if np.sum(new_h) < 1e-15:
                    alpha *= 0.5
                    continue
                new_val = smooth_metric(new_h, temp=0.02)
                if new_val < val:
                    break
                alpha *= 0.5
            
            if new_val < val - 1e-10:
                x = new_x
                val = new_val
                step_size = min(step_size * 1.1, 1.0)
                no_improve_pg = 0
            else:
                step_size *= 0.7
                no_improve_pg += 1
            
            # Convert to h and check
            h_pg = np.exp(x - np.max(x))
            v_pg = fast_metric(h_pg)
            if v_pg < best_val:
                best_val = v_pg
                best_h = h_pg.copy()
    
    # Phase 3: Cosine-series parameterized optimization (Remez-like)
    K = 15
    
    def fit_cosine(h, K_order):
        i_arr = np.arange(N)
        A = np.zeros((N, K_order + 1))
        A[:, 0] = 1.0
        for k in range(1, K_order + 1):
            A[:, k] = np.cos(2 * np.pi * k * i_arr / N)
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs
    
    def cheb_to_h(coeffs):
        i_arr = np.arange(N)
        h = np.full(N, coeffs[0])
        for k in range(1, len(coeffs)):
            h += coeffs[k] * np.cos(2 * np.pi * k * i_arr / N)
        return np.maximum(h, 0.0)
    
    coeffs = fit_cosine(best_h, K)
    
    for remez_iter in range(40):
        if elapsed() > DEADLINE - 25:
            break
        
        h = cheb_to_h(coeffs)
        if np.sum(h) < 1e-15:
            break
        
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        
        # Find max convolution location
        nfft = 1
        while nfft < 2 * N - 1:
            nfft *= 2
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        
        max_idx = np.argmax(c)
        max_val_c = c[max_idx]
        
        # Compute gradient w.r.t. cosine coefficients
        k = max_idx
        h_curr = h
        s = np.sum(h_curr)
        
        grad = np.zeros(K + 1)
        for j in range(K + 1):
            g = 0.0
            for idx in range(N):
                if k == 0:
                    g += 2 * np.cos(2*np.pi*j*idx/N) * h_curr[idx]
                else:
                    if idx + k < N:
                        g += np.cos(2*np.pi*j*idx/N) * h_curr[idx+k] + h_curr[idx] * np.cos(2*np.pi*j*(idx+k)/N)
                    if idx - k >= 0:
                        g += np.cos(2*np.pi*j*idx/N) * h_curr[idx-k] + h_curr[idx] * np.cos(2*np.pi*j*(idx-k)/N)
            grad[j] = g / (s * s)
        
        # Gradient step
        lr = 0.05 * max_val_c / (np.max(np.abs(grad)) + 1e-15)
        lr = min(lr, 0.5)
        
        new_coeffs = coeffs - lr * grad
        new_h = cheb_to_h(new_coeffs)
        
        if np.sum(new_h) > 1e-15:
            new_val = fast_metric(new_h)
            if new_val < best_val:
                best_val = new_val
                best_h = new_h.copy()
                coeffs = new_coeffs
            else:
                new_coeffs = coeffs - lr * 0.3 * grad
                new_h = cheb_to_h(new_coeffs)
                if np.sum(new_h) > 1e-15:
                    new_val = fast_metric(new_h)
                    if new_val < best_val:
                        best_val = new_val
                        best_h = new_h.copy()
                        coeffs = new_coeffs
    
    # Phase 4: Coordinate descent with FFT metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    no_improve = 0
    max_no_improve = 3000
    
    while elapsed() < DEADLINE - 20 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
        step_sizes = step_factors * mean_h
        candidates_val = list(orig + step_sizes) + list(orig - step_sizes) + [0.0, orig]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-10:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1
        
        # Periodic perturbation
        if no_improve % 300 == 0 and no_improve > 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 5: Fine-tune with small paired moves
    if elapsed() < DEADLINE - 8:
        step_frac = 0.02
        for _ in range(1000):
            if elapsed() > DEADLINE - 5:
                break
            i_idx, j_idx = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i_idx] += delta
            new_h[j_idx] -= delta
            if new_h[i_idx] >= 0 and new_h[j_idx] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    # Final exact evaluation
    final_val = metric(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
