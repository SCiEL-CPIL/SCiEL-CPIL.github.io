import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140
NFFT = 2048

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    def smooth_max(c, temp):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))

    def smooth_metric(h, temp):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)

    n = np.arange(N)

    # Phase 1: Structured candidates
    candidates = []
    candidates.append(np.ones(N))

    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))

    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)

    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))

    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))

    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))

    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) +
                0.277263158 * np.cos(4*np.pi*n/(N-1)) +
                0.083578947 * np.cos(6*np.pi*n/(N-1)) +
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))

    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)

    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)

    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)

    # Additional diverse candidates
    for freq in [1, 2, 3]:
        for amp in [0.05, 0.1, 0.2]:
            h_test = 1.0 + amp * np.cos(2 * np.pi * freq * n / N)
            h_test = np.clip(h_test, 0, None)
            if h_test.sum() > 0:
                candidates.append(h_test)

    best_val = 1e10
    best_h = np.ones(N)
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Projected gradient descent with log-sum-exp surrogate
    # We optimize in log-space to ensure non-negativity: h = exp(x)
    # The objective is smooth_metric(exp(x), temp)
    # We use adaptive step sizes and multiple temperature annealing steps

    def log_objective(x, temp):
        """Objective in log-space: smooth_metric(exp(x), temp)"""
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp)

    def log_gradient(x, temp):
        """Gradient of smooth_metric(exp(x), temp) w.r.t. x"""
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(N)
        
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        
        # Gradient of smooth_max(c, temp) w.r.t. c
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        exp_c = np.exp(c_shifted)
        w = exp_c / np.sum(exp_c)  # softmax weights
        
        # Gradient of c[k] w.r.t. h[i] for autoconvolution
        # c[k] = sum_i h[i]*h[i+k] (with appropriate indexing)
        # dc[k]/dh[i] = h[i+k] + h[i-k] (with boundary handling)
        # For the full autoconvolution: c[k] = sum_j h[j]*h[j+k]
        # dc[k]/dh[i] = h[i+k] if i+k < N, plus h[i-k] if i-k >= 0
        
        # Gradient of smooth_max w.r.t. h
        # d(smooth_max)/dh[i] = sum_k w[k] * dc[k]/dh[i]
        # dc[k]/dh[i] = h[i+k] (for k >= 0, i+k < N) + h[i-k] (for k > 0, i-k >= 0)
        
        # More carefully: c[k] = sum_j h[j] * h[j+k] for k=0,...,2N-2
        # Actually, the linear autoconvolution: c[k] = sum_j h[j]*h[j+k] where j+k < N
        # So c[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k] for k >= 0
        # And by symmetry c[k] = c[2N-2-k]
        
        # dc[k]/dh[i] = h[i+k] if 0 <= i+k < N, plus h[i-k] if 0 <= i-k < N and k > 0
        # Actually for k >= 0: c[k] = sum_{j: j+k<N} h[j]*h[j+k]
        # dc[k]/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0, from the term where j=i-k, so h[j]*h[j+k]=h[i-k]*h[i])
        
        # Let's compute this via FFT for efficiency
        # The gradient of sum_k w[k]*c[k] w.r.t. h is the cross-correlation of w with h
        # Specifically, d/dh[i] sum_k w[k] c[k] = sum_k w[k] (h[i+k] + h[i-k]) 
        # This is a convolution-like operation
        
        # For k >= 0: c[k] = sum_j h[j]h[j+k], so dc[k]/dh[i] = h[i+k] + h[i-k] (with bounds)
        # Let's use the fact that the gradient is a linear operation on h
        
        # Build the gradient vector: grad_h[i] = sum_k w[k] * (h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
        # This is like a weighted sum of shifted copies of h
        
        # Efficient computation using FFT:
        # Let W be the weights w[k] for k=0,...,2N-2
        # grad_h = ifft(fft(w_padded) * fft(h_padded)) where w_padded is appropriately arranged
        
        # Actually, let's think about it differently.
        # c = h * h (linear convolution)
        # d/dh (w^T c) where w^T c = sum_k w[k] c[k]
        # = sum_k w[k] sum_j h[j]h[j+k]
        # = sum_j h[j] (sum_k w[k] h[j+k])
        # So grad_h[j] = h[j] * (sum_k w[k] h[j+k]) ... no that's not right either
        
        # Let me just do it directly but efficiently
        # grad_h[i] = sum_k w[k] * dc[k]/dh[i]
        # dc[k]/dh[i] for k>=0: h[i+k] (if i+k<N) + h[i-k] (if i-k>=0)
        
        # This is O(N^2) which is too slow for N=600. Let's use FFT.
        
        # Note that c[k] = sum_j h[j] h[j+k]
        # The operation "for each i, compute sum_k w[k] h[i+k]" is a cross-correlation
        # Let's define v[i] = sum_k w[k] h[i+k] for i=0,...,N-1
        # This is like a convolution of w with h
        
        # Use FFT: v = ifft(fft(w_full) * fft(h_full)) where we pad appropriately
        # w_full has length 2N-1, h_full has length N
        # We want v[i] = sum_{k=0}^{2N-2} w[k] h[i+k] for i=0,...,N-1
        # This is a cross-correlation-like operation
        
        # Simpler: just compute it directly since N=600, N^2=360000 which is manageable
        # But we need to do this every gradient step, so let's be careful
        
        # Actually, let's use the matrix-free approach with a few FFTs
        # grad_h[i] = sum_k w[k] (h[i+k] + h[i-k])
        # = sum_k w[k] h[i+k] + sum_k w[k] h[i-k]
        # First term: for i=0..N-1, sum_{k=0}^{2N-2} w[k] h[i+k] where h[j]=0 for j>=N
        # Second term: for i=0..N-1, sum_{k=0}^{2N-2} w[k] h[i-k] where h[j]=0 for j<0
        
        # Let's compute these using FFT
        # Pad h to length NFFT
        h_padded = np.zeros(NFFT)
        h_padded[:N] = h
        
        # For the first term: v1[i] = sum_k w[k] h[i+k]
        # This is a cross-correlation. Let's define w_rev[k] = w[2N-2-k]
        # Then v1[i] = sum_k w[k] h[i+k] = sum_k w[2N-2-(2N-2-k)] h[i+k]
        # = sum_k' w[2N-2-k'] h[i + (2N-2-k')] ... this is getting complicated
        
        # Let's just do O(N^2) since N=600 and we have time budget
        # But actually, let's use a smarter approach
        
        # The gradient of the autoconvolution peak can be computed as follows:
        # d/dh[i] max_k c[k] ≈ sum_k w[k] * (h[i+k] + h[i-k])
        # where w[k] = softmax(c[k]/temp)
        
        # Let's compute this using two convolutions
        # term1[i] = sum_k w[k] h[i+k] = (w * h_rev)[i] type operation
        # Let's just do it with numpy operations
        
        # For efficiency, let's note that w is mostly concentrated near the argmax
        # But for correctness, let's compute the full gradient
        
        # Actually, I realize this O(N^2) gradient computation will be slow for 600 iterations.
        # Let me use a different approach: compute the gradient via FFT
        
        # c = ifft(fft(h)^2) (linear convolution)
        # d/dh[i] (w^T c) = d/dh[i] sum_k w[k] c[k]
        # Since c = h*h (conv), dc/dh[i] is the "cross-correlation" of h with the impulse at i
        # More precisely, if we perturb h[i] by epsilon, then c changes by:
        # delta_c[k] = epsilon * (h[i+k] + h[i-k]) (with bounds)
        # So d/dh[i] (w^T c) = epsilon * sum_k w[k] (h[i+k] + h[i-k])
        # = epsilon * (sum_k w[k] h[i+k] + sum_k w[k] h[i-k])
        
        # Let's compute a[i] = sum_k w[k] h[i+k] and b[i] = sum_k w[k] h[i-k]
        # a[i] is a "forward" correlation, b[i] is a "backward" correlation
        
        # Using FFT:
        # a[i] = sum_{k=0}^{2N-2} w[k] h[i+k] for i=0,...,N-1
        # Let's define W of length 2N-1, H of length N
        # a[i] = sum_k W[k] H[i+k]
        # This is like a cross-correlation. Let's reverse W: W_rev[k] = W[2N-2-k]
        # Then a[i] = sum_k W[2N-2-k] H[i+k] = sum_j W[j] H[i + 2N-2-j]
        # Hmm, this is still tricky.
        
        # Let me just do the O(N^2) computation but vectorized
        # grad_h = np.zeros(N)
        # for i in range(N):
        #     for k in range(2*N-1):
        #         if i+k < N:
        #             grad_h[i] += w[k] * h[i+k]
        #         if i-k >= 0:
        #             grad_h[i] += w[k] * h[i-k]
        
        # Vectorized version:
        # For each i, the valid k values for h[i+k] are k=0 to N-1-i
        # The valid k values for h[i-k] are k=0 to i
        
        # Let's use matrix operations
        # Create a matrix M where M[i,k] = 1 if i+k < N else 0
        # Then a = M @ (w * something)... no
        
        # Actually, let's use the following trick:
        # a[i] = sum_{k=0}^{N-1-i} w[k] h[i+k]
        # This is a sliding window dot product
        # We can compute this using FFT-based convolution
        
        # Let's define w_ext of length NFFT with w_ext[k] = w[k] for k < 2N-1, else 0
        # And h_ext of length NFFT with h_ext[j] = h[j] for j < N, else 0
        # Then a[i] = sum_k w_ext[k] h_ext[i+k]
        # This is the cross-correlation of w_ext and h_ext evaluated at lag i
        # cross_corr[lag] = sum_k w_ext[k] h_ext[k+lag]
        # We want cross_corr[i] for i=0,...,N-1
        
        # Using FFT: cross_corr = ifft(conj(fft(w_ext)) * fft(h_ext))
        # Wait, the cross-correlation is: R_xy[lag] = sum_k x[k] y[k+lag]
        # = ifft(conj(fft(x)) * fft(y))[lag]
        
        w_ext = np.zeros(NFFT)
        w_ext[:2*N-1] = w
        h_ext = np.zeros(NFFT)
        h_ext[:N] = h
        
        W_fft = np.fft.fft(w_ext)
        H_fft = np.fft.fft(h_ext)
        
        # a[i] = sum_k w[k] h[i+k] = cross-correlation of w and h at lag i
        # = ifft(conj(fft(w)) * fft(h))[i]
        a = np.fft.ifft(np.conj(W_fft) * H_fft).real[:N]
        
        # b[i] = sum_k w[k] h[i-k] = cross-correlation of w and h at lag -i
        # = ifft(conj(fft(w)) * fft(h))[-i] = ifft(fft(w) * conj(fft(h)))[i]
        # Actually, b[i] = sum_k w[k] h[i-k] = sum_k w[k] h[-(k-i)]
        # Let's define h_rev[j] = h[N-1-j]
        # b[i] = sum_k w[k] h[i-k]
        # This is the cross-correlation at lag -i, or equivalently,
        # b[i] = ifft(fft(w) * conj(fft(h)))[i] ... let me verify
        
        # Actually, simpler: b[i] = sum_k w[k] h[i-k]
        # Let j = i-k, then k = i-j, so b[i] = sum_j w[i-j] h[j]
        # This is the convolution of w and h evaluated at i
        # b = conv(w, h)[i] = ifft(fft(w) * fft(h))[i]
        
        b = np.fft.ifft(W_fft * H_fft).real[:N]
        
        # But we need to be careful about the range of k
        # w has non-zero values for k=0,...,2N-2
        # For b[i] = sum_k w[k] h[i-k], we need i-k >= 0 and i-k < N, so k <= i and k >= i-N+1
        # The FFT-based computation should handle this correctly with zero-padding
        
        grad_h = a + b
        
        # Now, the full gradient in log-space:
        # d/dx[i] smooth_metric(exp(x), temp)
        # = d/dh[i] (smooth_max(c, temp) / s^2) * dh[i]/dx[i]
        # = (1/s^2) * d(smooth_max)/dh[i] * h[i] - (2/s^3) * smooth_max * s * h[i]
        # Wait, let me be more careful
        # f = smooth_max(c, temp) / s^2
        # df/dh[i] = (1/s^2) * d(smooth_max)/dh[i] - 2*smooth_max/s^3 * ds/dh[i]
        # ds/dh[i] = 1
        # d(smooth_max)/dh[i] = grad_h[i] (computed above)
        # So df/dh[i] = grad_h[i]/s^2 - 2*smooth_max/(s^3)
        # And dx[i]/dh[i] = 1/h[i] (since h[i] = exp(x[i]), dh/dx = h, so dx/dh = 1/h)
        # Thus df/dx[i] = (grad_h[i]/s^2 - 2*smooth_max/s^3) / h[i]
        
        smooth_max_val = smooth_max(c, temp)
        df_dh = grad_h / (s * s) - 2 * smooth_max_val / (s ** 3)
        df_dx = df_dh / h
        
        return df_dx

    # Phase 2: Projected gradient descent with adaptive step sizes
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)
    
    # Anneal temperature from high to low
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for temp in temperatures:
        if elapsed() > DEADLINE - 30:
            break
        
        # Adaptive step size: start with a reasonable step, adjust based on improvement
        step_size = 0.1
        x_curr = x0.copy()
        
        # Run gradient descent for this temperature
        max_iter = 100
        for it in range(max_iter):
            if elapsed() > DEADLINE - 30:
                break
            
            # Compute gradient
            try:
                grad = log_gradient(x_curr, temp)
            except Exception:
                break
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            grad_normed = grad / grad_norm
            
            # Try the step
            x_trial = x_curr - step_size * grad_normed
            x_trial = x_trial - np.max(x_trial)  # re-center
            
            f_trial = log_objective(x_trial, temp)
            f_curr = log_objective(x_curr, temp)
            
            if f_trial < f_curr:
                x_curr = x_trial
                # Maybe increase step size slightly
                step_size = min(step_size * 1.1, 0.5)
            else:
                # Decrease step size
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        
        # Update best
        h_opt = np.exp(x_curr - np.max(x_curr))
        v = fast_metric(h_opt)
        if v < best_val:
            best_val = v
            best_h = h_opt.copy()
            x0 = x_curr - np.max(x_curr)
    
    # Phase 3: Coordinate descent with FFT-based metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)

    no_improve = 0
    max_no_improve = 5000

    while elapsed() < DEADLINE - 30 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]

        mean_h = np.mean(current_h)
        # Use adaptive step sizes based on local scale
        local_scale = max(orig, 0.1 * mean_h, 1e-6)
        step_factors = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
        step_sizes = step_factors * local_scale
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig

        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
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

        if no_improve % 200 == 0:
            # Try multiple perturbation scales
            for scale in [0.01, 0.03, 0.08, 0.15]:
                pert = current_h.copy()
                pert += scale * np.random.randn(N) * np.mean(current_h)
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
                        break

    # Phase 4: Fine polish with tiny paired moves
    if elapsed() < DEADLINE - 5:
        for trial in range(3000):
            if elapsed() > DEADLINE - 3:
                break
            # Use geometrically decreasing step sizes
            step_frac = 0.005 * (0.7 ** (trial // 100))
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
