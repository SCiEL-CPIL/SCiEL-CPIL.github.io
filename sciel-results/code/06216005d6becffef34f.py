import numpy as np
import time
import random

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv(h):
        """Compute full linear autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h."""
        conv = compute_autoconv(h)
        return np.max(conv)
    
    def grad_autoconv_max(h, eps=1e-8):
        """Compute subgradient of max(autoconv) w.r.t. h using finite differences."""
        conv = compute_autoconv(h)
        max_idx = np.argmax(conv)
        
        # Subgradient: the gradient of the max is the gradient of that particular lag
        # conv[k] = sum_i h[i] * h[i+k-N] (for appropriate k)
        # d(conv[k])/d(h[j]) = h[j+k-N] + h[j-k+N] (with appropriate bounds)
        
        grad = np.zeros(N)
        k = max_idx  # lag index in the full convolution
        
        # For lag k in [0, 2N-2], conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[i+k-N+1]
        # Actually, let me think more carefully.
        # h is defined on indices 0..N-1, representing intervals on [-1/4, 1/4]
        # Autoconvolution at lag m: (h*h)[m] = sum_i h[i] * h[i+m]
        # where h is extended by 0 outside [0, N-1]
        # In the FFT computation with size=2N, the convolution is circular but we padded.
        
        # Let me use a simpler approach: finite difference
        # This is expensive but correct. For 600 dimensions, we can do it.
        # Actually, let's use the fact that grad of sum_i h[i]*h[i+k] w.r.t h[j] 
        # = h[j+k] + h[j-k] (with h out of bounds = 0)
        
        # In our FFT setup, conv is computed as ifft(fft(h_pad)^2).real
        # h_pad has h in first N elements, zeros in last N
        # So conv[m] = sum_i h_pad[i] * h_pad[i+m mod 2N]
        # For m in [0, N-1]: conv[m] = sum_{i=0}^{N-1-m} h[i] * h[i+m]
        # For m in [N, 2N-1]: conv[m] = sum_{i=0}^{2N-1-m} h_pad[i] * h_pad[i+m-2N]
        #   = sum_{i=m-N+1}^{N-1} h[i] * h[i+m-2N] = sum_{i=m-N+1}^{N-1} h[i] * h[i-(2N-m)]
        
        # This is getting complex. Let me just use finite differences for the subgradient.
        # With N=600, doing 600 forward differences is feasible if we cache the FFT.
        
        # Actually, let's use a smarter approach: the gradient of the max of a set of linear/quadratic functions
        # is the gradient of the active function.
        
        # For now, use finite difference with a small number of random coordinates to estimate gradient
        # This is a stochastic gradient approach
        
        # Better: use the analytic formula
        # conv[k] for k in [0, 2N-2] from linear convolution
        # Let's recompute conv as linear convolution properly
        
        # Actually, the FFT method with zero-padding gives linear convolution.
        # conv[m] = sum_{i} h[i] * h[i+m] for m >= 0, with h out of [0,N-1] = 0
        # And conv is symmetric: conv[m] = conv[2N-2-m]
        
        # For the subgradient at the max, we need d(conv[k])/d(h[j])
        # conv[k] = sum_i h[i] * h[i+k] (for k >= 0, interpreting h as zero outside [0,N-1])
        # d(conv[k])/d(h[j]) = h[j+k] + h[j-k] (where h out of bounds = 0)
        
        # But in our FFT computation, the indexing might be different.
        # Let me just verify by computing conv directly.
        
        # For simplicity and correctness, use finite differences
        # But that's O(N) FFTs. Let's do it smartly.
        
        # Alternative: use the fact that we can compute the gradient analytically
        # if we track which lag is the max.
        
        # Let me just use a simple finite difference gradient for now
        # Sample a subset of coordinates for speed
        
        n_samples = min(50, N)
        sample_idx = random.sample(range(N), n_samples)
        
        grad_est = np.zeros(N)
        for j in sample_idx:
            h_plus = h.copy()
            h_plus[j] += eps
            h_plus /= np.sum(h_plus)
            
            conv_plus = compute_autoconv(h_plus)
            max_plus = np.max(conv_plus)
            
            grad_est[j] = (max_plus - np.max(conv)) / eps
        
        # Scale up to approximate full gradient
        scale = N / n_samples
        grad_est *= scale
        
        return grad_est
    
    def projected_grad_descent(h0, lr_init=0.01, n_iter=200, tol=1e-8):
        """Projected gradient descent with backtracking line search."""
        h = h0.copy()
        h /= np.sum(h)
        current_val = compute_autoconv_max(h)
        lr = lr_init
        
        for it in range(n_iter):
            if time_left() < 10:
                break
            
            # Compute gradient
            grad = grad_autoconv_max(h)
            
            # Projected gradient step: minimize <grad, h_new> subject to h_new >= 0, sum(h_new) = 1
            # This is a projection onto the simplex
            # Use the standard algorithm
            
            # Step: h_temp = h - lr * grad
            h_temp = h - lr * grad
            
            # Project onto simplex {x >= 0, sum(x) = 1}
            # Algorithm: find tau such that sum(max(h_temp - tau, 0)) = 1
            h_sorted = np.sort(h_temp)[::-1]
            css = np.cumsum(h_sorted)
            rho = np.nonzero(h_sorted * np.arange(1, N + 1) > (css[-1] - 1))[0][-1]
            tau = (css[rho] - 1) / (rho + 1)
            h_new = np.maximum(h_temp - tau, 0)
            
            # Line search: reduce lr if needed
            new_val = compute_autoconv_max(h_new)
            
            # Backtracking
            while new_val > current_val + 1e-12 and lr > 1e-10:
                lr *= 0.5
                h_temp = h - lr * grad
                h_sorted = np.sort(h_temp)[::-1]
                css = np.cumsum(h_sorted)
                rho = np.nonzero(h_sorted * np.arange(1, N + 1) > (css[-1] - 1))[0][-1]
                tau = (css[rho] - 1) / (rho + 1)
                h_new = np.maximum(h_temp - tau, 0)
                new_val = compute_autoconv_max(h_new)
            
            if new_val < current_val:
                h = h_new
                current_val = new_val
                # Increase lr slightly if we made good progress
                if lr < lr_init:
                    lr = min(lr * 1.1, lr_init)
            else:
                lr *= 0.8
            
            if it % 50 == 0 and it > 0:
                pass  # Could add convergence check
        
        return h, current_val
    
    # Generate diverse initializations
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal
    for sigma in [N/12, N/10, N/8, N/6]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # U-shaped (high at edges)
    for a in [0.5, 1.0, 1.5, 2.0]:
        u_shape = 1.0 + a * ((i - center) / (N/2))**2
        inits.append(u_shape)
    
    # W-shape
    for freq in [3, 4, 5]:
        for amp in [0.5, 1.0, 1.5]:
            w_shape = 1.0 + amp * np.cos(2 * np.pi * freq * i / N)
            inits.append(np.maximum(w_shape, 0.01))
    
    # Smooth random
    for _ in range(8):
        freqs = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=4, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=4)
        amps = np.random.uniform(0.3, 1.2, size=4)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    # Triangular
    tri = 1.0 - np.abs(i - center) / (N/2)
    inits.append(np.maximum(tri, 0.01))
    
    # Gaussian
    for sigma in [N/8, N/6, N/5, N/4]:
        gauss = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
        inits.append(gauss)
    
    best_h = None
    best_val = np.inf
    
    # Evaluate initializations and run gradient descent from the best few
    init_results = []
    for init in inits:
        if time_left() < 30:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        init_results.append((val, h.copy()))
    
    # Sort by initial value
    init_results.sort(key=lambda x: x[0])
    
    # Run gradient descent from top 5 initializations
    n_restarts = min(5, len(init_results))
    for val, h in init_results[:n_restarts]:
        if time_left() < 20:
            break
        
        h_opt, val_opt = projected_grad_descent(h, lr_init=0.005, n_iter=150)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Final local refinement: try small perturbations around the best solution
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        
        # Try reducing the peak of the autoconvolution directly
        conv = compute_autoconv(current_h)
        peak_lag = np.argmax(conv)
        
        # The peak is at lag peak_lag. To reduce it, we want to reduce h[i]*h[i+peak_lag] terms.
        # This suggests reducing h at indices that contribute to the peak.
        
        # Compute which indices contribute most to the peak
        if 0 <= peak_lag < N:
            # conv[peak_lag] = sum_i h[i] * h[i+peak_lag]
            contrib = np.zeros(N)
            for i in range(N - peak_lag):
                contrib[i] += current_h[i] * current_h[i + peak_lag]
                contrib[i + peak_lag] += current_h[i] * current_h[i + peak_lag]
        else:
            # peak_lag >= N, use symmetry
            lag = 2*N - 2 - peak_lag
            if 0 <= lag < N:
                contrib = np.zeros(N)
                for i in range(N - lag):
                    contrib[i] += current_h[i] * current_h[i + lag]
                    contrib[i + lag] += current_h[i] * current_h[i + lag]
            else:
                contrib = np.ones(N)
        
        # Reduce the top contributing indices
        top_k = min(20, N)
        top_indices = np.argsort(contrib)[::-1][:top_k]
        
        for idx in top_indices:
            if time_left() < 5:
                break
            
            # Try reducing this index
            for factor in [0.9, 0.8, 0.7, 0.6]:
                new_h = current_h.copy()
                new_h[idx] *= factor
                s = np.sum(new_h)
                if s < 1e-10:
                    continue
                new_h /= s
                
                new_val = compute_autoconv_max(new_h)
                if new_val < current_val:
                    current_h = new_h
                    current_val = new_val
                    break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
