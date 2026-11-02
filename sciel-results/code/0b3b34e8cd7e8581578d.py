"""Stronger algorithm for First autocorrelation inequality step function."""
import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def autoconv(h):
        """Compute discrete autoconvolution using FFT.
        Returns array of length 2*N-1 where index i corresponds to lag i-(N-1).
        ac[N-1] is lag 0 (sum of squares), ac[N-1+1] is lag 1, etc.
        """
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        H = np.fft.rfft(h, n=n_fft)
        power = H * np.conj(H)
        ac_full = np.fft.irfft(power, n=n_fft)
        # ac_full has length n_fft
        # The circular convolution result: the linear convolution of h with h (length 2*N-1)
        # is embedded in ac_full. For non-negative h, the linear convolution is:
        # lin_conv[k] = sum_i h[i] * h[i+k] for k = -(N-1) to N-1
        # Using the standard FFT convolution: 
        # np.convolve(h, h, 'full') gives length 2*N-1 where index 0 = lag -(N-1)
        # The FFT approach gives circular conv. For n_fft >= 2*N-1, no wrap-around.
        # ac_full[0] = lag 0 in circular, but for linear conv we need:
        # lin[k] for k=0..2N-2 where lin[k] = sum_i h[i]*h[i+k-(N-1)]
        # Actually, np.convolve(h, h, 'full')[k] = sum_i h[i] h[k-i] for valid i
        # This equals sum_j h[j] h[k-j], which is the correlation at lag k-(N-1) if we define
        # lag l = k-(N-1), then k = l+N-1
        # So lin_conv[k] for k=0..2N-2 gives correlation at lag k-(N-1)
        # In FFT: the result at index m of the full linear conv corresponds to...
        # Let me just use np.convolve for clarity and correctness, it's fast enough for N=600
        pass
    
    def compute_ac(h):
        """Compute autoconvolution coefficients. Returns array of length 2*N-1.
        ac[i] corresponds to lag i - (N-1), i.e., ac[N-1] = sum h^2 (lag 0).
        """
        # Using direct FFT-based linear convolution
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        H = np.fft.rfft(h, n=n_fft)
        power = H * np.conj(H)
        ac_circ = np.fft.irfft(power, n=n_fft)
        # For linear convolution with n_fft >= 2N-1, the first 2N-1 elements of ac_circ
        # contain the linear convolution, but we need to handle the wrap-around correctly.
        # Actually for circular conv of length n_fft with zero-padding, the linear conv
        # is in ac_circ[0:2N-1] directly if n_fft >= 2N-1.
        # Wait, that's not quite right. Let me verify:
        # Circular conv of h_padded (length n_fft) with itself: result[m] = sum_i h_padded[i]*h_padded[(m-i)%n_fft]
        # Since h_padded is zero outside [0,N-1], for m in [0, 2N-2], (m-i)%n_fft = m-i for 
        # i in [0,N-1] and m-i >= 0, and no wrap-around since m <= 2N-2 < n_fft.
        # So ac_circ[0:2N-1] = linear conv.
        # But linear conv[k] = sum_i h[i]*h[k-i] which is the cross-correlation at lag k-(N-1).
        # So ac[i] for i=0..2N-2 gives correlation at lag i-(N-1).
        return ac_circ[:2*N-1]
    
    def score(h):
        """Normalized max of autoconvolution. h is normalized so sum(h) = 1."""
        ac = compute_ac(h)
        s = np.sum(h)
        if s < 1e-15:
            return np.inf
        return np.max(ac) / (s * s)
    
    def smooth_score(h, beta):
        """Smooth max using log-sum-exp."""
        ac = compute_ac(h)
        s = np.sum(h)
        if s < 1e-15:
            return np.inf
        max_ac = np.max(ac)
        if max_ac < 0:
            max_ac = 0
        log_sum_exp = max_ac + np.log(np.sum(np.exp((ac - max_ac) * beta)) / beta)
        return log_sum_exp / (s * s)
    
    def grad_smooth(h, beta):
        """Gradient of smooth score w.r.t h."""
        ac = compute_ac(h)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(N), np.inf
        
        max_ac = np.max(ac)
        exp_term = np.exp((ac - max_ac) * beta)
        sum_exp = np.sum(exp_term)
        if sum_exp < 1e-300:
            weights = np.ones(len(ac)) / len(ac)
        else:
            weights = exp_term / sum_exp
        
        # Gradient of sum_k weights[k] * ac[k] w.r.t h[j]
        # ac[k] = sum_i h[i] * h[i + (k-(N-1))]  (with bounds)
        # d(ac[k])/d(h[j]) = sum_i delta(i,j)*h[i+k-N+1] + sum_i h[i]*delta(i+k-N+1,j)
        #                   = h[j+k-N+1] + h[j-k+N-1]  (with valid indices)
        # 
        # More efficiently: the autoconvolution is h * h (convolution).
        # The gradient of the quadratic form sum_k w[k] (h*h)[k] w.r.t h is:
        # 2 * (w * h) where * is cross-correlation appropriately aligned.
        # 
        # Let's use the FFT approach:
        # If f(h) = sum_k w[k] * (h*h)[k], then grad = 2 * (w * h)
        # where (w*h) is the appropriate convolution/correlation.
        #
        # For ac[k] = sum_i h[i]h[i+k-(N-1)], define l = k-(N-1), so ac corresponds to lags l=-(N-1)..N-1
        # f = sum_l w[l+N-1] * sum_i h[i]h[i+l]
        # df/dh[j] = sum_l w[l+N-1] * (h[j+l] + h[j-l])  (with valid indices)
        #          = (w * h)[j] + (w * h)[N-1-j] ... hmm, this is getting complex.
        #
        # Let me use a different approach: use the fact that if we define
        # g = ifft(fft(w) * conj(fft(h))) appropriately.
        #
        # Actually, the simplest correct approach:
        # ac = linear_conv(h, h)  (length 2N-1)
        # f = sum_k w[k] ac[k]
        # grad[j] = sum_k w[k] d(ac[k])/d(h[j])
        # d(ac[k])/d(h[j]) = h[k-j] + h[j-(k-(N-1))] ... 
        # Let me just compute it via: 
        # grad[j] = (w * h)_corr[j] where we define the cross-correlation properly.
        
        # Using FFT: 
        # ac = ifft(|fft(h)|^2) -- but we have linear conv, not circular.
        # For the gradient, note that:
        # sum_k w[k] ac[k] = sum_k w[k] sum_i h[i] h[i+k-(N-1)]
        # Let l = k - (N-1):
        # = sum_l w[l+N-1] sum_i h[i] h[i+l]
        # df/dh[j] = sum_l w[l+N-1] [h[j+l] (when i=j) + h[j-l] (when i+l=j)]
        #          = sum_l w[l+N-1] h[j+l] + sum_l w[l+N-1] h[j-l]
        #          = (w_pad * h)_circ[j] + (w_pad_rev * h)_circ[j]
        # where w_pad is w aligned appropriately.
        
        # Let me just do it with a simple loop-free numpy approach:
        # grad[j] = sum_l w[l+N-1] * (h[(j+l) mod N if valid] + h[(j-l) mod N if valid])
        # This is a circular correlation-like operation.
        
        # Use FFT: define w_ext of length N with appropriate alignment
        # w_ext[l] = w[l+N-1] for l in -(N-1)..N-1, but we need circular.
        
        # Simpler: just compute the gradient via the autoconvolution structure.
        # The gradient of the max of autoconv w.r.t. h is proportional to the 
        # "reversed" autoconv weights convolved with h.
        
        # Let me use a direct approach with FFT for speed:
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        
        # Pad weights to length n_fft
        w_ext = np.zeros(n_fft)
        # w_ext[k] should be w[k] for the correlation computation
        # We want: grad[j] = sum_k w[k] * (h[j+k-N+1] + h[j-k+N-1]) with valid indices
        # This equals: (w * h)[j] in some circular sense + (w_rev * h)[j]
        
        # Let's define:
        # For the first term: sum_k w[k] h[j+k-(N-1)]
        # Let m = k-(N-1), so k = m+N-1, m ranges from -(N-1) to N-1
        # = sum_m w[m+N-1] h[j+m]
        # This is the circular correlation of w with h at shift j.
        
        # In FFT: corr[m] = sum_k a[k] b[k+m] 
        # Using FFT: corr = ifft(conj(fft(a)) * fft(b))
        
        W = np.fft.rfft(w_ext, n=n_fft)
        H = np.fft.rfft(h, n=n_fft)
        
        # First term: sum_m w[m+N-1] h[j+m]
        # This is ifft(conj(fft(w_shifted)) * fft(h)) where w_shifted[m] = w[m+N-1]
        w_shifted = np.zeros(n_fft)
        # w_shifted[m] = w[m+N-1] for m in -(N-1)..N-1
        # In 0-indexed: for m=0..N-1, w_shifted[m] = w[m+N-1]
        # for m=N..2N-2, we need negative m: w_shifted[N+m'] where m' = m-N, 
        # corresponds to m = m'-N < 0, so w[m'+N-1-N] = w[m'-1]... 
        # This is getting complicated. Let me just use np.convolve-based approach.
        
        # Direct gradient computation using convolution:
        # grad[j] = sum_l w_l * h_{j+l} + sum_l w_l * h_{j-l}
        # where w_l = w[l+N-1] and h is zero-padded outside [0,N-1]
        
        # First term: (w_full * h)_full where w_full has w_l at position l+N-1
        # But we need circular-like with zero padding... 
        
        # Let me just do it element-wise with numpy broadcasting for N=600, it's manageable:
        # Actually N=600, so O(N^2) = 360K operations per gradient, that's fine.
        
        # w_l for l = -(N-1) to N-1, stored in w_arr where w_arr[i] = w[i+N-1]
        w_arr = weights  # length 2N-1, w_arr[i] = w[i+N-1] = w at lag i-(N-1)
        
        # grad[j] = sum_{l=-(N-1)}^{N-1} w_arr[l+N-1] * h[j+l] (if 0<=j+l<N) 
        #         + sum_{l=-(N-1)}^{N-1} w_arr[l+N-1] * h[j-l] (if 0<=j-l<N)
        
        # Vectorized:
        # For each j, sum over l of w_arr[l+N-1] * h[j+l] with bounds
        # This is a convolution: (w_arr * h)[j+N-1] in full convolution sense
        # where w_arr is centered.
        
        # Use np.convolve:
        # conv = np.convolve(h, w_arr, 'full')  -- length 2N-1+N-1 = 3N-2
        # conv[k] = sum_i h[i] w_arr[k-i]
        # We want sum_i h[i] w_arr[j-i+N-1]... hmm.
        
        # Let me think differently. 
        # Let v = np.convolve(h, w_arr, 'full')  # length 3N-2
        # v[m] = sum_i h[i] * w_arr[m-i]
        # We want for each j: sum_i h[i] * w_arr[j + N - 1 - i]  -- this doesn't quite work.
        
        # Alternative: just compute it directly with a loop over j, using vectorized inner products.
        # For N=600 this is 600 vector operations of length up to 600, = 360K ops, fast enough.
        
        grad = np.zeros(N)
        for j in range(N):
            # Term 1: sum_l w_arr[l+N-1] * h[j+l] for 0 <= j+l < N
            l_min = max(-(j), -(N-1))
            l_max = min(N-1-j, N-1)
            if l_min <= l_max:
                l_vals = np.arange(l_min, l_max+1)
                w_vals = w_arr[l_vals + N - 1]
                h_vals = h[j + l_vals]
                grad[j] += np.sum(w_vals * h_vals)
            
            # Term 2: sum_l w_arr[l+N-1] * h[j-l] for 0 <= j-l < N
            l_min2 = max(-(N-1), j-(N-1))
            l_max2 = min(N-1, j)
            if l_min2 <= l_max2:
                l_vals2 = np.arange(l_min2, l_max2+1)
                w_vals2 = w_arr[l_vals2 + N - 1]
                h_vals2 = h[j - l_vals2]
                grad[j] += np.sum(w_vals2 * h_vals2)
        
        # Normalize by s^2 derivative: d/dh (max_ac/s^2) ≈ grad/s^2 - 2*max_ac/s^3 * (1/s)
        # For simplicity, use just grad/s^2
        s2 = s * s
        if s2 < 1e-15:
            return np.zeros(N), np.inf
        return grad / s2, smooth_score(h, beta)

    # Initialize with uniform distribution
    h = np.ones(N) / N
    best_h = h.copy()
    best_s = score(h)
    
    # Also try some structured initializations
    init_candidates = [h.copy()]
    
    # Cosine bump
    x = np.linspace(-0.25, 0.25, N+1)[:-1]
    dx = 0.5/N
    cos_h = np.maximum(0, 1 + np.cos(4*np.pi*x))
    cos_h = cos_h / np.sum(cos_h)
    init_candidates.append(cos_h.copy())
    
    # Gaussian bump
    gauss_h = np.exp(-40 * (x**2) / (0.25**2))
    gauss_h = gauss_h / np.sum(gauss_h)
    init_candidates.append(gauss_h.copy())
    
    # Two-bump
    two_bump = np.exp(-100*(x-0.1)**2/(0.05**2)) + np.exp(-100*(x+0.1)**2/(0.05**2))
    two_bump = two_bump / np.sum(two_bump)
    init_candidates.append(two_bump.copy())
    
    # Plateau
    plateau = np.zeros(N)
    center_idx = N // 2
    half_width = N // 6
    plateau[center_idx-half_width:center_idx+half_width] = 1.0
    plateau = plateau / np.sum(plateau)
    init_candidates.append(plateau.copy())
    
    # Evaluate all initial candidates
    for hc in init_candidates:
        s_val = score(hc)
        if s_val < best_s:
            best_s = s_val
            best_h = hc.copy()
    
    h = best_h.copy()
    
    # Gradient descent with adaptive learning rate
    beta = 50.0
    lr = 0.1
    patience = 0
    max_patience = 500
    
    for iteration in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient
        grad, cur_score = grad_smooth(h, beta)
        
        # Check if we're near the deadline more aggressively
        if time.time() > deadline - 15.0:
            beta = 200.0
            grad, cur_score = grad_smooth(h, beta)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project to non-negative and normalize
        h_new = np.maximum(0, h_new)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            h_new = best_h.copy()
        else:
            h_new = h_new / s_new
        
        new_score = score(h_new)
        
        if new_score < best_s - 1e-12:
            h = h_new
            best_h = h_new.copy()
            best_s = new_score
            patience = 0
            # Reduce learning rate slightly
            lr *= 0.98
        else:
            patience += 1
            lr *= 0.95
            if patience > max_patience:
                # Try a different learning rate and reset
                lr = 0.01
                patience = 0
                # Try restarting from best
                h = best_h.copy()
                
                # Reduce beta for smoother landscape
                if beta > 10:
                    beta *= 0.5
        
        # Every 500 iterations, try a random restart from best
        if iteration % 500 == 0 and time.time() < deadline - 30:
            h_restart = best_h.copy()
            # Add small random perturbation
            noise = np.random.randn(N) * 0.01
            h_restart = np.maximum(0, h_restart + noise)
            s_r = np.sum(h_restart)
            if s_r > 1e-15:
                h_restart = h_restart / s_r
                s_r_score = score(h_restart)
                if s_r_score < best_s:
                    h = h_restart
                    best_h = h_restart.copy()
                    best_s = s_r_score
    
    # Final polish: coordinate descent on the best solution
    if time.time() < deadline - 10:
        h = best_h.copy()
        for _ in range(1000):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(0, N, 3):
                if time.time() > deadline - 3:
                    break
                # Try increasing and decreasing h[i] while maintaining sum=1
                # Adjust h[i] by delta, and proportionally adjust others
                for delta in [0.001, 0.0005, 0.0001]:
                    h_try = h.copy()
                    h_try[i] += delta
                    if h_try[i] < 0:
                        continue
                    # Redistribute: scale others to maintain sum
                    others_sum = np.sum(h_try) - h_try[i]
                    if others_sum < 1e-15:
                        continue
                    h_try = h_try.copy()
                    h_try[0] = h_try[0] - delta * (h_try[0] / others_sum)
                    # This is getting complicated. Just renormalize.
                    s_t = np.sum(h_try)
                    h_try = h_try / s_t
                    s_t_score = score(h_try)
                    if s_t_score < best_s - 1e-12:
                        h = h_try.copy()
                        best_h = h_try.copy()
                        best_s = s_t_score
                        improved = True
                        break
                    # Try decreasing
                    h_try2 = h.copy()
                    h_try2[i] -= delta
                    if h_try2[i] < 0:
                        continue
                    s_t2 = np.sum(h_try2)
                    if s_t2 < 1e-15:
                        continue
                    h_try2 = h_try2 / s_t2
                    s_t2_score = score(h_try2)
                    if s_t2_score < best_s - 1e-12:
                        h = h_try2.copy()
                        best_h = h_try2.copy()
                        best_s = s_t2_score
                        improved = True
                        break
    
    return list(best_h)
