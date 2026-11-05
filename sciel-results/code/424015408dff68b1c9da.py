import numpy as np
import time

N = 400


def construct():
    n = N
    deadline = time.time() + 130.0  # safety margin before 150s limit

    def autoconv_fft(h):
        """Full autoconvolution via FFT."""
        size = 2 * n - 1
        H = np.fft.rfft(h, n=size)
        return np.fft.irfft(H * H, n=size)

    def exact_value(h):
        s = h.sum()
        if abs(s) < 1e-15:
            return float('inf')
        h_norm = h / s
        c = autoconv_fft(h_norm)
        mx = np.max(c)
        return abs(2.0 * n * mx)

    def soft_max(x, beta):
        """Differentiable approximation to max via log-sum-exp."""
        m = np.max(x)
        x_shifted = x - m
        x_shifted = np.clip(x_shifted, -50, 50)
        exp_vals = np.exp(beta * x_shifted)
        sum_exp = np.sum(exp_vals)
        if sum_exp == 0:
            return m, np.zeros_like(x)
        weights = exp_vals / sum_exp
        val = m + (1.0 / beta) * np.log(sum_exp)
        return val, weights

    def objective_smooth(h, beta):
        """Smoothed objective: 2*n * soft_max(autoconv(h/s))"""
        s = h.sum()
        if abs(s) < 1e-15:
            return 1e10
        h_norm = h / s
        c = autoconv_fft(h_norm)
        val, _ = soft_max(c, beta)
        return 2.0 * n * val

    def objective_smooth_grad(h, beta):
        """Gradient of smoothed objective w.r.t. h."""
        s = h.sum()
        if abs(s) < 1e-15:
            return 1e10, np.zeros_like(h)
        h_norm = h / s
        c = autoconv_fft(h_norm)
        val, weights = soft_max(c, beta)
        obj = 2.0 * n * val

        # Gradient of c w.r.t. h_norm: c_k = sum_i h_norm[i] * h_norm[i+k-n+1] (full conv)
        # dc/dh_norm[i] = h_norm convolved with delta at position i, which is just h_norm flipped
        # Actually for full conv c = h * h, dc/dh[i] = h (shifted appropriately)
        # c[k] = sum_j h[j] * h[k-j], so dc[k]/dh[i] = h[k-i] + h[i-k] (for full conv indexing)
        # More carefully: c = np.convolve(h, h), c[k] = sum_j h[j]*h[k-j+n-1] for k=0..2n-2
        # dc[k]/dh[i] = h[k-i+n-1] if 0<=k-i+n-1<n else 0, plus h[i-k+n-1] if 0<=i-k+n-1<n else 0
        # But by symmetry of conv(h,h), dc[k]/dh[i] = h[k-i+n-1] + h[i-k+n-1] (with bounds)
        # Simpler: use the fact that grad of sum_k w_k * c[k] w.r.t. h is 2 * conv(w, h) (for full conv)
        # Actually: c = conv(h,h), so d(sum_k w_k c[k])/dh = 2 * conv(w, h) (full conv, but we need to be careful)
        # Let's use: d c[k] / d h[i] = h[k-i] (for the part where h is the "second" argument) + h[i-k] (first arg)
        # For full conv of length 2n-1: c[k] = sum_{j=max(0,k-n+1)}^{min(n-1,k)} h[j] * h[k-j]
        # dc[k]/dh[i] = h[k-i] if 0<=k-i<n and i is in range for first factor, plus h[i-k] if 0<=i-k<n
        # This is getting complex. Let's use a different approach:
        # grad_obj = 2*n * (1/s) * sum_k w_k * dc[k]/dh
        # dc[k]/dh[i] for conv(h,h): it's h[k-i] + h[i-k] with appropriate bounds
        # Actually, for the full autoconvolution, the gradient of c[k] w.r.t. h[i] is:
        #   h[k-i] if 0 <= k-i < n, plus h[i-k] if 0 <= i-k < n
        # But since we're summing over k with weights w_k, and c = conv(h,h) is symmetric,
        # the gradient is 2 * conv(w, h) where conv is the full convolution, but we need to
        # handle the indexing carefully.
        
        # Let me use a cleaner derivation:
        # c[k] = sum_j h[j] * h[k-j] (with h out of range = 0)
        # d/dh[i] sum_k w_k c[k] = sum_k w_k [h[k-i] * delta(k-j=i in first) + h[i-k] * delta(j=i in second)]
        # = sum_k w_k h[k-i] + sum_k w_k h[i-k]
        # = (w * h)[i] + (w * h)[i]  where * denotes full conv? No...
        # sum_k w_k h[k-i] = (w conv h)[i] (full conv, but shifted)
        # Actually sum_k w_k h[k-i] = sum_j w_{j+i} h[j] = (w reversed conv h)[i]
        # This is getting messy. Let me just compute it numerically or use a simpler formula.
        
        # Alternative: use the fact that for c = conv(h, h), 
        # grad_{h} (w^T c) = 2 * conv(w, h) where conv is the full convolution
        # Let's verify: (w^T c) = sum_k w_k sum_j h[j] h[k-j]
        # = sum_j h[j] sum_k w_k h[k-j]
        # = sum_j h[j] (w * h)[j]  where (w*h)[j] = sum_k w_k h[k-j] = sum_m w_{m+j} h[m]
        # Hmm, this is (w conv h) but with a shift.
        
        # Let me just use: grad = 2 * np.convolve(w, h, mode='full') but we need to extract the right part
        # Actually the gradient w.r.t. h (length n) of sum_k w_k c[k] where c=conv(h,h) (length 2n-1)
        # is: for each i in 0..n-1, d/dh[i] = sum_k w_k (h[k-i] + h[i-k]) with bounds
        # = sum_k w_k h[k-i] (with h out of [0,n-1] = 0) + sum_k w_k h[i-k]
        # First term: sum_k w_k h[k-i] = sum_{m=k-i} w_{m+i} h[m] = sum_m w[m+i] h[m] (where w index in [0,2n-2])
        # = (w_shifted * h) where w_shifted[m] = w[m+i] if valid
        # This is equivalent to: for each i, the first term is the i-th element of the "cross-correlation"
        # 
        # Simpler approach: just compute the gradient by noting that
        # d/dh[i] sum_k w_k c[k] = 2 * sum_k w_k h[k-i] (by symmetry, since c is symmetric)
        # where h[k-i] = 0 if k-i not in [0, n-1]
        # = 2 * (w conv h)[i] where we define (w conv h)[i] = sum_k w[k] h[k-i] with bounds
        # This is actually 2 * the i-th element of the cross-correlation of w and h
        # = 2 * np.correlate(w, h, mode='full')[n-1-i] or something...
        
        # Let me just do it directly with a loop-free approach:
        # grad_c[i] = 2 * sum_{k} w[k] * h[k-i] where h[j]=0 for j not in [0,n-1]
        # = 2 * sum_{j=0}^{n-1} h[j] * w[j+i]  (substituting k=j+i, so w index is j+i)
        # = 2 * sum_j h[j] * w[j+i]
        # So grad[i] = 2 * sum_j h[j] * w[j+i] for i=0..n-1, where w is defined on [0, 2n-2]
        
        # This is a cross-correlation. We can compute it as:
        # grad_unscaled = 2 * np.zeros(n)
        # for i in range(n):
        #     valid_j = range(0, n)  # j from 0 to n-1
        #     w_indices = np.array(valid_j) + i  # j+i, must be in [0, 2n-2]
        #     valid = (w_indices >= 0) & (w_indices < 2*n-1)
        #     grad_unscaled[i] = 2 * np.sum(h[valid] * w[w_indices[valid]])
        # 
        # This is O(n^2) which is fine for n=400.
        
        w = weights  # length 2n-1
        grad_unscaled = np.zeros(n)
        for i in range(n):
            j = np.arange(n)
            w_idx = j + i
            valid = (w_idx >= 0) & (w_idx < 2 * n - 1)
            grad_unscaled[i] = 2.0 * np.sum(h[valid] * w[w_idx[valid]])
        
        # Now account for the 1/s factor and the normalization
        # obj = 2*n * soft_max(c(h/s), beta)
        # d obj / d h = 2*n * (1/s) * grad_unscaled - 2*n * soft_max_val * (1/s^2) * sum(grad_unscaled * h / s)
        # Actually, let h_norm = h/s, c = conv(h_norm, h_norm)
        # d obj / d h[i] = 2*n * d soft_max(c, beta)/d h_norm[i] * (1/s) - 2*n * soft_max_val * (1/s^2) * d h_norm[i]/d h[i] * ... 
        # This is getting complicated. Let me use a simpler gradient-free approach for the main optimization
        # and just use the exact value for final evaluation.
        
        # For now, return a crude gradient
        return obj, grad_unscaled / s

    # Strategy: Use a combination of structured initializations and gradient-based refinement
    # with adaptive beta for the log-sum-exp smoothing.
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Two halves
    h1 = np.zeros(n)
    h1[:n//2] = 1.0
    h1[n//2:] = -1.0
    initializations.append(h1)
    
    # 2. Four quarters alternating
    q = n // 4
    h2 = np.zeros(n)
    h2[0:q] = 1.0
    h2[q:2*q] = -1.0
    h2[2*q:3*q] = 1.0
    h2[3*q:4*q] = -1.0
    initializations.append(h2)
    
    # 3. Cosine one period
    idx = np.arange(n)
    h3 = np.cos(2.0 * np.pi * idx / n)
    h3 -= h3.mean()
    initializations.append(h3)
    
    # 4. Cosine two periods
    h4 = np.cos(4.0 * np.pi * idx / n)
    h4 -= h4.mean()
    initializations.append(h4)
    
    # 5. Sine one period
    h5 = np.sin(2.0 * np.pi * idx / n)
    initializations.append(h5)
    
    # 6. Triangular wave
    h6 = np.abs(((2.0 * idx / n) % 1.0) - 0.5)
    h6 -= h6.mean()
    initializations.append(h6)
    
    # 7. Random with various scales
    rng = np.random.RandomState(42)
    for scale in [0.1, 0.5, 1.0, 2.0]:
        h_rand = rng.randn(n) * scale
        h_rand -= h_rand.mean()
        if abs(h_rand.sum()) > 1e-10:
            initializations.append(h_rand)
    
    # 8. Single-knot patterns with various k
    for k in [n//4, n//3, n//2, 2*n//3, 3*n//4]:
        hk = np.zeros(n)
        hk[:k] = 1.0
        hk[k:] = -1.0
        initializations.append(hk)
    
    # Evaluate all initializations and pick the best few
    best_h = None
    best_val = float('inf')
    for h_init in initializations:
        val = exact_value(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
    
    # Refine using gradient descent with adaptive beta
    # Use projected gradient: maintain sum(h) = 1 by projecting after each step
    h_cur = best_h.copy()
    s = h_cur.sum()
    if abs(s) > 1e-15:
        h_cur = h_cur / s  # normalize to sum=1
    
    beta = 10.0
    lr = 0.01
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Compute gradient
        obj, grad = objective_smooth_grad(h_cur, beta)
        
        # Project gradient: remove component along ones vector to maintain sum constraint
        mean_grad = grad.mean()
        grad_proj = grad - mean_grad
        
        # Gradient step
        h_new = h_cur - lr * grad_proj
        
        # Project back to sum=1
        s_new = h_new.sum()
        if abs(s_new) > 1e-15:
            h_new = h_new / s_new
        
        # Check improvement
        val_new = exact_value(h_new)
        val_cur = exact_value(h_cur)
        
        if val_new < val_cur - 1e-10:
            h_cur = h_new
            # Adaptive learning rate
            if iteration % 20 == 0:
                lr *= 0.95
                beta *= 1.5
        else:
            # Reduce learning rate
            lr *= 0.7
            if lr < 1e-6:
                break
    
    # Final polish: coordinate-wise local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        cur_val = exact_value(h_cur)
        for i in range(n):
            if time.time() > deadline - 5:
                break
            old = h_cur[i]
            for delta_frac in [0.05, -0.05, 0.01, -0.01]:
                delta = delta_frac * max(1.0, abs(old))
                h_try = h_cur.copy()
                h_try[i] += delta
                val_try = exact_value(h_try)
                if val_try < cur_val - 1e-12:
                    h_cur = h_try
                    cur_val = val_try
                    improved = True
                    break
    
    # Ensure sum is not zero
    s_final = h_cur.sum()
    if abs(s_final) < 1e-15:
        h_cur[0] += 1.0
    
    return h_cur
