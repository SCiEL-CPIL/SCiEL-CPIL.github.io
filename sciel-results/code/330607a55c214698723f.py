import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Objective function
    def objective(h):
        # h is shape (N,)
        conv_h = np.convolve(h, h)
        # Soft max
        beta = 50.0
        # Numerical stability for log-sum-exp
        c_max = np.max(conv_h)
        if c_max > 1e6:
            return c_max # Overflow guard
        log_sum_exp = c_max + np.log(np.sum(np.exp(beta * (conv_h - c_max))))
        soft_max = log_sum_exp / beta
        # We want to minimize this
        return soft_max

    def gradient(h):
        conv_h = np.convolve(h, h)
        beta = 50.0
        c_max = np.max(conv_h)
        exp_term = np.exp(beta * (conv_h - c_max))
        sum_exp = np.sum(exp_term)
        # d(soft_max)/d(conv_h)
        d_soft_max_d_conv = exp_term / sum_exp
        
        # d(conv_h)/d(h)
        # conv_h[k] = sum_i h[i] h[k-i]
        # grad[k] = h[k-i] ...
        # The gradient of conv(h,h) w.r.t h is the vector such that:
        # grad[i] = sum_k d_soft_max_d_conv[k] * (h[k-i] + h[k-i] if k-i==i else h[k-i])
        # This is equivalent to:
        # grad = d_soft_max_d_conv * h  (convolved) ?
        # Let's derive:
        # J = sum_k w_k conv_h[k]
        # dJ/dh_i = sum_k w_k (h[k-i] + h[k-i] if k-i==i)
        # This is the convolution of w with h, plus the diagonal term?
        # Actually, d(conv(h,h))/dh is the linear operator:
        # L(h) = conv(h, h)
        # L'(h) = 2 * conv(h, .) ? No.
        # If we perturb h by delta:
        # conv(h+delta, h+delta) = conv(h,h) + conv(delta, h) + conv(h, delta)
        # So gradient is conv(delta, h) + conv(h, delta)
        # conv(delta, h) at index k is sum_i delta[i] h[k-i]
        # So dJ = sum_k w_k (sum_i delta[i] h[k-i] + sum_i h[i] delta[k-i])
        # dJ = sum_i delta[i] (sum_k w_k h[k-i] + sum_k w_k h[k-i]) ?
        # Let j = k-i. sum_k w_k h[k-i] = (w * h)_i ?
        # w * h is conv(w, h).
        # So grad_i = 2 * (w * h)_i ?
        # Check diagonal: if k-i=i, k=2i.
        # In conv(delta, h), term is delta[i] h[i].
        # In conv(h, delta), term is h[i] delta[i].
        # So yes, it's 2 * conv(w, h) if we define conv properly?
        # np.convolve(w, h) gives sum w[j] h[i-j].
        # So grad = 2 * np.convolve(w, h) ?
        # Wait, w is the gradient of the soft max w.r.t conv entries.
        # w has length 2N-1. h has length N.
        # np.convolve(w, h) has length 3N-2.
        # We need the part corresponding to h indices 0..N-1.
        # The derivative of conv(h,h) w.r.t h_i is:
        # sum_k w_k (h[k-i] + h[k-i] if k-i==i)
        # Let's just compute it directly for simplicity and correctness, N=400 is small.
        
        grad = np.zeros(N)
        # w = d_soft_max_d_conv
        for i in range(N):
            # sum_k w_k h[k-i]
            # k-i is index in h.
            # k ranges 0 to 2N-2.
            # h[k-i] is non-zero if 0 <= k-i < N.
            # So k ranges i to i+N-1.
            # Also need to handle the double counting for k-i=i.
            
            # Efficient way:
            # term1 = np.convolve(w, h)
            # term2 = np.convolve(h, w)
            # Actually, grad_i = (w * h)_i + (h * w)_i ?
            # (w * h)_i = sum_j w[j] h[i-j].
            # (h * w)_i = sum_j h[j] w[i-j].
            # These are the same if w is symmetric? w is not necessarily symmetric.
            # But conv(h,h) is symmetric? No, h is not symmetric.
            # However, the gradient formula is:
            # grad_i = sum_k w_k (h[k-i] + h[k-i] if k-i==i)
            
            # Let's use the convolution trick.
            # A_i = sum_k w_k h[k-i]
            # This is the cross-correlation of w and h?
            # A = np.correlate(w, h, mode='full')?
            # Let's just use the loop or a vectorized slice sum.
            pass

        # Vectorized gradient calculation
        # grad_i = sum_{k} w_k h_{k-i}
        # This is the convolution of w and h, but reversed?
        # Let's compute T = np.convolve(w, h)
        # T[m] = sum_j w[j] h[m-j]
        # We want sum_k w[k] h[k-i]. Let m=k-i => k=m+i.
        # sum_m w[m+i] h[m].
        # This is the cross-correlation.
        
        # Simpler:
        # grad = 2 * np.convolve(w, h)[:N] ? No.
        
        # Let's stick to the definition for safety, N=400 is small.
        # But 400 * 800 = 320,000 ops per gradient. 1000 iters = 320M ops. Might be slow in Python loop.
        # Use NumPy:
        # w is length 2N-1.
        # We want for each i, sum_k w[k] h[k-i].
        # This is the convolution of w and h, evaluated at specific indices?
        # Let's compute C_wh = np.convolve(w, h)
        # C_wh[p] = sum_j w[j] h[p-j]
        # We want sum_k w[k] h[k-i].
        # Let p = k-i. k = p+i.
        # sum_p w[p+i] h[p].
        # This is the cross-correlation of w and h at lag i.
        # np.correlate(w, h, mode='full') gives sum_p w[p] h[p+i] ?
        
        # Let's use the property:
        # grad = np.zeros(N)
        # for i in range(N):
        #    grad[i] = np.dot(w, h[i:i+N-1] if ... )
        
        # Actually, just use the loop with NumPy dot products.
        grad = np.zeros(N)
        for i in range(N):
            # h[k-i] is non-zero for k in [i, i+N-1]
            k_start = i
            k_end = i + N
            # w slice: w[k_start:k_end]
            # h slice: h[0:N] (indices k-i from 0 to N-1)
            # Wait, h[k-i] for k in [i, i+N-1] is h[0] to h[N-1].
            # So we need w[i : i+N]
            if i + N <= len(w):
                grad[i] = np.dot(w[i : i+N], h)
            else:
                # Handle bounds
                w_slice = w[i:]
                h_slice = h[:len(w_slice)]
                grad[i] = np.dot(w_slice, h_slice)
                
        # Add the diagonal term: if k-i=i, i.e., k=2i, add w[2i] h[i]
        # The dot product above sums w[k] h[k-i].
        # If k=2i, term is w[2i] h[i].
        # The derivative of h_i^2 is 2 h_i.
        # The dot product includes w[2i] h[i] once.
        # We need it twice?
        # d/dh_i (h_i h_i) = 2 h_i.
        # d/dh_i (h_j h_i) = h_j.
        # In conv(h,h), the term h_i^2 appears only once (when j=i and k-i=i).
        # So the derivative is 2 h_i * w[2i].
        # My dot product computed sum w[k] h[k-i].
        # For k=2i, it is w[2i] h[i].
        # So I need to add w[2i] h[i] again.
        
        # Add diagonal correction
        diag_indices = np.arange(0, N, 1)
        w_diag = w[2*diag_indices]
        grad += w_diag * h
        
        return grad

    # Optimization
    best_h = np.ones(N)
    best_val = float('inf')
    
    # Try a few initializations
    inits = []
    # 1. Ones
    inits.append(np.ones(N))
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 * 5))
    # 3. Cosine bell
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.randn(N))
    
    for h0 in inits:
        if time.time() > deadline:
            break
            
        # Normalize to sum 1
        h0 = h0 / np.sum(h0)
        
        try:
            res = scipy.optimize.minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            h_opt = res.x
            # Ensure sum is not zero
            if np.sum(h_opt) == 0:
                continue
            # Normalize
            h_opt = h_opt / np.sum(h_opt)
            
            # Evaluate true value
            conv_h = np.convolve(h_opt, h_opt)
            val = abs(2 * N * np.max(conv_h) / np.sum(h_opt)**2)
            
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception as e:
            pass

    # Final check
    if np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# Need to import scipy
import scipy.optimize

# Re-run construct to ensure it works
# Note: The prompt asks for the complete program.
