import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
CONV_LEN = 2 * N - 1  # Length of full convolution

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _conv_matrix():
    """Build matrix C such that C @ h gives conv(h, h) for h with sum=1.
    Actually, conv is quadratic. We use: conv(h,h)[k] = sum_i h[i]*h[k-i+N-1]
    We can compute this as h @ M_k @ h for each k, but for optimization we need
    gradients. Let's use a different approach: build the full convolution as
    a quadratic form.
    
    For h of length N, conv(h,h) has length 2N-1.
    conv(h,h)[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i]
    
    We can write this as: for each output index k, it's a bilinear form.
    For optimization, we'll compute conv and its gradient numerically or analytically.
    """
    pass

def _compute_conv_and_grad(h):
    """Compute conv(h,h) and its gradient w.r.t. h.
    conv_k = sum_i h[i]*h[k-i] (with appropriate bounds)
    d(conv_k)/d(h[j]) = h[k-j] + h[j-k] (when indices valid)
    """
    conv = np.convolve(h, h, mode='full')
    grad = np.zeros_like(h)
    for k in range(CONV_LEN):
        # conv[k] = sum_i h[i]*h[k-i] where 0<=i<N and 0<=k-i<N
        # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
        # But since conv is symmetric, and we sum over i, the gradient is:
        # For each i in valid range: d/dh[j] of h[i]*h[k-i] = 
        #   delta(i,j)*h[k-i] + h[i]*delta(k-i,j)
        # = delta(i,j)*h[k-i] + h[i]*delta(k-i,j)
        # Sum over i: h[k-j] (from first term, when i=j) + h[j-k] (from second term, when k-i=j => i=k-j)
        # But we need to be careful with bounds.
        i_min = max(0, k - (N - 1))
        i_max = min(N - 1, k)
        for i in range(i_min, i_max + 1):
            # Contribution to grad[j] from term h[i]*h[k-i]:
            # If j == i: adds h[k-i] = h[k-j]
            # If j == k-i: adds h[i] = h[k-j]
            # So if j is in [i_min, i_max] and j == i or j == k-i, add h[k-j]
            # More simply: grad[j] += h[k-j] if 0<=k-j<N, plus h[j-k] if 0<=j-k<N
            # But j-k = -(k-j), so h[j-k] is only valid if j>k, giving h[j-k]
            pass
    # Simpler: use the fact that conv(h,h) = ifft(fft(h, 2N)**2)
    # Gradient can be computed via FFT too
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv_fft = H * H
    conv = np.real(np.fft.ifft(conv_fft))[:CONV_LEN]
    # Gradient: d(conv)/d(h) 
    # conv = ifft(|H|^2) ... actually conv_fft = H*H not |H|^2
    # Let's just use numerical gradient for simplicity, or compute analytically
    # Analytical: conv_k = sum_{i} h_i h_{k-i}
    # d(conv_k)/d(h_j) = h_{k-j} + h_{j-k} (with h_m = 0 for m outside [0,N-1])
    for k in range(CONV_LEN):
        for j in range(N):
            g = 0.0
            if 0 <= k - j < N:
                g += h[k - j]
            if 0 <= j - k < N:
                g += h[j - k]
            grad[j] += g  # This is wrong - we're accumulating over k
    # Actually, we need d(conv_k)/d(h_j) for each k separately, but for the
    # objective we need the gradient of the max, which is more complex.
    # Let's return just conv and compute gradient numerically if needed.
    return conv

def _optimize_h_given_t(t, h_init, max_iter=1000):
    """Optimize h to maximize sum(h^2) subject to conv(h,h) <= t and sum(h)=1, h>=0.
    Using SLSQP on log-scale variables.
    """
    def neg_obj(x):
        h = np.exp(x)
        h = h / np.sum(h)
        return -np.sum(h**2)
    
    # Constraints: conv(h,h) <= t for each component
    # And h >= 0 (handled by exp), sum(h)=1 (handled by normalization)
    constraints = []
    for k in range(CONV_LEN):
        def make_constraint(kk):
            def constraint(x):
                h = np.exp(x)
                h = h / np.sum(h)
                conv = np.convolve(h, h, mode='full')
                return t - conv[kk]
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_constraint(k)})
    
    x0 = np.log(h_init + 1e-12)
    try:
        res = minimize(neg_obj, x0, method='SLSQP', 
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success:
            h = np.exp(res.x)
            h /= np.sum(h)
            return h
    except Exception:
        pass
    return h_init

def _optimize_t_given_h(h, max_iter=100):
    """Optimize t to minimize t subject to conv(h,h) <= t.
    This is simply t = max(conv(h,h)).
    """
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # Random sparse initialization
    for _ in range(200):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 20)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.05
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # 2. Block-coordinate descent with epigraph formulation
    # Objective: maximize sum(h^2) / t  subject to conv(h,h) <= t, sum(h)=1, h>=0
    # Equivalent to: minimize t / sum(h^2) or use the ratio directly
    # Block 1: Given t, optimize h to maximize sum(h^2) s.t. conv(h,h) <= t
    # Block 2: Given h, optimize t = max(conv(h,h))
    
    h_curr = best_h.copy()
    t_curr = np.max(np.convolve(h_curr, h_curr, mode='full'))
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Block 1: Optimize h given t
        h_new = _optimize_h_given_t(t_curr, h_curr, max_iter=500)
        if h_new is not None and np.sum(h_new) > 0:
            h_new /= np.sum(h_new)
            h_curr = h_new
        
        # Block 2: Optimize t given h
        t_new = _optimize_t_given_h(h_curr)
        t_curr = t_new
        
        # Update best
        s = _score_raw(h_curr)
        if s > best_score:
            best_score = s
            best_h = h_curr.copy()
        
        # Check convergence
        if t_curr > 0 and abs(t_new - t_curr) / t_curr < 1e-10:
            break
    
    # 3. Final polish with direct optimization of the ratio
    if time.time() < deadline - 20:
        def neg_obj_ratio(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            if max_conv == 0:
                return 0.0
            return -(l2_sq / max_conv)
        
        h_start = best_h.copy()
        h_start[h_start < 1e-10] = 1e-10
        x0 = np.log(h_start)
        
        try:
            res = minimize(neg_obj_ratio, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-15})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
        
        # SLSQP polish
        if time.time() < deadline - 10:
            def neg_obj_slsqp(x):
                h = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h**2)
                conv = np.convolve(h, h, mode='full')
                max_conv = np.max(conv)
                if max_conv == 0:
                    return 0.0
                return -(l2_sq / max_conv)
            
            try:
                res = minimize(neg_obj_slsqp, x0, method='SLSQP', 
                              options={'maxiter': 1000, 'ftol': 1e-14})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
            except Exception:
                pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
