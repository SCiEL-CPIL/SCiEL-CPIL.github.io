import numpy as np
import scipy.optimize as opt
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Initial guess: ones
    h0 = np.ones(N)
    
    # We want to minimize V(h)
    # V(h) = 2 * N * max(conv(h,h)) / sum(h)^2
    
    # To handle max, we use soft-max
    # M_beta(h) = 1/beta * log(sum(exp(beta * C_k)))
    
    # We also need to handle sum(h) != 0.
    # Since V is scale invariant, we can fix sum(h) = N (or 1) to avoid scaling issues.
    # But L-BFGS-B doesn't handle equality constraints well.
    # We can add a penalty term: (sum(h) - N)^2
    # Or we can parameterize h such that sum is fixed.
    # Let's try penalty first.
    
    def objective(params, beta):
        h = params
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        c = np.convolve(h, h)
        # Soft max
        # c can have large values, beta*c might overflow.
        # Subtract max for stability
        c_max = np.max(c)
        # If c_max is very large, exp(beta*c) overflows.
        # Use log-sum-exp trick
        # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        x = beta * c
        x_max = np.max(x)
        log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
        m_soft = log_sum_exp / beta
        
        v = 2 * N * m_soft / (s * s)
        
        # Penalty for sum(h) != N
        # This helps keep h in a reasonable range and avoids s->0
        penalty = 1e-6 * (np.sum(h) - N) ** 2
        
        return v + penalty

    def gradient(params, beta):
        h = params
        s = np.sum(h)
        if abs(s) < 1e-6:
            return np.zeros_like(h)
        
        c = np.convolve(h, h)
        x = beta * c
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x)
        
        # d m_soft / d c_k = (1/beta) * (1/sum_exp) * exp(x_k) * beta = exp(x_k)/sum_exp
        # d m_soft / d h_i = sum_k (d m_soft / d c_k) * (d c_k / d h_i)
        # c_k = sum_j h_j h_{j+k}
        # d c_k / d h_i = h_{i-k} + h_{i+k} (with appropriate bounds)
        # This is equivalent to correlation.
        # grad_m = np.correlate(m_soft_grad_c, h, mode='full') + ... ?
        # Actually, grad of sum_k w_k c_k wrt h is 2 * convolve(w, h) if w is symmetric?
        # c_k = <h, shift_k(h)>
        # d/dh_i sum_k w_k c_k = sum_k w_k (h_{i-k} + h_{i+k})
        # This is the convolution of w with h, doubled?
        # Let w_k = d m_soft / d c_k.
        # grad = np.convolve(w, h) + np.convolve(w, h[::-1])? No.
        # np.correlate(w, h) gives sum w_k h_{i+k}.
        # We need sum_k w_k h_{i-k} + sum_k w_k h_{i+k}.
        # Since c_k = c_{-k} is not true for full conv indices 0..2N-2, but the values are symmetric if h is real?
        # c_k = sum h_j h_{j+k}. c_{2N-2-k} = sum h_j h_{j+2N-2-k}.
        # Actually, let's just use finite differences or analytic gradient of conv.
        # grad_c_k_i = h_{i-k} + h_{i+k} (indices mod/clip)
        # Vectorized:
        # grad_m = np.correlate(w, h, mode='full') + np.correlate(w, h[::-1], mode='full')[::-1]?
        # Simpler:
        # g = np.zeros(2*N-1)
        # for i in range(N):
        #   for k in range(2*N-1):
        #     if 0 <= i-k < N: g[k] += w[k] * h[i-k]
        #     if 0 <= i+k < N: g[k] += w[k] * h[i+k]
        # This is O(N^2). N=400, N^2=160k. Fast enough.
        
        w = exp_x / sum_exp # d m_soft / d c
        
        # Compute gradient of m_soft wrt h
        grad_m = np.zeros(N)
        # c_k = sum_{j} h_j h_{j+k}
        # d c_k / d h_i = h_{i-k} (if 0<=i-k<N) + h_{i+k} (if 0<=i+k<N)
        # grad_m_i = sum_k w_k (h_{i-k} + h_{i+k})
        
        # Term 1: sum_k w_k h_{i-k}
        # This is correlation of w and h
        # w has length 2N-1. h has length N.
        # np.correlate(w, h, mode='same')? No, w is centered?
        # c_k for k=0..2N-2.
        # h_{i-k} is non-zero if i-k in [0, N-1].
        # Let's use FFT for speed? N is small, loops might be fine but Python loops are slow.
        # Use np.correlate.
        # corr1 = np.correlate(w, h, mode='full')
        # w is length 2N-1, h is N.
        # corr1[k] = sum_j w[j] h[j-k]?
        # We want sum_k w_k h_{i-k}.
        # This is (w * h)[i] where * is correlation?
        # Let's just do the loop in NumPy vectorized way or accept O(N^2) with NumPy ops.
        
        # Actually, grad of max(conv(h,h)) is related to the gradient of the autocorrelation.
        # Let's use a simpler gradient: finite difference? No, too slow.
        
        # Analytic gradient:
        # grad_m = np.correlate(w, h, mode='full') + np.correlate(w[::-1], h, mode='full')[::-1]
        # Need to check indexing carefully.
        
        # Alternative: Use scipy.optimize.minimize with grad=None (finite diff) for simplicity and correctness,
        # since N=400 is small enough for finite diff if number of iterations is limited?
        # Finite diff takes N evaluations per gradient. 400 evals * 100 iters = 40000 evals.
        # Each eval is convolve (fast). 40000 * 1ms = 40s. Might be too slow.
        # Let's implement analytic gradient.
        
        # c_k = sum_{j=0}^{N-1-k} h_j h_{j+k} for k>=0? No, full conv.
        # c_k = sum_{j} h_j h_{j+k} where indices are valid.
        # d c_k / d h_i = h_{i-k} + h_{i+k}
        
        # grad_m_i = sum_{k=0}^{2N-2} w_k (h_{i-k} + h_{i+k})
        
        # We can compute this using convolution.
        # Let h_pad = np.pad(h, (N-1, N-1), 'constant')
        # This is getting complex.
        
        # Let's stick to L-BFGS-B with numerical gradient first to ensure it works,
        # then optimize if time permits.
        # But 120s limit.
        # Let's try to implement the gradient correctly.
        
        # w_k is the weight for c_k.
        # grad_h_i = sum_k w_k h_{i-k} + sum_k w_k h_{i+k}
        
        # term1_i = sum_k w_k h_{i-k}
        # term2_i = sum_k w_k h_{i+k}
        
        # term2 is correlation of w and h.
        # np.correlate(w, h, mode='full')
        # w length 2N-1, h length N.
        # Result length 3N-2.
        # We need the part corresponding to i in 0..N-1.
        
        # Let's use a helper to compute the gradient of the soft-max of conv.
        
        # For now, return None to let L-BFGS-B use finite diff, but limit iterations.
        # Or implement a simple gradient.
        
        # Let's try to implement the gradient using FFT?
        # No, let's use the property that grad of sum w_k c_k is 2 * convolve(w, h) if w is symmetric and we are in the right range?
        
        # Let's just use finite difference with a small number of iterations.
        return None

    # Use L-BFGS-B
    # Since gradient is None, it uses finite difference.
    # To speed up, we can provide a gradient.
    # Let's try to write the gradient.
    
    def grad_obj(params, beta):
        h = params
        s = np.sum(h)
        if abs(s) < 1e-6:
            return np.zeros_like(h)
        
        c = np.convolve(h, h)
        x = beta * c
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x)
        w = exp_x / sum_exp
        
        # grad_m_i = sum_k w_k (h_{i-k} + h_{i+k})
        # We can compute this by convolving w with h.
        # Note that w is defined on 0..2N-2.
        # h is defined on 0..N-1.
        
        # h_{i-k} is non-zero if 0 <= i-k <= N-1 => i <= k <= i+N-1.
        # h_{i+k} is non-zero if 0 <= i+k <= N-1 => -i <= k <= N-1-i.
        
        # Let's compute the sum directly for each i? O(N^2).
        # N=400, N^2=160,000.
        # In Python, a loop of 400 with vectorized inner ops is fast.
        
        grad_m = np.zeros(N)
        for i in range(N):
            # k range for h_{i-k}: max(0, i-(N-1)) to min(2N-2, i)
            k_start1 = max(0, i - (N - 1))
            k_end1 = min(2 * N - 2, i)
            if k_start1 <= k_end1:
                # h_{i-k} for k in [k_start1, k_end1]
                # indices of h: i - k
                idx_h1 = i - np.arange(k_start1, k_end1 + 1)
                grad_m[i] += np.dot(w[k_start1:k_end1 + 1], h[idx_h1])
            
            # k range for h_{i+k}: max(0, -i) to min(2N-2, N-1-i)
            k_start2 = max(0, -i)
            k_end2 = min(2 * N - 2, N - 1 - i)
            if k_start2 <= k_end2:
                # h_{i+k} for k in [k_start2, k_end2]
                idx_h2 = i + np.arange(k_start2, k_end2 + 1)
                grad_m[i] += np.dot(w[k_start2:k_end2 + 1], h[idx_h2])
                
        # d V / d h = 2 N / S^2 * d m_soft / d h - 2 * 2 N m_soft / S^3 * d S / d h
        # d S / d h = 1
        # d m_soft / d h = grad_m
        # d V / d h_i = (2 N / S^2) grad_m_i - (4 N m_soft / S^3) * 1
        
        m_soft = (x_max + np.log(sum_exp)) / beta
        v_grad = (2 * N / (s * s)) * grad_m - (4 * N * m_soft / (s ** 3)) * np.ones(N)
        
        # Add penalty gradient
        # penalty = 1e-6 * (s - N)^2
        # d penalty / d h_i = 2 * 1e-6 * (s - N)
        v_grad += 2e-6 * (s - N) * np.ones(N)
        
        return v_grad

    # Optimization loop with increasing beta
    current_h = h0.copy()
    best_h = h0.copy()
    best_v = objective(h0, 1.0)
    
    betas = [1.0, 5.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        res = opt.minimize(
            objective, 
            current_h, 
            method='L-BFGS-B', 
            jac=lambda x: grad_obj(x, beta),
            args=(beta,),
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_v:
            current_h = res.x
            # Normalize to keep sum close to N?
            # The penalty handles it.
            v_curr = objective(current_h, beta)
            if v_curr < best_v:
                best_v = v_curr
                best_h = current_h.copy()
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Final check
    # Ensure sum is not zero
    s = np.sum(best_h)
    if abs(s) < 1e-6:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    construct()
