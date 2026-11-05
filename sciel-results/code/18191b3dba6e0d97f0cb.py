import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 140.0 # Safety margin

def compute_conv(h):
    return np.convolve(h, h, mode='full')

def soft_max_val(u, beta):
    # Numerical stability
    u_max = np.max(u)
    if u_max > 500:
        return u_max
    exp_u = np.exp(beta * (u - u_max))
    S = np.sum(exp_u)
    if S == 0:
        return u_max
    return u_max + np.log(S) / beta

def objective_and_grad(x, beta):
    # x has length N-1
    h = np.empty(N)
    h[:N-1] = x
    h[N-1] = 1.0 - np.sum(x)
    
    u = compute_conv(h)
    
    # Soft max value
    u_max = np.max(u)
    # Avoid overflow
    if beta * (u_max - np.min(u)) > 500:
         # If range is huge, just use max
         # Gradient is basically gradient of max
         idx_max = np.argmax(u)
         # grad u_k w.r.t h_j
         # d/dh_j u_k = h_{k-j} + h_{j-k}
         # We need grad J w.r.t h
         # If we just use max, p_k is 1 at max, 0 else
         p = np.zeros_like(u)
         p[idx_max] = 1.0
    else:
        exp_u = np.exp(beta * (u - u_max))
        S = np.sum(exp_u)
        p = exp_u / S # Probabilities
        J = u_max + np.log(S) / beta
    
    # Compute gradient of J w.r.t h
    # dJ/dh_j = sum_k p_k * (h_{k-j} + h_{j-k})
    # Note: h is defined on 0..N-1. u on 0..2N-2.
    # We can compute this by convolving p with h?
    # Let's check:
    # sum_k p_k h_{k-j} is (p * h)_{?}
    # (p * h)_m = sum_k p_k h_{m-k}.
    # We want sum_k p_k h_{k-j}. Let m = -j? No.
    # sum_k p_k h_{k-j} = sum_k p_k h_{-(j-k)} = (p * \text{flip}(h))_{j}?
    # Let flip(h)_i = h_{N-1-i}.
    # Actually, simpler:
    # Term 1: T1_j = sum_k p_k h_{k-j} (where h is 0 outside 0..N-1)
    # Term 2: T2_j = sum_k p_k h_{j-k}
    # Note T1_j = T2_{-j}? No, indices are different.
    # T2_j is the standard convolution (p * h)_j if p was centered?
    # Let's just use explicit loops or vectorized operations for N=400.
    
    # Vectorized approach:
    # Create a matrix P of shape (2N-1, N) where P[k, j] = p_k if 0 <= k-j < N else 0?
    # No, that's too big (800*400 = 320k, fine).
    
    # Let's construct the gradient directly.
    # grad_h = np.zeros(N)
    # for j in range(N):
    #    # h_{k-j} is non-zero for k in [j, j+N-1]
    #    k1_start = max(0, j)
    #    k1_end = min(2*N-1, j+N-1)
    #    # h_{j-k} is non-zero for k in [j-N+1, j]
    #    k2_start = max(0, j-N+1)
    #    k2_end = min(2*N-1, j)
    
    # This is O(N^2). 160k ops. Fast.
    
    grad_h = np.zeros(N)
    # Precompute indices?
    # Let's use np.convolve if possible.
    # sum_k p_k h_{k-j} = (p * h)_{2N-2-j}?
    # Let's test small case.
    # h=[1,2], p=[0.5, 0.3, 0.2] (len 3)
    # u = [1, 4, 4]
    # j=0: sum p_k h_k = 0.5*1 + 0.3*2 + 0 = 1.1
    # j=1: sum p_k h_{k-1} = 0.5*0 + 0.3*1 + 0.2*2 = 0.7
    # j=0 term 2: sum p_k h_{-k} = 0.5*1 + 0.3*0 + 0.2*0 = 0.5
    # j=1 term 2: sum p_k h_{1-k} = 0.5*0 + 0.3*2 + 0.2*0 = 0.6
    
    # Convolution p * h:
    # [0.5*1, 0.5*2+0.3*1, 0.3*2+0.2*1, 0.2*2] = [0.5, 1.3, 0.8, 0.4]
    # Indices 0..3.
    # T1_0 = 1.1. (p*h)[?]
    # T1_j = sum_k p_k h_{k-j}.
    # This is cross-correlation?
    # Let's just do the O(N^2) loop in Python or vectorized with broadcasting.
    
    # Vectorized:
    # k is 0..2N-2. j is 0..N-1.
    # We want grad_h[j] = sum_k p[k] * (h[k-j] (if valid) + h[j-k] (if valid))
    
    # Let's use scipy.signal.correlate or similar?
    # No, let's stick to numpy.
    
    # T1: sum_k p_k h_{k-j}
    # This is the convolution of p and h, but shifted?
    # Let v = np.convolve(p, h, 'full')
    # v[m] = sum_k p_k h_{m-k}
    # We want sum_k p_k h_{k-j} = sum_k p_k h_{-(j-k)}
    # Let i = j-k. k = j-i.
    # sum_i p_{j-i} h_i
    # This is (p * h)_{j} if p was reversed?
    # Let's just use the explicit calculation for safety and simplicity with N=400.
    
    # p shape (2N-1,)
    # h shape (N,)
    
    # Create array of indices k for each j?
    # No, just use slicing.
    
    for j in range(N):
        # Part 1: h_{k-j}
        # k-j in [0, N-1] => k in [j, j+N-1]
        k_start1 = j
        k_end1 = min(2*N-1, j+N-1) + 1
        if k_start1 < k_end1:
            # h indices: k - j
            # k is k_start1...k_end1-1
            # h indices are 0...k_end1-k_start1-1
            idx_h1 = np.arange(0, k_end1 - k_start1)
            grad_h[j] += np.dot(p[k_start1:k_end1], h[idx_h1])
            
        # Part 2: h_{j-k}
        # j-k in [0, N-1] => k in [j-N+1, j]
        k_start2 = max(0, j-N+1)
        k_end2 = j + 1
        if k_start2 < k_end2:
            # h indices: j-k
            # k is k_start2...k_end2-1
            # h indices are j-k_start2 ... j-(k_end2-1) = 0
            # So h indices are descending from j-k_start2 to 0
            idx_h2 = np.arange(j - k_start2, -1, -1)
            grad_h[j] += np.dot(p[k_start2:k_end2], h[idx_h2])
            
    # Now we have grad_h w.r.t h.
    # We need grad_x w.r.t x (first N-1 elements).
    # h_{N-1} = 1 - sum(x)
    # dJ/dx_j = dJ/dh_j - dJ/dh_{N-1}
    grad_x = grad_h[:N-1] - grad_h[N-1]
    
    return J, grad_x

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    h0 = np.ones(N) / N
    x0 = h0[:N-1]
    
    best_val = np.inf
    best_x = x0.copy()
    
    # Try a few random restarts to avoid local minima
    # But uniform is a strong local min?
    # Let's try one optimization run first.
    
    beta = 10.0 # Start with moderate beta
    
    # Increase beta to sharpen max
    for beta_iter in [10.0, 50.0, 100.0]:
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta_iter)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta_iter)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_val:
                best_x = res.x
                # Compute actual value
                h_test = np.empty(N)
                h_test[:N-1] = best_x
                h_test[N-1] = 1.0 - np.sum(best_x)
                u_test = compute_conv(h_test)
                val = 2 * N * np.max(u_test) / (np.sum(h_test)**2)
                if np.isfinite(val) and val < best_val:
                    best_val = val
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
    # Final check and return
    h_final = np.empty(N)
    h_final[:N-1] = best_x
    h_final[N-1] = 1.0 - np.sum(best_x)
    
    # Ensure sum is exactly 1 (floating point)
    s = np.sum(h_final)
    if s != 0:
        h_final = h_final / s
        
    # Calculate final score
    u_final = compute_conv(h_final)
    score = 2 * N * np.max(u_final) / (np.sum(h_final)**2)
    
    # If it's worse than seed (2.0), return seed
    if score >= 2.0:
        return np.ones(N)
        
    return h_final

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 140.0

def _compute_conv(h):
    return np.convolve(h, h, mode='full')

def _objective_and_grad(x, beta):
    h = np.empty(N)
    h[:N-1] = x
    h[N-1] = 1.0 - np.sum(x)
    
    u = _compute_conv(h)
    
    u_max = np.max(u)
    # Check for overflow risk
    range_u = np.max(u) - np.min(u)
    if beta * range_u > 500:
        idx_max = np.argmax(u)
        p = np.zeros_like(u)
        p[idx_max] = 1.0
        J = u_max
    else:
        exp_u = np.exp(beta * (u - u_max))
        S = np.sum(exp_u)
        if S == 0:
            p = np.zeros_like(u)
            p[np.argmax(u)] = 1.0
            J = u_max
        else:
            p = exp_u / S
            J = u_max + np.log(S) / beta
    
    grad_h = np.zeros(N)
    for j in range(N):
        # Part 1: h_{k-j}
        k_start1 = j
        k_end1 = min(2*N-1, j+N-1) + 1
        if k_start1 < k_end1:
            idx_h1 = np.arange(0, k_end1 - k_start1)
            grad_h[j] += np.dot(p[k_start1:k_end1], h[idx_h1])
            
        # Part 2: h_{j-k}
        k_start2 = max(0, j-N+1)
        k_end2 = j + 1
        if k_start2 < k_end2:
            idx_h2 = np.arange(j - k_start2, -1, -1)
            grad_h[j] += np.dot(p[k_start2:k_end2], h[idx_h2])
            
    grad_x = grad_h[:N-1] - grad_h[N-1]
    return J, grad_x

def construct():
    start_time = time.time()
    
    # Start with uniform
    h0 = np.ones(N) / N
    x0 = h0[:N-1].copy()
    
    best_x = x0.copy()
    best_val = 2.0 # Seed value
    
    # Try increasing beta
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        try:
            # Use current best as start
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                best_x,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_val:
                # Verify actual max
                h_test = np.empty(N)
                h_test[:N-1] = res.x
                h_test[N-1] = 1.0 - np.sum(res.x)
                u_test = _compute_conv(h_test)
                val = 2 * N * np.max(u_test) / (np.sum(h_test)**2)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass

    h_final = np.empty(N)
    h_final[:N-1] = best_x
    h_final[N-1] = 1.0 - np.sum(best_x)
    
    s = np.sum(h_final)
    if s != 0:
        h_final = h_final / s
    else:
        return np.ones(N)
        
    return h_final

# EVOLVE-BLOCK-END
