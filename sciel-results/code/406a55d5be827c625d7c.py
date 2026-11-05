"""
Strategy: Direct Convex-Style Optimization over Heights with Active Set Handling.

Instead of restricting to a parametric family (chirp, sinusoid), we optimize the
400 heights directly using L-BFGS-B. The objective function is:
    value = abs(2 * N * max(conv(h)) / sum(h)^2)

This function is non-smooth due to the max and abs. However, we can use
L-BFGS-B which handles non-smoothness reasonably well via finite differences
or by treating the max term carefully. 

Key Insight:
The term `max(conv(h))` is the maximum of 799 linear functions of h (since
conv(h) is linear in h). The absolute value makes it V-shaped.
We can minimize the upper bound or use a smooth approximation, but more simply,
we can just minimize the raw value. L-BFGS-B often works surprisingly well on
these "minimax" type problems if the landscape isn't too rugged.

To improve convergence:
1. Start from a random Gaussian or Zadoff-Chu-like initial point to escape the
   trivial basin of `ones`.
2. Use a multi-start strategy with different random seeds.
3. Add a small regularization term to prevent sum(h) from going to 0 or infinity,
   though the problem statement says sum(h) != 0. We can normalize h internally
   to keep sum(h) = 1 to simplify the optimization landscape, but the evaluator
   computes the value from the raw h. 
   
   Actually, the value is scale-invariant:
   value(c*h) = abs(2*N*max(conv(c*h)) / (c*sum(h))^2)
   = abs(2*N*c^2*max(conv(h)) / c^2*(sum(h))^2)
   = value(h)
   
   So we can fix the scale. Let's enforce sum(h) = 1 during optimization by
   projecting or by adding a penalty, OR simply optimize h and normalize at the end?
   No, if we normalize at the end, the gradient in the optimizer needs to account
   for that. 
   
   Better approach: Optimize h directly, but the objective is scale invariant.
   This creates a valley of equal value. L-BFGS-B might struggle with the flat
   direction. 
   
   Solution: Fix the scale by constraining sum(h) = 1. We can do this by
   optimizing 399 free variables and setting the 400th to make the sum 1.
   Or, more simply, just add a large penalty if |sum(h)| is not close to 1.
   Actually, let's just optimize h and rely on the fact that the objective is
   scale invariant. The gradient will be zero along the scaling direction.
   To avoid issues, we can normalize h inside the objective function for
   numerical stability, but return the raw value.
   
   Wait, if I normalize h inside the objective, the gradient computed by
   L-BFGS-B (finite diff) will be correct for the normalized version.
   
   Let's try a different structural change: 
   Use the fact that we want max(conv(h)) to be small. 
   This is equivalent to minimizing the peak sidelobe level of the autocorrelation.
   
   A very different strategy: 
   Use a genetic algorithm or simulated annealing directly on the 400-vector.
   But 400 dimensions is high for GA.
   
   Let's stick to L-BFGS-B but with a smarter initialization and a 
   "smooth max" approximation to help the gradient.
   
   Objective:
   f(h) = 2 * N * max_k (conv_k(h)) / sum(h)^2
   
   We want to minimize abs(f(h)).
   
   Let's try minimizing f(h) first, assuming we can make max(conv) negative?
   If max(conv) is negative, abs(f(h)) = -f(h).
   If max(conv) is positive, abs(f(h)) = f(h).
   
   The seed `ones` gives max(conv) = N (positive).
   If we can make the autocorrelation have negative peaks, we might reduce the value.
   However, conv(h, h)[N-1] = sum(h^2) is always positive.
   So max(conv(h)) is always >= sum(h^2) > 0.
   Therefore, abs() is redundant for the maximum, as the max is always positive.
   
   So we minimize:
   value = 2 * N * max_k (conv_k(h)) / sum(h)^2
   
   Since conv(h) is linear, max_k conv_k(h) is convex in h.
   sum(h)^2 is convex in h (but we are dividing by it).
   The ratio of a convex function and a convex function is not necessarily convex.
   However, if we fix the scale, say sum(h) = 1, then we minimize max_k conv_k(h).
   This IS a convex optimization problem!
   min_h max_k (conv_k(h)) subject to sum(h) = 1.
   
   This is a linear program (LP) if we introduce a variable t >= conv_k(h) for all k.
   But we can solve it with scipy.optimize.linprog or just use L-BFGS-B on the
   smoothed max.
   
   Let's use L-BFGS-B on the smoothed max:
   max_k x_k ~ logsumexp(beta * x_k) / beta
   As beta -> inf, it approaches max.
   
   We will optimize h with sum(h)=1 constraint handled by fixing one variable or
   using a penalty.
   
   Let's define:
   h[0:N-1] are free.
   h[N-1] = 1 - sum(h[0:N-1])
   
   Objective:
   smoothed_max_conv(h) / 1^2  (since sum=1)
   
   Then value = 2 * N * smoothed_max_conv(h)
   
   We minimize this.
   
   This is a much more structured approach than the chirp one.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_conv(h):
    return np.convolve(h, h)

def _objective_and_grad(params):
    """
    params: 399 values for h[0:N-1]
    h[N-1] is determined by sum constraint = 1.
    Returns smoothed max of conv(h).
    """
    h = np.zeros(N)
    h[:N-1] = params
    h[N-1] = 1.0 - np.sum(params)
    
    conv = _compute_conv(h)
    
    # Use logsumexp to approximate max
    # Scale beta to control sharpness. Higher beta -> closer to true max.
    # Start with moderate beta, increase during optimization? 
    # For simplicity, use a fixed high beta like 100 or 500.
    beta = 500.0
    # logsumexp(beta * conv) / beta
    # Need to handle numerical stability
    max_conv = logsumexp(beta * conv) / beta
    
    # We don't easily get gradient of logsumexp in scipy.optimize.minimize
    # without custom gradient. Let's provide one.
    # d/dh logsumexp(beta * conv) = sum_j (beta * exp(beta * conv_j) * d(conv_j)/dh) / sum_j exp(beta * conv_j)
    # = sum_j softmax_j * beta * d(conv_j)/dh
    # d(conv_j)/dh = h[j::-1] padded + h[j+1::-1] padded ... essentially the impulse response of the convolution.
    # Actually, conv_j = sum_k h[k] h[j-k]
    # d(conv_j)/dh[m] = h[j-m] + h[m-j] (if indices valid)
    
    # This gradient computation is O(N^2) per step. 
    # N=400, so 160,000 ops. This is fast.
    
    # Let's compute the gradient.
    # softmax_weights = exp(beta * (conv - max_conv)) / sum(exp(...))
    # Actually, logsumexp already handles the shift.
    # Let's just use finite differences for simplicity first, or implement grad.
    # L-BFGS-B with analytic gradient is faster.
    
    # Let's implement the gradient.
    # conv_j = sum_{k} h_k h_{j-k}
    # Let C be the convolution matrix such that conv = C h.
    # C is (2N-1) x N.
    # We don't want to build C explicitly (799x400 = 320k floats, ok).
    # But C changes if h changes? No, C is linear operator.
    # C_{j, m} = 1 if there exists k such that j-k = m? 
    # conv_j = sum_{m} h_m h_{j-m}
    # So C_{j, m} = h_{j-m} if 0 <= j-m < N else 0.
    # This matrix depends on h! So it's not a linear operator in the standard sense for gradient?
    # Wait, conv(h) is quadratic in h.
    # d(conv_j)/dh_m = h_{j-m} + h_{m-j} (if indices valid)
    
    # Let's compute gradient numerically to avoid bugs, but check time.
    # 399 params. 399 evaluations of objective per gradient step.
    # Objective is O(N^2) = 160k ops.
    # 399 * 160k = 64 million ops per gradient.
    # L-BFGS-B might take 100 steps. 6.4 billion ops. Too slow in Python.
    
    # We MUST use analytic gradient.
    
    # Gradient of max_conv w.r.t h:
    # g = sum_j w_j * d(conv_j)/dh
    # where w_j = exp(beta * conv_j) / sum_k exp(beta * conv_k)
    
    # d(conv_j)/dh_m = h_{j-m} + h_{m-j}
    
    # Let's compute w first.
    # Use stable logsumexp
    conv_shifted = conv - np.max(conv)
    exp_conv = np.exp(beta * conv_shifted)
    sum_exp = np.sum(exp_conv)
    w = exp_conv / sum_exp
    
    # Now compute grad_h = sum_j w_j * (h_{j-m} + h_{m-j})
    # This is a convolution-like operation.
    # Let A = w. 
    # grad_h[m] = sum_j w_j (h_{j-m} + h_{m-j})
    # = sum_j w_j h_{j-m} + sum_j w_j h_{m-j}
    # Let i = j-m => j = i+m. Sum_i w_{i+m} h_i
    # This is a cross-correlation of w and h.
    
    # corr1[m] = sum_i w_{i+m} h_i
    # corr2[m] = sum_j w_j h_{m-j} = sum_k w_{m-k} h_k (let k=m-j => j=m-k)
    
    # We can compute these using np.convolve or fft.
    # w is length 2N-1. h is length N.
    
    # corr1: 
    # w is defined for j in [0, 2N-2].
    # For a fixed m in [0, N-1], i+m must be in [0, 2N-2].
    # i in [0, N-1].
    
    # Let's use FFT for speed.
    # corr1[m] = sum_i w_{i+m} h_i
    # This is like a convolution if we flip one.
    
    # Simpler: just loop over m? N=400. 400 iterations.
    # Inside loop, sum over i (N=400). 160,000 ops. Fast enough in Python?
    # 160k ops is very fast.
    
    grad_h = np.zeros(N)
    for m in range(N):
        # Term 1: sum_i w_{i+m} h_i
        # i ranges 0..N-1
        # j = i+m ranges m..m+N-1
        # Check if j is in [0, 2N-2]
        # Since m>=0 and i<N, j < m+N <= 2N-1. 
        # j >= 0 always.
        # So we just need to slice w[m:m+N]
        # But w has length 2N-1.
        # If m+N > 2N-1, we need to pad.
        # m+N <= 2N-1 is true for m <= N-1.
        # So w[m:m+N] is valid.
        t1 = np.dot(w[m:m+N], h)
        
        # Term 2: sum_j w_j h_{m-j}
        # Let k = m-j => j = m-k
        # k ranges such that m-k is in [0, 2N-2] and k in [0, N-1]
        # k in [0, N-1] => j in [m-N+1, m]
        # We need to sum w_{m-k} h_k for k=0..N-1
        # This is dot(w[m-N+1:m+1] reversed?, h)
        # Let's just iterate k
        # Or use slice:
        # We want sum_k w_{m-k} h_k
        # This is cross-correlation of w and h at lag m?
        # Let's just do the loop for simplicity and correctness, it's only 400*400=160k ops.
        # Actually, the loop above is O(N^2) total.
        
        # Let's optimize the inner loop using numpy slicing if possible.
        # t2 = sum_{k=0}^{N-1} w[m-k] h[k]
        # Indices for w: m, m-1, ..., m-N+1
        # We need to handle boundaries.
        
        # Let's stick to the simple loop for t1 and t2 to be safe, 
        # but vectorize the outer sum if possible.
        # Actually, t1 and t2 are related.
        # t1[m] = sum_i h_i w_{i+m}
        # t2[m] = sum_k h_k w_{m-k}
        
        # We can compute all t1 and t2 via convolution.
        # t1[m] is the cross-correlation of h and w.
        # t2[m] is the cross-correlation of h and w (reversed?).
        
        # Let's compute grad_h using FFT.
        # break out of loop and do FFT
        pass
    
    # Let's use FFT for the whole gradient computation.
    # t1[m] = sum_i h_i w_{i+m}
    # This is (h * w)[m] ? No.
    # Convolution (h*w)[m] = sum_i h_i w_{m-i}
    # Cross-correlation (h*w')[m] = sum_i h_i w_{m+i} ?
    
    # Let's just use np.correlate
    # np.correlate(h, w, mode='valid')? No, lengths differ.
    
    # Let's just do the O(N^2) loop. It's 160,000 operations.
    # In Python, a simple loop of 160k ops takes ~10ms.
    # L-BFGS-B might do 100 iterations. 1 second. Acceptable.
    
    grad_h = np.zeros(N)
    for m in range(N):
        # t1: sum_i h_i w_{i+m}
        # i from 0 to N-1
        # j = i+m
        # w is length 2N-1
        # j ranges m to m+N-1
        # Since m <= N-1, max j = 2N-2. Valid.
        t1 = np.dot(h, w[m:m+N])
        
        # t2: sum_k h_k w_{m-k}
        # k from 0 to N-1
        # j = m-k
        # j ranges m to m-N+1
        # We need w[j] where j is in [0, 2N-2]
        # Let's construct the slice for w
        # Indices: m, m-1, ..., m-N+1
        # We need to handle negative indices or out of bounds.
        # m >= 0.
        # m-N+1 can be negative.
        
        # Let's use a mask or just sum valid parts.
        # For simplicity, let's use the fact that w is zero outside [0, 2N-2].
        # We can pad w to length 2N to make indexing easier?
        # Or just loop k.
        # t2 = 0
        # for k in range(N):
        #    j = m - k
        #    if 0 <= j < 2*N-1:
        #       t2 += h[k] * w[j]
        
        # Vectorized:
        # k = 0..N-1
        # j = m - k
        # j starts at m, ends at m-N+1
        # We want to sum h[k] * w[m-k]
        
        # Let's create an array of indices j
        # k = np.arange(N)
        # j = m - k
        # valid = (j >= 0) & (j < 2*N-1)
        # t2 = np.dot(h[valid], w[j[valid]])
        
        # This is O(N) per m. Total O(N^2).
        k = np.arange(N)
        j = m - k
        valid = (j >= 0) & (j < 2*N-1)
        t2 = np.dot(h[valid], w[j[valid]])
        
        grad_h[m] = t1 + t2
    
    # grad_h is gradient w.r.t h.
    # But we are optimizing params = h[0:N-1].
    # h[N-1] = 1 - sum(params)
    # d(h[N-1])/d(params_i) = -1
    # d(h[i])/d(params_i) = 1
    # d(h[j])/d(params_i) = 0 for j != i, N-1
    
    # grad_params[i] = d(obj)/d(h_i) * 1 + d(obj)/d(h_{N-1}) * (-1)
    # = grad_h[i] - grad_h[N-1]
    
    grad_params = grad_h[:N-1] - grad_h[N-1]
    
    return max_conv, grad_params

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = np.inf
    
    # Objective wrapper for scipy
    def obj_and_grad(params):
        val, grad = _objective_and_grad(params)
        return val, grad
        
    def obj_only(params):
        h = np.zeros(N)
        h[:N-1] = params
        h[N-1] = 1.0 - np.sum(params)
        conv = np.convolve(h, h)
        s = np.sum(h)
        if s == 0: return np.inf
        return abs(2.0 * N * np.max(conv) / (s*s))

    # Multi-start
    rng = np.random.default_rng(42)
    
    for i in range(10):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        # Ensure sum is 1
        # Start with random normal, normalize
        h0 = rng.normal(0, 1, N)
        h0 = h0 / np.sum(h0)
        
        params0 = h0[:N-1].copy()
        
        try:
            res = minimize(
                obj_and_grad,
                params0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = np.zeros(N)
            h_opt[:N-1] = res.x
            h_opt[N-1] = 1.0 - np.sum(res.x)
            
            val = obj_only(h_opt)
            
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_h = h_opt
                
        except Exception as e:
            pass
            
    # Final check
    final_val = obj_only(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

