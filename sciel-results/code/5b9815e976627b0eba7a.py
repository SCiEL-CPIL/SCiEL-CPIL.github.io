import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0

def _compute_score(heights):
    """Compute the normalized max autoconvolution: max_k conv[k] / (sum h)^2"""
    h = np.array(heights, dtype=np.float64)
    s = h.sum()
    if s <= 0:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return conv.max() / (s * s)

def _smooth_max(conv, alpha):
    """Smooth approximation of max(conv)."""
    # Use log-sum-exp for numerical stability
    max_val = np.max(conv)
    if max_val == 0:
        return 0.0
    # conv is non-negative
    shifted = conv - max_val
    # alpha * shifted
    # log(sum(exp(alpha * shifted))) / alpha + max_val
    # To avoid overflow, we use the standard log-sum-exp trick
    # But since conv >= 0, max_val >= 0.
    # If alpha is large, exp(alpha * shifted) might underflow for small shifted.
    # This is fine.
    return max_val + np.log(np.sum(np.exp(alpha * shifted))) / alpha

def _objective_and_grad(h, alpha):
    """
    Objective: smooth_max(conv(h)) / (sum(h))^2
    Gradient computed via chain rule or finite differences if needed.
    For SLSQP, we can use finite differences or provide analytic gradient.
    Given N=600, finite differences might be slow (600 evals per step).
    Analytic gradient is better.
    """
    s = h.sum()
    conv = np.convolve(h, h, mode='full')
    
    # Smooth max
    max_val = np.max(conv)
    shifted = conv - max_val
    exp_shifted = np.exp(alpha * shifted)
    sum_exp = np.sum(exp_shifted)
    smooth_max_val = max_val + np.log(sum_exp) / alpha
    
    # Gradient of smooth_max_val w.r.t conv_k
    # d(smooth_max)/d(conv_k) = exp(alpha * (conv_k - max_val)) / sum_exp
    # Note: if max_val changes, the derivative is more complex, but log-sum-exp is differentiable.
    # Actually, the gradient of log(sum(exp(alpha*x_k))) is alpha * exp(alpha*x_k) / sum(exp(alpha*x_k)).
    # The max_val term in the expression `max_val + log(sum(exp(alpha*(conv-max_val))))/alpha`
    # is just for stability. The gradient is:
    # d/d conv_k [ max_val + 1/alpha log(sum_j exp(alpha(conv_j - max_val))) ]
    # = 1 + 1/alpha * sum_j [ exp(alpha(conv_j - max_val)) * alpha * (delta_jk - delta_kk) ] / sum_j exp(...)
    # This is messy.
    # Simpler: Just use log(sum(exp(alpha*conv_k))) / alpha.
    # This is numerically unstable if conv_k is large.
    # Let's use the stable form but compute gradient carefully.
    # Actually, for optimization, we can just use the value and let SLSQP use finite differences?
    # N=600. 600 convolutions per gradient eval.
    # Convolution is O(N^2) = 360,000 ops.
    # 600 * 360,000 = 216,000,000 ops per gradient.
    # This might be too slow for many iterations.
    
    # Better: Analytic gradient.
    # Let F = log(sum_k exp(alpha * conv_k)) / alpha
    # dF/dh_i = 1/alpha * sum_k [ exp(alpha * conv_k) / sum_j exp(alpha * conv_j) * d(conv_k)/dh_i ]
    # d(conv_k)/dh_i = h_{i-k} (with appropriate bounds)
    # This is a convolution of the weights with h.
    
    # Let's compute weights W_k = exp(alpha * conv_k) / sum_j exp(alpha * conv_j)
    # To avoid overflow, subtract max.
    max_c = np.max(conv)
    shifted_c = conv - max_c
    exp_c = np.exp(alpha * shifted_c)
    sum_exp_c = np.sum(exp_c)
    W = exp_c / sum_exp_c
    
    # grad_F_i = 1/alpha * sum_k W_k * h_{i-k}
    # This is a correlation/convolution of W and h.
    # W has length 2N-1. h has length N.
    # The sum is over k.
    # Let's compute grad_F = np.convolve(W, h, mode='same')? No, indices are tricky.
    # conv_k = sum_j h_j h_{j+k}
    # d(conv_k)/dh_i = h_{i-k} + h_{i+k} (if symmetric) ?
    # No, conv_k = sum_j h_j h_{j+k}.
    # If we vary h_i, it appears in terms where j=i or j+k=i.
    # Term j=i: h_i h_{i+k}. Deriv w.r.t h_i is h_{i+k}.
    # Term j+k=i => j=i-k: h_{i-k} h_i. Deriv w.r.t h_i is h_{i-k}.
    # So d(conv_k)/dh_i = h_{i+k} + h_{i-k} (with h_out=0).
    
    # We need G_i = sum_k W_k * (h_{i+k} + h_{i-k})
    # This is a linear operator.
    
    # For efficiency, we can use FFT.
    # But for N=600, direct computation might be okay if iterations are few.
    # Let's try to implement the gradient using FFT.
    
    # Pad h to size 2N
    h_pad = np.zeros(2*N)
    h_pad[:N] = h
    
    # W is defined on lags -(N-1) to N-1.
    # We need to compute sum_k W_k h_{i+k} and sum_k W_k h_{i-k}.
    # This is a convolution.
    
    # Let's just use finite differences if analytic is too complex to code quickly.
    # But SLSQP with finite diff on 600 vars might be slow.
    
    # Alternative: Use Mirror Descent (Multiplicative Weights) which doesn't need gradient?
    # No, it needs the subgradient.
    
    # Let's stick to a simpler heuristic: 
    # 1. Uniform
    # 2. Cosine taper
    # 3. Random restarts with local search (gradient descent on smooth max)
    
    # Given the time, I will implement a robust Gradient Descent with analytic gradient using FFT.
    
    # Gradient of smooth max part:
    # grad_sm = (1/alpha) * (W * h)_i where * is the appropriate correlation.
    
    # Let's compute grad_sm via FFT.
    # W_k is defined for k in [-(N-1), N-1].
    # We want G_i = sum_k W_k h_{i+k} + sum_k W_k h_{i-k}
    # Note W is symmetric? conv is symmetric. So W is symmetric. W_k = W_{-k}.
    # So G_i = 2 * sum_k W_k h_{i+k} (for k>0) + W_0 h_i.
    
    # Let's build W_full of size 2N-1.
    # W_full[k] corresponds to lag k-(N-1).
    
    # Actually, simpler:
    # Use the fact that d(conv)/dh is a linear operator.
    
    # For the purpose of this rewrite, I will use `scipy.optimize.minimize` with `SLSQP` 
    # and provide the analytic gradient using FFT.
    
    # Compute W
    # conv length 2N-1.
    # W = exp(alpha * (conv - max_c)) / sum_exp_c
    
    # Compute grad_sm
    # G = np.zeros(N)
    # for i in range(N):
    #    for k in range(-N+1, N):
    #       if 0 <= i+k < N:
    #          G[i] += W[k+N-1] * h[i+k]
    # This is O(N^2). 360,000 ops. Fast.
    
    G = np.zeros(N)
    # Vectorized version
    # We can use np.convolve on W and h?
    # W is centered.
    # Let's create W_padded of size 2N-1.
    # convolve(W, h, mode='valid')? No.
    
    # Let's just do the O(N^2) loop in numpy.
    # W_idx = np.arange(-N+1, N)
    # For each i, we need h[i+W_idx] where valid.
    
    # Faster:
    # G = np.convolve(W, h, mode='full')?
    # W has length 2N-1. h has length N.
    # Full conv length 3N-2.
    # The relevant part is the middle N elements?
    
    # Let's test this logic.
    # d(conv_k)/dh_i = h_{i+k} + h_{i-k}.
    # Sum_k W_k h_{i+k}.
    # This is the cross-correlation of W and h.
    
    # Let's use np.correlate.
    # np.correlate(h, W, mode='full')
    
    # Given time constraints, I will use a simpler approach:
    # 1. Uniform
    # 2. Try a few parametric families (Cosine, Gaussian)
    # 3. Use `scipy.optimize.minimize` with `Nelder-Mead` or `Powell` on a reduced set of parameters?
    # No, N=600 is too high for Nelder-Mead.
    
    # I will use the analytic gradient with O(N^2) computation.
    
    # Gradient of objective J = F / s^2
    # dJ = (s^2 dF - F * 2s * ds) / s^4 = (dF * s - 2F * ds) / s^3
    # ds = sum(dh) = N (if we project?) No, ds = sum(dh_i).
    # dF = dot(grad_sm, dh)
    # So grad_J_i = (s * grad_sm_i - 2F * 1) / s^3
    # Wait, ds/dh_i = 1.
    # So grad_J_i = (s * grad_sm_i - 2F) / s^3
    
    # We must project onto simplex (sum h=1, h>=0).
    # SLSQP handles constraints.
    
    pass # Placeholder for actual implementation in the final code

def construct():
    start_time = time.time()
    
    # 1. Uniform
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_score = _compute_score(best_h)
    
    # 2. Try Cosine Taper
    i = np.arange(N)
    x = 2.0 * i / (N - 1) - 1.0
    h_cos = np.maximum(0.0, 1.0 - 0.5 * x * x)
    if h_cos.sum() > 0:
        h_cos = h_cos / h_cos.sum()
        s_cos = _compute_score(h_cos)
        if s_cos < best_score:
            best_score = s_cos
            best_h = h_cos
    
    # 3. Gradient Descent on Smooth Max
    # Use SLSQP
    alpha = 50.0 # Smoothness parameter
    
    def obj(h):
        s = h.sum()
        if s <= 0: return np.inf
        conv = np.convolve(h, h, mode='full')
        sm = _smooth_max(conv, alpha)
        return sm / (s * s)
        
    def grad(h):
        s = h.sum()
        conv = np.convolve(h, h, mode='full')
        sm = _smooth_max(conv, alpha)
        
        # Compute grad_sm
        max_c = np.max(conv)
        shifted_c = conv - max_c
        exp_c = np.exp(alpha * shifted_c)
        sum_exp_c = np.sum(exp_c)
        W = exp_c / sum_exp_c
        
        # Compute G = sum_k W_k (h_{i+k} + h_{i-k})
        # Using O(N^2) loop for simplicity and correctness
        G = np.zeros(N)
        # W is indexed 0 to 2N-2 corresponding to lags -(N-1) to N-1
        for i in range(N):
            for k in range(-N+1, N):
                idx_w = k + N - 1
                if idx_w < 0 or idx_w >= len(W): continue
                w_val = W[idx_w]
                # h_{i+k}
                j1 = i + k
                if 0 <= j1 < N:
                    G[i] += w_val * h[j1]
                # h_{i-k}
                j2 = i - k
                if 0 <= j2 < N:
                    G[i] += w_val * h[j2]
        
        grad_sm = G / alpha
        grad_obj = (s * grad_sm - 2 * sm) / (s * s * s)
        
        return grad_obj

    constraints = [
        {'type': 'eq', 'fun': lambda h: h.sum() - 1},
        {'type': 'ineq', 'fun': lambda h: h} # h >= 0
    ]
    
    # Run optimization
    try:
        res = minimize(obj, best_h, method='SLSQP', jac=grad, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if res.success and res.fun < best_score:
            best_h = res.x
            best_score = res.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE:
        # Try Mirror Descent (Multiplicative Weights)
        # h_{t+1} = h_t * exp(-eta * grad)
        # Normalize
        h = best_h.copy()
        eta = 0.1
        for _ in range(100):
            if time.time() - start_time > DEADLINE: break
            g = grad(h)
            h = h * np.exp(-eta * g)
            h = np.maximum(h, 1e-10)
            h = h / h.sum()
            s = _compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
            else:
                # Decay eta
                eta *= 0.9
    
    best_h = np.maximum(0.0, best_h)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print(f"Constructed step function. Score: {_compute_score(res)}")
