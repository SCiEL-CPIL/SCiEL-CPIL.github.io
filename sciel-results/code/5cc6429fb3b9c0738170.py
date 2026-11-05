"""
Strategy: Direct high-dimensional optimization with smooth surrogate.

The current approach restricts the solution to a 5-parameter polynomial family.
This approach treats the 400 step heights as independent variables.
We use a soft-max surrogate for the maximum of the autoconvolution to make the
objective differentiable, allowing the use of gradient-based optimization (Adam)
and coordinate descent.

The objective is:
    value = abs(2 * N * max(conv(h,h)) / sum(h)**2)

We assume sum(h) > 0 by initializing with positive values and keeping the search
in a region where the sum remains positive (or by adding a large bias if necessary,
though for this specific ratio, scaling invariance means we can just optimize the shape).
Actually, the ratio is scale-invariant:
    value(h) = value(alpha * h) for alpha > 0.
So we can fix sum(h) = 1 during optimization to simplify the gradient, or just
optimize the raw ratio. Optimizing the raw ratio is safer to avoid sum=0.

We will use:
1. Adam optimizer with a soft-max surrogate for max(conv).
2. Coordinate descent for local refinement.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0  # Wall-clock limit with safety margin

def _exact_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_value(h, temp=0.1):
    """
    Compute a smooth surrogate for the task value.
    Uses soft-max for the max of the convolution.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    # Soft max: max(x) ~ temp * log(sum(exp(x/temp)))
    # To avoid overflow, subtract max
    c_max = np.max(conv)
    log_sum_exp = temp * np.log(np.sum(np.exp((conv - c_max) / temp)))
    soft_max_val = c_max + log_sum_exp
    
    val = abs(2.0 * N * soft_max_val / (s * s))
    return val

def _gradient_surrogate(h, temp=0.1):
    """
    Compute the gradient of the surrogate value.
    This is computationally expensive if done naively, but we can use analytic gradients.
    
    Let V = 2*N * M / S^2 where M is soft-max(conv) and S=sum(h).
    dV/dh = 2*N * ( (dM/dh) * S^2 - M * 2*S * dS/dh ) / S^4
          = 2*N * ( (dM/dh) * S - 2*M * dS/dh ) / S^3
    
    dS/dh = 1 (vector of ones)
    dM/dh: M = c_max + temp * log(sum(exp((conv-c_max)/temp)))
           dM/dh_i = sum_j (dM/dconv_j) * (dconv_j/dh_i)
           dM/dconv_j = exp((conv_j - c_max)/temp) / sum(exp((conv_k - c_max)/temp))
           dconv_j/dh_i = h[j-i] + h[j+N-1-i] (from full convolution definition)
    
    This is O(N^2) per gradient step. N=400, so 160k operations. This is feasible.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.zeros(N)
    
    conv = np.convolve(h, h)
    c_max = np.max(conv)
    exp_conv = np.exp((conv - c_max) / temp)
    sum_exp = np.sum(exp_conv)
    
    # Probability vector for soft-max
    p = exp_conv / sum_exp
    
    # dM/dconv = p
    # dconv_j/dh_i
    # conv[j] = sum_k h[k] h[j-k]
    # dconv_j/dh_i = h[j-i] (if 0<=j-i<N) + h[j-(i+N)]? No, indices are 0..2N-1.
    # conv[j] = sum_{k} h[k] h[j-k] where h is 0 outside [0, N-1].
    # dconv_j/dh_i = h[j-i] + h[j-i+N]? No.
    # Let's use the property: conv = h * h.
    # d(conv)_j = sum_k (dh_k h_{j-k} + h_k dh_{j-k})
    # For a specific i, d(conv)_j/dh_i = h_{j-i} + h_{j-i+N}? 
    # Actually, simpler:
    # d(conv)_j/dh_i = h[(j-i) % N] is not quite right for linear conv.
    # Let's just compute the gradient via finite differences or a more direct formula?
    # Finite difference is O(N) evaluations of O(N^2) conv -> O(N^3) = 64M ops. Too slow for 150s?
    # 64M ops in numpy is fast (<0.1s). So we can use finite differences if analytic is tricky.
    # But let's try analytic.
    
    # d(conv)_j/dh_i = h[j-i] if 0 <= j-i < N else 0
    #                 + h[j-(i+N)]? No.
    # conv[j] = sum_{k=0}^{N-1} h[k] h[j-k]
    # Term involving h[i]:
    # 1. k=i: h[i] h[j-i]
    # 2. j-k=i => k=j-i: h[j-i] h[i]
    # So d(conv)_j/dh_i = h[j-i] (if 0<=j-i<N) + h[j-i] (if 0<=j-i<N)
    # Wait, if j-i is in [0, N-1], both terms are present?
    # If j-i is in [0, N-1], then k=i and k=j-i are distinct indices in the sum unless i = j-i.
    # If i != j-i, d(conv)_j/dh_i = 2 * h[j-i].
    # If i == j-i (j=2i), d(conv)_j/dh_i = h[i] (derivative of h[i]^2 is 2h[i], but here it's one term? No, sum has h[i]h[i] only once? No, sum is over k. If k=i, term is h[i]h[j-i]=h[i]h[i]. Is there another k? No, k is unique. So derivative is 2h[i].)
    # Actually, d/dh_i (h[i]h[j-i]) = h[j-i] + h[i] (if j-i != i). If j-i=i, d/dh_i (h[i]^2) = 2h[i].
    # So d(conv)_j/dh_i = 2 * h[j-i] if 0 <= j-i < N.
    # Wait, conv is symmetric? Yes.
    
    # Let's use a matrix approach or just compute dM/dh directly.
    # dM/dh_i = sum_j p_j * (2 * h[j-i] if 0<=j-i<N else 0)
    
    # This is a convolution of p with h (scaled by 2)?
    # Let q = 2*h.
    # dM/dh_i = sum_j p_j q[j-i]
    # This is (p * q)[i] ?
    # (p * q)[i] = sum_j p[j] q[i-j].
    # We have sum_j p[j] h[j-i].
    # Let k = j-i => j = k+i.
    # sum_k p[k+i] h[k].
    # This is cross-correlation?
    # Let's just compute it with np.convolve.
    # dM/dh = np.convolve(p, 2*h, mode='same')?
    # Let's test with small N.
    
    # Actually, simpler:
    # grad_M = np.zeros(N)
    # for i in range(N):
    #    # Indices j such that 0 <= j-i < N => i <= j < i+N
    #    js = np.arange(i, i+N)
    #    grad_M[i] = 2.0 * np.dot(p[js], h[js-i])
    
    # This is O(N^2). 160k ops. Fast.
    
    grad_M = np.zeros(N)
    for i in range(N):
        js = np.arange(i, i + N)
        grad_M[i] = 2.0 * np.dot(p[js], h[js - i])
        
    # Now combine
    # V = 2*N*M / S^2
    # dV/dh_i = 2*N * ( S^2 * dM_i - M * 2*S * 1 ) / S^4
    #         = 2*N * ( S * dM_i - 2*M ) / S^3
    
    M = c_max + temp * np.log(sum_exp)
    dV_dh = 2.0 * N * (s * grad_M - 2.0 * M) / (s**3)
    
    return dV_dh

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initialize with ones
    h = np.ones(N)
    best_h = h.copy()
    best_val = _exact_value(best_h)
    
    # Phase 1: Adam Optimization
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    
    # We will run Adam for a fixed number of steps or until time runs out
    # Each step is O(N^2). 160k ops. 1000 steps = 160M ops. ~1-2 seconds in Python/NumPy.
    # We can afford many steps.
    
    max_steps = 5000
    for step in range(max_steps):
        if time.time() > deadline - 20:
            break
            
        val = _surrogate_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        grad = _gradient_surrogate(h)
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        h = h - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Optional: enforce positivity? 
        # If h becomes negative, sum might drop or conv max might increase.
        # The problem allows negative heights.
        # However, if sum(h) approaches 0, value explodes.
        # We can add a small penalty if |sum(h)| is small.
        s = np.sum(h)
        if abs(s) < 1e-3:
             h += np.sign(s) * 0.01 * np.ones(N)
             
    # Phase 2: Coordinate Descent Refinement
    # Try small perturbations to each coordinate
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        # Shuffle indices
        idx_order = rng.permutation(N)
        for _ in range(3): # Passes
            if time.time() > deadline - 5:
                break
            for idx in idx_order:
                if time.time() > deadline - 5:
                    break
                orig_val = h[idx]
                # Try a few steps
                for delta in [-0.1, -0.01, 0.01, 0.1]:
                    h[idx] = orig_val + delta
                    val = _exact_value(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
                    else:
                        h[idx] = orig_val
                # Also try random small change
                delta = rng.normal(0, 0.05)
                h[idx] = orig_val + delta
                val = _exact_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                else:
                    h[idx] = orig_val

    # Final check
    final_val = _exact_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

if __name__ == "__main__":
    construct()
