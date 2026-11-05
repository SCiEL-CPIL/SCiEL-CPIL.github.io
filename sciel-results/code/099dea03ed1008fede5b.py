"""
Strategy: High-Dimensional Gradient Descent with Analytic Gradients and Multi-Start
Instead of restricting to a parametric family, we optimize the full 400-dimensional vector h directly.
We derive the analytic gradient of the objective function V(h) = |2*N*max(conv(h,h)) / sum(h)^2|.
We use L-BFGS-B, which is efficient for large-scale optimization, with multiple random restarts
to escape local minima. The key insight is that the autoconvolution max is often dominated by the
zero-lag term (sum h^2) or a specific lag. We can track which lag is the maximum and compute
gradients only with respect to that lag's contribution, treating the max as a fixed index locally.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin (150s limit)

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    """
    Compute the objective value and gradient.
    V = |2 * N * M / S^2| where M = max_k conv[k], S = sum(h).
    Let k* be the index of the max.
    conv[k] = sum_i h[i] * h[i+k] (with appropriate boundary handling, but np.convolve is full).
    Actually, np.convolve(h, h) has length 2N-1.
    conv[k] = sum_{i} h[i] * h[k-i] ? No, standard convolution.
    Let's be precise.
    c = np.convolve(h, h)
    c[k] = sum_{i} h[i] * h[k-i] is not quite right for full convolution indices.
    Let's use the property:
    c[k] = sum_{j} h[j] * h[k-j] is for linear convolution if indices align.
    For np.convolve(h, h), c[k] = sum_{i} h[i] * h[k-i] where h is 0 outside [0, N-1].
    
    Gradient of M = c[k*] with respect to h[m]:
    d c[k*] / d h[m] = h[k*-m] + h[m-k*] (if we consider the symmetry? No, c is symmetric if h is real? No, c is symmetric around N-1).
    Actually, c[k] = sum_i h[i] h[k-i].
    d c[k] / d h[m] = h[k-m] (from term i=m) + h[m-k] (from term i=k-m? No).
    Let's expand: c[k] = h[0]h[k] + h[1]h[k-1] + ...
    d c[k] / d h[m] = h[k-m] + h[m-k] ?
    If we differentiate sum_i h[i] h[k-i] wrt h[m]:
    Term i=m: h[k-m]
    Term i=k-m: h[m] * d(h[k-(k-m)])/dh[m] = h[m] * 1? No, h[k-i] depends on h if i is such that k-i = m => i = k-m.
    So d c[k] / d h[m] = h[k-m] + h[m-k] (if k-m and m-k are within bounds and distinct).
    Wait, h is a vector. c[k] is a scalar.
    c[k] = sum_{i} h[i] h[k-i].
    d c[k] / d h[m] = h[k-m] (derivative of h[m] in the first slot) + h[m-k] (derivative of h[k-m] in the second slot, which is h[m]).
    So grad_c_k[m] = h[k-m] + h[m-k] (with h[j]=0 if j<0 or j>=N).
    
    Objective V = 2*N*M / S^2 (assuming M>0, S>0 for now, abs handles sign).
    Let's assume M > 0 and S > 0 to simplify signs, we can handle signs later or just minimize V^2 or use abs in value but gradient of abs is sign.
    Actually, let's minimize f(h) = 2*N*M / S^2.
    grad f = 2*N * ( (grad M * S^2) - (M * 2*S * grad S) ) / S^4
    grad S = ones(N).
    grad M = grad c[k*].
    
    So grad f = 2*N / S^3 * ( S * grad M - 2*M * ones )
    
    We need to be careful with the abs. If 2*N*M/S^2 is negative, we minimize -f.
    Let val = 2*N*M/S^2.
    We want to minimize |val|.
    If val > 0, grad = grad val.
    If val < 0, grad = -grad val.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    max_idx = np.argmax(conv)
    M = conv[max_idx]
    
    # Gradient of M = conv[max_idx]
    # d conv[max_idx] / d h[m] = h[max_idx - m] + h[m - max_idx]
    # Note: h is defined on [0, N-1].
    # Let's construct the gradient vector efficiently.
    grad_M = np.zeros(N)
    # For each m in 0..N-1:
    # term1: h[max_idx - m] if 0 <= max_idx - m < N
    # term2: h[m - max_idx] if 0 <= m - max_idx < N
    
    # This can be done with slicing or loops. Loops are fine for N=400.
    for m in range(N):
        i1 = max_idx - m
        i2 = m - max_idx
        if 0 <= i1 < N:
            grad_M[m] += h[i1]
        if 0 <= i2 < N:
            grad_M[m] += h[i2]
            
    val = 2.0 * N * M / (s * s)
    
    # Gradient of val
    # val = 2*N * M * s^-2
    # d val / d h = 2*N * (grad_M * s^-2 + M * (-2) * s^-3 * grad_S)
    # grad_S = 1
    grad_val = 2.0 * N * (grad_M / (s * s) - 2.0 * M * ones / (s * s * s))
    # Wait, s*s*s is s^3.
    grad_val = 2.0 * N / (s * s * s) * (s * grad_M - 2.0 * M * np.ones(N))
    
    # Handle absolute value
    if val < 0:
        grad_val = -grad_val
        val = -val
        
    return val, grad_val

def construct():
    """
    Construct step heights by optimizing the full 400-dimensional vector using L-BFGS-B
    with multiple random restarts.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # We will try several restarts.
    # Each restart:
    # 1. Initialize h randomly (or with some structure).
    # 2. Run L-BFGS-B.
    # 3. Check if it improved best_val.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialization strategy:
        # Mix of random noise and low-frequency components to help convergence.
        # Ensure sum is not close to 0.
        # Let's try: h = A * cos(2*pi*f*i/N + phi) + B + noise
        # Or just random Gaussian, then normalize?
        # Normalizing h by a constant doesn't change the value (scale invariance?
        # V(c*h) = |2*N*max(c^2 conv) / (c sum)^2| = |2*N*c^2 max / c^2 sum^2| = V(h).
        # Yes, the objective is scale-invariant.
        # So we can fix the scale, e.g., sum(h) = 1.
        # But L-BFGS-B doesn't enforce constraints easily unless we use bounds or penalty.
        # We can just let it optimize, and the scale invariance means it might wander in scale.
        # To avoid numerical issues with S->0, we can add a small penalty if |S| is small.
        # Or simply restart if S is small.
        
        # Let's initialize with random values in [-1, 1] and add a constant to ensure sum > 0.
        h_init = rng.uniform(-1, 1, N)
        s_init = np.sum(h_init)
        if abs(s_init) < 1e-6:
            h_init += 1.0
        # Scale to have unit sum? Not necessary, but might help.
        # Let's just use it as is.
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_and_grad,
                h_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success and np.isfinite(res.fun):
                h_candidate = res.x
                val_candidate = res.fun
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
        except Exception:
            pass

    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
