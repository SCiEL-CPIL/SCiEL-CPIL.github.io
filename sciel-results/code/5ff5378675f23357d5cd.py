import numpy as np
import time
import sys

N = 400

def _evaluate(h):
    """Evaluate the objective: abs(2*N*max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta=100.0):
    """Soft maximum approximation."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)) / len(x)) / beta

def _objective_soft(h_flat, beta=100.0):
    h = h_flat.reshape(N)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = 1e10
    
    # Initial candidates
    candidates = []
    
    # 1. Constant vector
    candidates.append(np.ones(N))
    
    # 2. Random vectors
    for _ in range(10):
        h_rand = rng.standard_normal(N)
        # Ensure sum is not zero
        if abs(np.sum(h_rand)) < 1e-5:
            h_rand += 1.0
        candidates.append(h_rand)
        
    # 3. Structured vectors (sines)
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        h_sin = np.sin(2 * np.pi * freq * t / N)
        # Add DC offset to ensure sum is not zero and positive
        h_sin += 0.5
        candidates.append(h_sin)
        
    # Evaluate initial candidates
    for h in candidates:
        val = _evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
        
    # Optimization using L-BFGS-B with soft-max
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
        
    if use_scipy:
        # We optimize the soft-max objective
        # We need to handle the absolute value and the scaling invariance.
        # To break scaling invariance, we can fix sum(h) = 1.
        # But the objective is scale invariant, so gradients might be zero or ill-conditioned.
        # Better: optimize h directly, but the landscape is flat along the scaling direction.
        # L-BFGS-B might struggle.
        # We can fix one variable, e.g., h[0] = 1, or sum(h) = 1.
        
        # Let's try fixing sum(h) = 1 by projecting.
        # Or just let L-BFGS-B run, it might find a local min.
        
        def obj_wrapper(h_flat):
            return _objective_soft(h_flat, beta=50.0)
            
        # Try multiple starting points
        for i in range(5):
            if time.time() > deadline - 10.0:
                break
                
            if i == 0:
                h0 = best_h.copy()
            else:
                h0 = best_h + rng.standard_normal(N) * 0.1 * np.std(best_h)
                
            try:
                result = minimize(
                    obj_wrapper,
                    h0.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                h_opt = result.x.reshape(N)
                val_opt = _evaluate(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
                
    # Custom Gradient Descent with Soft-Max
    if time.time() < deadline - 20.0:
        h_curr = best_h.copy()
        lr = 0.01
        momentum = np.zeros(N)
        beta_m = 0.9
        beta_soft = 50.0
        
        for iteration in range(5000):
            if time.time() > deadline - 5.0:
                break
                
            s = np.sum(h_curr)
            if abs(s) < 1e-10:
                break
                
            conv = np.convolve(h_curr, h_curr)
            m = np.max(conv)
            
            # Soft max gradient
            # exp((conv - m) * beta) / sum(exp((conv - m) * beta))
            # To avoid overflow, subtract max
            conv_shifted = conv - m
            weights = np.exp(conv_shifted * beta_soft)
            weights /= np.sum(weights)
            
            # Gradient of soft max w.r.t. conv is weights
            # Gradient of conv w.r.t. h is the Toeplitz matrix of h (symmetric)
            # We can compute this via FFT or direct convolution of weights with h?
            # d(soft_max)/d(h) = sum_k weights[k] * d(conv[k])/d(h)
            # d(conv[k])/d(h) is the k-th shift of h (circularly? no, full conv).
            # This is equivalent to convolving h with the "gradient of max" which is a sparse vector?
            # No, it's a weighted sum of shifts of h.
            # This is equivalent to convolving h with a kernel derived from weights?
            # Actually, d(conv)/dh is a linear operator.
            # Let's approximate the gradient by finite differences or use the fact that
            # the gradient of the max is the subgradient at the max index.
            # For soft max, it's a weighted average of the subgradients.
            
            # Simpler: Use the hard max gradient (subgradient)
            max_idx = np.argmax(conv)
            shift = max_idx - N + 1
            
            grad_conv = np.zeros(N)
            for j in range(N):
                val = 0.0
                if 0 <= j - shift < N:
                    val += h_curr[j - shift]
                if 0 <= j + shift < N:
                    val += h_curr[j + shift]
                grad_conv[j] = val
            
            # Objective gradient
            max_conv = conv[max_idx]
            grad_obj = (2.0 * N / (s * s)) * (grad_conv - 2.0 * max_conv)
            
            # Normalize
            grad_norm = np.linalg.norm(grad_obj)
            if grad_norm > 1e-10:
                grad_obj = grad_obj / grad_norm
            
            # Momentum
            momentum = beta_m * momentum + (1 - beta_m) * grad_obj
            h_curr = h_curr - lr * momentum
            
            # Project sum away from 0
            s_new = np.sum(h_curr)
            if abs(s_new) < 0.1:
                h_curr += (0.1 - s_new) / N
                
            val_new = _evaluate(h_curr)
            if val_new < best_val:
                best_val = val_new
                best_h = h_curr.copy()
                lr *= 1.05
            else:
                lr *= 0.95
                
            if val_new > best_val * 1.5:
                h_curr = best_h + rng.standard_normal(N) * 0.05 * np.std(best_h)
                momentum = np.zeros(N)
                lr = 0.01
                
    # Final check
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        
    return best_h
