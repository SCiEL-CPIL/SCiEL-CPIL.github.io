"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _smooth_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x = np.asarray(x)
    m = np.max(x)
    if m > 1000:
        return m
    e = np.exp(beta * (x - m))
    return m + np.log(np.sum(e)) / beta

def _smooth_abs(x, beta=100.0):
    """Smooth approximation of |x| using log-sum-exp."""
    x = np.asarray(x)
    return _smooth_max(np.concatenate([x, -x]), beta)

def _c3_upper_bound(h, beta=100.0):
    """Compute smooth upper bound on C3."""
    h = np.asarray(h)
    conv = np.convolve(h, h, mode='full')
    # Exclude the main lobe at center
    n = len(h)
    center = n - 1
    # Get sidelobes (exclude center)
    left = conv[:center]
    right = conv[center+1:]
    sidelobes = np.concatenate([left, right])
    
    # Smooth max of absolute values
    peak = _smooth_max(np.concatenate([sidelobes, -sidelobes]), beta)
    
    # L1 norm squared
    l1 = np.sum(np.abs(h))
    l1_sq = l1 * l1
    
    if l1_sq < 1e-10:
        return 1e10
    return peak / l1_sq

def _objective_and_grad(h, beta=100.0):
    """Objective and gradient for L-BFGS-B."""
    h = np.asarray(h)
    conv = np.convolve(h, h, mode='full')
    n = len(h)
    center = n - 1
    left = conv[:center]
    right = conv[center+1:]
    sidelobes = np.concatenate([left, right])
    
    # Compute smooth max gradient
    x = np.concatenate([sidelobes, -sidelobes])
    m = np.max(x)
    e = np.exp(beta * (x - m))
    weights = e / np.sum(e)
    
    # Gradient of smooth max w.r.t. x
    grad_x = weights
    
    # Gradient w.r.t. h: conv = h * h
    # d(conv[k])/d(h[i]) = h[k-i] + h[i] (for valid k)
    # Need to propagate through sidelobes
    
    # Build full gradient
    grad_h = np.zeros(n)
    
    # For each sidelobe position, compute contribution
    # conv[k] = sum_j h[j] * h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] + h[i] (when indices valid)
    
    # Map sidelobe indices to conv indices
    # left sidelobes: conv[0] to conv[center-1]
    # right sidelobes: conv[center+1] to conv[2*center]
    
    # Weighted sum of gradients
    for i in range(len(sidelobes)):
        # Determine conv index
        if i < center:
            conv_idx = i  # left sidelobe
        else:
            conv_idx = i + 1  # right sidelobe (skip center)
        
        w = grad_x[i] if i < len(sidelobes) else grad_x[i - len(sidelobes)]
        
        # d(conv[conv_idx])/d(h[j]) for each j
        for j in range(n):
            k = conv_idx - j
            if 0 <= k < n:
                grad_h[j] += w * (h[k] + h[j])
    
    # L1 norm term
    l1 = np.sum(np.abs(h))
    l1_sq = l1 * l1
    
    if l1_sq < 1e-10:
        return 1e10, np.zeros(n)
    
    peak = m + np.log(np.sum(e)) / beta
    obj = peak / l1_sq
    
    # Gradient of obj = peak / l1_sq
    # d(obj)/d(h) = (grad_peak * l1_sq - peak * d(l1_sq)/d(h)) / l1_sq^2
    
    # d(l1_sq)/d(h[i]) = 2 * l1 * sign(h[i])
    sign_h = np.sign(h)
    sign_h[h == 0] = 1  # subgradient
    dl1_sq = 2 * l1 * sign_h
    
    grad_obj = (grad_h * l1_sq - peak * dl1_sq) / (l1_sq * l1_sq)
    
    return obj, grad_obj

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Initial guess: alternating signs
    h = np.zeros(N)
    h[::2] = 1.0
    h[1::2] = -1.0
    
    best_h = h.copy()
    best_score = _c3_upper_bound(h, beta=50.0)
    
    # Try different initializations and optimizations
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > deadline - 10:
            break
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() - start_time > deadline - 10:
                break
            
            if trial == 0:
                h0 = best_h.copy()
            elif trial == 1:
                # Random with structure
                h0 = np.random.randn(N)
                h0[::2] = np.abs(h0[::2])
                h0[1::2] = -np.abs(h0[1::2])
            else:
                # Block structure
                block_size = 10
                blocks = np.random.choice([-1, 1], size=N // block_size)
                h0 = np.repeat(blocks, block_size)
                if N % block_size != 0:
                    h0 = np.concatenate([h0, np.random.choice([-1, 1], size=N % block_size)])
            
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, beta)[0],
                    h0,
                    jac=lambda x: _objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_h = result.x
            except Exception:
                pass
        
        # Refine with higher beta
        if time.time() - start_time > deadline - 10:
            break
    
    # Final evaluation with moderate beta for stability
    final_score = _c3_upper_bound(best_h, beta=50.0)
    
    return best_h
# EVOLVE-BLOCK-END
