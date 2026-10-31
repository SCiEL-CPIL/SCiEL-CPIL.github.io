import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute grid
    x = np.arange(N)
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    # Try different numbers of Fourier modes
    for K in [3, 5, 8, 12]:
        if time.time() > deadline - 20:
            break
            
        # Basis functions
        # h = 0.5 + sum(c_k cos + s_k sin)
        # We use L-BFGS-B on the coefficients
        n_params = 2 * K
        
        def get_h(params):
            h = np.full(N, 0.5)
            for k in range(1, K + 1):
                c_k = params[2*(k-1)]
                s_k = params[2*(k-1)+1]
                h += c_k * np.cos(2 * np.pi * k * x / N)
                h += s_k * np.sin(2 * np.pi * k * x / N)
            return h

        def objective(params):
            h = get_h(params)
            # Penalize out-of-bounds to guide optimizer back, or just clip
            # Clipping is non-differentiable, but L-BFGS-B uses gradients.
            # Let's use a smooth penalty for bounds to keep it differentiable.
            # Or, simpler: just clip and accept the kink, L-BFGS-B often handles it okay if we are not too aggressive.
            # Better: use a soft clip / sigmoid mapping? No, mean constraint is broken.
            # Let's stick to linear penalty for bounds in the objective, but rely on the fact that optimal h is likely in [0,1].
            
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            # Add penalty for deviation from [0,1]
            penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            
            return _smooth_max(overlap, alpha=200.0) + penalty

        def objective_grad(params):
            h = get_h(params)
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            # Gradient of smooth max
            x_max = np.max(overlap)
            e = np.exp(200.0 * (overlap - x_max))
            p = e / np.sum(e)
            
            # Gradient of overlap w.r.t h_clipped
            # overlap[i] = h[i](1-h[i+1])
            # d/dh[i] = (1-h[i+1]) - h[i-1] (with h[-1]=0, h[N]=0)
            # Note: h_clipped is a function of h.
            # If h is in [0,1], d h_clipped/d h = 1. Else 0.
            # Let's approximate d h_clipped/d h as 1 where in bounds, 0 otherwise.
            mask = (h >= 0.0) & (h <= 1.0)
            
            grad_h = np.zeros(N)
            # term from overlap[i] to h[i]: (1 - h_clipped[i+1])
            h_next = np.roll(h_clipped, -1)
            h_next[-1] = 0.0
            # term from overlap[i-1] to h[i]: -h_clipped[i-1]
            h_prev = np.roll(h_clipped, 1)
            h_prev[0] = 0.0
            
            grad_h = p * (1.0 - h_next) - np.roll(p, 1) * h_prev
            # Mask out gradients where h is out of bounds (subgradient of clip is 0)
            grad_h *= mask
            
            # Gradient of penalty
            # penalty = 100 * (sum(max(0, h-1)^2) + sum(max(0, -h)^2))
            # d/dh penalty = 100 * (2*(h-1) * I(h>1) + 2*(-h) * I(h<0))
            grad_penalty = np.zeros(N)
            grad_penalty[h > 1.0] = 200.0 * (h[h > 1.0] - 1.0)
            grad_penalty[h < 0.0] = -200.0 * h[h < 0.0]
            
            total_grad_h = grad_h + grad_penalty
            
            # Now chain rule: grad_params = sum_j (d h_j / d param_k) * total_grad_h[j]
            # h_j = 0.5 + sum c_k cos + s_k sin
            # d h_j / d c_k = cos(2 pi k j / N)
            # d h_j / d s_k = sin(2 pi k j / N)
            
            grad_params = np.zeros(n_params)
            for k in range(1, K + 1):
                cos_k = np.cos(2 * np.pi * k * x / N)
                sin_k = np.sin(2 * np.pi * k * x / N)
                grad_params[2*(k-1)] = np.dot(total_grad_h, cos_k)
                grad_params[2*(k-1)+1] = np.dot(total_grad_h, sin_k)
                
            return grad_params

        # Initial guess: small random coefficients
        np.random.seed(42)
        h0 = np.random.randn(n_params) * 0.1
        
        try:
            res = minimize(
                objective,
                h0,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            
            h_candidate = get_h(res.x)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            # Ensure sum is N/2 (it should be, but clipping might break it slightly? No, sum of cos/sin is 0.
            # Wait, if we clip, the sum is no longer N/2.
            # The task requires sum EXACTLY len/2.
            # So we must re-project to sum N/2 after clipping.
            
            # Re-project to sum N/2
            diff = N/2.0 - np.sum(h_candidate)
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            # If clipping again changes sum, we might need iterative projection
            for _ in range(10):
                diff = N/2.0 - np.sum(h_candidate)
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
                
        except Exception as e:
            pass

    return h_best
