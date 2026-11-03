import numpy as np
import time
import warnings

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Try several initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Sparse 2-element
    h2 = np.zeros(N)
    h2[0] = 2.0 / 3.0
    h2[1] = 1.0 / 3.0
    initializations.append(h2)
    
    # Delta
    hd = np.zeros(N)
    hd[0] = 1.0
    initializations.append(hd)
    
    # Sparse alternating
    ha = np.zeros(N)
    for i in range(0, N, 2):
        ha[i] = 1.0
    ha /= np.sum(ha)
    initializations.append(ha)
    
    # Random sparse
    for _ in range(5):
        mask = np.random.rand(N) > 0.7
        if not np.any(mask):
            mask[0] = True
        hr = mask.astype(float)
        hr /= np.sum(hr)
        initializations.append(hr)
    
    # Update best with initializations
    for h0 in initializations:
        r = _compute_ratio(h0)
        if r > best_ratio:
            best_ratio = r
            best_h = h0.copy()
    
    # Gradient-based optimization with log-sum-exp smoothing
    # We want to maximize R(h) = (sum h_i^2) / ((sum h_i) * max_k conv_k)
    # Fix sum h_i = 1 by normalizing
    # Maximize Q(h) = (sum h_i^2) / max_k conv_k
    # Use log-sum-exp to smooth the max
    
    def objective_smooth(h, temperature=0.1):
        """Smoothed objective using log-sum-exp for max"""
        h = np.asarray(h, dtype=float)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        h = h / s
        
        q = np.sum(h**2)
        if q < 1e-15:
            return 1e10, np.zeros(N)
        
        conv = np.convolve(h, h, mode='full')
        
        # Log-sum-exp for max
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 1e10, np.zeros(N)
        
        # Soft max using log-sum-exp
        scaled = conv / temperature
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_max = temperature * log_sum_exp
        
        # Objective: minimize soft_max / q
        obj = soft_max / q
        
        # Gradient w.r.t h
        # d(soft_max)/d(conv_k) = exp(conv_k/temp) / sum(exp(conv_j/temp))
        weights = np.exp(scaled - max_scaled)
        weights /= np.sum(weights)
        
        # d(soft_max)/d h_i = sum_k weights[k] * d(conv_k)/d h_i
        # conv_k = sum_j h_j h_{k-j}
        # d(conv_k)/d h_i = h_{k-i} + h_{i-k} (with boundary checks)
        
        grad_conv = np.zeros(N)
        for k in range(len(conv)):
            if weights[k] < 1e-12:
                continue
            for i in range(N):
                val = 0.0
                idx1 = k - i
                if 0 <= idx1 < N:
                    val += h[idx1]
                idx2 = i - k
                if 0 <= idx2 < N:
                    val += h[idx2]
                grad_conv[i] += weights[k] * val
        
        # d(obj)/d h_i = (d(soft_max)/d h_i * q - soft_max * d(q)/d h_i) / q^2
        # d(q)/d h_i = 2 h_i
        # But we're normalizing, so the gradient is more complex
        # For simplicity, use the unnormalized gradient and normalize after update
        
        dq_dh = 2.0 * h
        grad_obj = (grad_conv * q - soft_max * dq_dh) / (q * q)
        
        return obj, grad_obj
    
    # Multiplicative updates with gradient
    eta = 0.05
    max_iters = 3000
    patience = 100
    stale = 0
    
    # Start from best initialization
    current_h = best_h.copy()
    s = np.sum(current_h)
    if s > 0:
        current_h /= s
    
    current_ratio = _compute_ratio(current_h)
    
    for iteration in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        # Compute objective and gradient
        obj, grad = objective_smooth(current_h, temperature=0.1)
        
        # Multiplicative update: reduce h_i where gradient is positive (increases obj)
        # We want to DECREASE obj, so move against gradient
        # h_i <- h_i * exp(-eta * grad_i)
        
        # Clip gradient to prevent extreme updates
        grad_clipped = np.clip(grad, -10, 10)
        
        update = np.exp(-eta * grad_clipped)
        current_h = current_h * update
        
        # Renormalize
        s = np.sum(current_h)
        if s < 1e-15:
            break
        current_h /= s
        
        # Evaluate
        r_new = _compute_ratio(current_h)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
        else:
            stale += 1
            
            if stale > patience:
                # Try a random restart
                if np.random.rand() < 0.3:
                    mask = np.random.rand(N) > 0.6
                    if not np.any(mask):
                        mask[0] = True
                    current_h = mask.astype(float)
                    current_h /= np.sum(current_h)
                    stale = 0
                else:
                    # Decay eta
                    eta *= 0.9
                    if eta < 1e-5:
                        break
    
    # Final polish: try a few more iterations with smaller eta
    eta_final = 0.01
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
        
        obj, grad = objective_smooth(best_h, temperature=0.05)
        grad_clipped = np.clip(grad, -5, 5)
        update = np.exp(-eta_final * grad_clipped)
        h_new = best_h * update
        s = np.sum(h_new)
        if s < 1e-15:
            break
        h_new /= s
        
        r_new = _compute_ratio(h_new)
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h_new
    
    # Ensure non-negative and return as list
    best_h = np.maximum(best_h, 0.0)
    s = np.sum(best_h)
    if s > 0:
        best_h /= s
    
    return list(best_h)
# EVOLVE-BLOCK-END
