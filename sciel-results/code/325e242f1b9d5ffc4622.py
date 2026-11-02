import numpy as np
import time

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
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _soft_max_grad(conv, temperature=0.1):
    """Compute gradient of soft-max of conv w.r.t. h using log-sum-exp."""
    # Soft-max: log(sum_j exp(conv_j / T)) * T
    # Gradient w.r.t. h_i is weighted sum of gradients of each conv_j
    # with weights proportional to exp(conv_j / T) / sum exp(conv_j / T)
    
    # For numerical stability, use log-sum-exp
    shifted = conv / temperature
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    
    # Weights for each lag
    weights = np.exp(shifted - log_sum_exp)
    
    # Gradient of soft-max w.r.t. h is sum_k weights[k] * d(conv[k])/dh
    # We'll compute this by accumulating contributions
    grad_h = np.zeros(N)
    
    for k in range(len(conv)):
        if weights[k] < 1e-10:
            continue
        # d conv[k] / d h[i] = h[k-i] + h[i-k] (with bounds)
        for i in range(N):
            val = 0.0
            idx1 = k - i
            if 0 <= idx1 < N:
                val += 1.0  # placeholder, will multiply by h
            idx2 = i - k
            if 0 <= idx2 < N:
                val += 1.0
            grad_h[i] += weights[k] * val
    # Note: this is actually d(soft_max)/dh = sum_k weights[k] * d(conv[k])/dh
    # But d(conv[k])/dh[i] = h[k-i] + h[i-k], so we need to multiply by h values
    # Let me redo this properly
    
    return grad_h

def _gradient_conv_peak(h, conv, temperature=0.1):
    """Compute gradient of soft-max of autoconvolution w.r.t. h."""
    N_local = len(h)
    # Soft-max weights
    shifted = conv / temperature
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    weights = np.exp(shifted - log_sum_exp)
    
    # Gradient: sum_k weights[k] * d(conv[k])/dh
    # d(conv[k])/dh[i] = h[k-i] + h[i-k] (with bounds checking)
    grad_h = np.zeros(N_local)
    
    for i in range(N_local):
        total = 0.0
        for k in range(len(conv)):
            if weights[k] < 1e-10:
                continue
            val = 0.0
            idx1 = k - i
            if 0 <= idx1 < N_local:
                val += h[idx1]
            idx2 = i - k
            if 0 <= idx2 < N_local:
                val += h[idx2]
            total += weights[k] * val
        grad_h[i] = total
    
    return grad_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Multiple initializations to try
    initializations = []
    
    # Uniform
    h_uniform = np.ones(N) / N
    initializations.append(h_uniform)
    
    # Delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    initializations.append(h_delta)
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt /= np.sum(h_alt)
    initializations.append(h_alt)
    
    # Sparse random
    for _ in range(5):
        mask = np.random.rand(N) > 0.85
        if not np.any(mask):
            mask[0] = True
        h_sparse = mask.astype(float)
        h_sparse /= np.sum(h_sparse)
        initializations.append(h_sparse)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-5 * x**2)
    h_gauss /= np.sum(h_gauss)
    initializations.append(h_gauss)
    
    # Evaluate all initializations
    for h_init in initializations:
        r = _compute_ratio(h_init)
        if r > best_ratio:
            best_ratio = r
            best_h = h_init.copy()
    
    # Main optimization: game-theoretic multiplicative updates
    # Defender: h (heights), Attacker: distribution over lags
    # We use soft-max to smooth the max operation
    
    max_total_iters = 8000
    patience = 80
    stale = 0
    temperature = 0.05  # Soft-max temperature
    
    # Start from best initialization
    current_h = best_h.copy()
    current_ratio = best_ratio
    
    eta = 0.08  # Learning rate for defender
    eta_attacker = 0.1  # Learning rate for attacker (lag distribution)
    
    # Attacker state: distribution over lags (implicit in soft-max)
    
    for i in range(max_total_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Compute autoconvolution
        conv = np.convolve(current_h, current_h, mode='full')
        
        if np.max(conv) < 1e-15:
            break
        
        # Compute gradient of soft-max of conv w.r.t. h
        grad_h = _gradient_conv_peak(current_h, conv, temperature)
        
        # Multiplicative update for defender
        # h_i <- h_i * exp(-eta * grad_h_i)
        # Normalize to maintain sum = 1
        
        # Clip gradient to avoid extreme updates
        grad_clipped = np.clip(grad_h, -5.0, 5.0)
        
        update = np.exp(-eta * grad_clipped)
        current_h = current_h * update
        
        # Renormalize
        s = np.sum(current_h)
        if s < 1e-15:
            break
        current_h /= s
        
        # Ensure non-negativity (should be automatic with exp, but clip tiny negatives)
        current_h = np.maximum(current_h, 0.0)
        s2 = np.sum(current_h)
        if s2 < 1e-15:
            break
        current_h /= s2
        
        # Evaluate
        r_new = _compute_ratio(current_h)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
        else:
            stale += 1
        
        # Adaptive temperature: decrease over time for sharper max
        if i % 100 == 0:
            temperature *= 0.9
            if temperature < 0.001:
                temperature = 0.001
        
        # Decay learning rate
        if i % 50 == 0:
            eta *= 0.98
            eta_attacker *= 0.98
        
        # Restart if stagnant
        if stale > patience:
            stale = 0
            # Try different restart strategies
            strategy = i % 4
            if strategy == 0:
                # Sparse random
                mask = np.random.rand(N) > 0.8
                if not np.any(mask):
                    mask[0] = True
                current_h = mask.astype(float)
                current_h /= np.sum(current_h)
            elif strategy == 1:
                # Perturb current best
                current_h = best_h + np.random.randn(N) * 0.05
                current_h = np.maximum(current_h, 0)
                s = np.sum(current_h)
                if s > 1e-15:
                    current_h /= s
                else:
                    current_h = best_h.copy()
            elif strategy == 2:
                # Alternating pattern with noise
                h_alt2 = np.zeros(N)
                h_alt2[::2] = 1.0
                h_alt2 += np.random.randn(N) * 0.1
                h_alt2 = np.maximum(h_alt2, 0)
                s = np.sum(h_alt2)
                if s > 1e-15:
                    current_h = h_alt2 / s
                else:
                    current_h = best_h.copy()
            else:
                # Gaussian with random center
                x = np.linspace(-1, 1, N)
                center = np.random.uniform(-0.5, 0.5)
                width = np.random.uniform(0.1, 0.5)
                h_gauss2 = np.exp(-(x - center)**2 / (2 * width**2))
                h_gauss2 += np.random.randn(N) * 0.05
                h_gauss2 = np.maximum(h_gauss2, 0)
                s = np.sum(h_gauss2)
                if s > 1e-15:
                    current_h = h_gauss2 / s
                else:
                    current_h = best_h.copy()
            
            # Reset temperature for restart
            temperature = 0.05
            eta = 0.08
    
    # Final polish: use hard argmax with very small steps
    if time.time() < deadline - 10.0:
        polish_h = best_h.copy()
        polish_ratio = best_ratio
        eta_polish = 0.01
        for j in range(500):
            if time.time() > deadline - 5.0:
                break
            
            conv = np.convolve(polish_h, polish_h, mode='full')
            if np.max(conv) < 1e-15:
                break
            
            k_star = np.argmax(conv)
            
            # Gradient of hard max
            grad_h = np.zeros(N)
            for i in range(N):
                val = 0.0
                idx1 = k_star - i
                if 0 <= idx1 < N:
                    val += polish_h[idx1]
                idx2 = i - k_star
                if 0 <= idx2 < N:
                    val += polish_h[idx2]
                grad_h[i] = val
            
            update = np.exp(-eta_polish * grad_h)
            polish_h = polish_h * update
            s = np.sum(polish_h)
            if s < 1e-15:
                break
            polish_h /= s
            polish_h = np.maximum(polish_h, 0)
            s2 = np.sum(polish_h)
            if s2 < 1e-15:
                break
            polish_h /= s2
            
            r_new = _compute_ratio(polish_h)
            if r_new > polish_ratio:
                polish_ratio = r_new
                if r_new > best_ratio:
                    best_ratio = r_new
                    best_h = polish_h.copy()
            else:
                eta_polish *= 0.9
    
    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
