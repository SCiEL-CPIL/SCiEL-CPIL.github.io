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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    
    # Update the best if uniform is good
    r = _compute_ratio(h)
    if r > best_ratio:
        best_ratio = r
        best_h = h.copy()
        
    # Also check delta as baseline
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    r_delta = _compute_ratio(h_delta)
    if r_delta > best_ratio:
        best_ratio = r_delta
        best_h = h_delta.copy()

    # Strategy: Gradient Ascent on log(h) with Soft-Max Convolution
    # We maximize J(h) = ||h||_2^2 / S(h) subject to ||h||_1 = 1, h >= 0.
    # Let u = log(h). h = exp(u) / sum(exp(u)).
    
    eta = 0.05  # Learning rate for log-h
    max_iters = 10000
    patience = 100
    stale = 0
    beta = 10.0  # Temperature for soft-max
    
    current_h = h.copy()
    current_ratio = r
    log_h = np.log(current_h + 1e-30)
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Compute autoconvolution
        conv = np.convolve(current_h, current_h, mode='full')
        
        # 2. Compute Soft-Max and its gradient
        max_conv = np.max(conv)
        # Numerical stability
        exp_conv = np.exp(beta * (conv - max_conv))
        sum_exp = np.sum(exp_conv)
        if sum_exp < 1e-15:
            break
        weights = exp_conv / sum_exp
        
        S = (1.0 / beta) * np.log(sum_exp) + max_conv # Actual soft max value
        
        # 3. Compute gradient of S w.r.t h
        # grad_S = A + B
        # A_i = sum_j w_{i+j} h_j = correlate(weights, h, 'full')[i]
        # B_i = sum_j w_{i-j} h_j = convolve(weights, h, 'full')[i]
        
        A = np.correlate(weights, current_h, mode='full')[0:N]
        B = np.convolve(weights, current_h, mode='full')[0:N]
        grad_S = A + B

        # 4. Compute Gradient of Objective J = L2_sq / S
        L2_sq = np.sum(current_h**2)
        # dJ/dh = (2h * S - L2_sq * grad_S) / S^2
        grad_J = (2.0 * current_h * S - L2_sq * grad_S) / (S * S)
        
        # 5. Convert to gradient w.r.t u = log(h)
        # dJ/du = dJ/dh * dh/du = dJ/dh * h
        grad_u = grad_J * current_h
        
        # 6. Update u
        log_h += eta * grad_u
        
        # 7. Renormalize h
        # h = exp(log_h) / sum(exp(log_h))
        # To prevent overflow, shift log_h
        max_log = np.max(log_h)
        h_new = np.exp(log_h - max_log)
        sum_h = np.sum(h_new)
        if sum_h < 1e-15:
            break
        h_new /= sum_h
        
        # 8. Evaluate
        r_new = _compute_ratio(h_new)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h_new.copy()
            stale = 0
            current_h = h_new
        else:
            stale += 1
            if stale > patience:
                # Restart with random sparse or different beta
                if i % 100 == 0:
                    # Random restart
                    mask = np.random.rand(N) > 0.7
                    if not np.any(mask):
                        mask[0] = True
                    current_h = mask.astype(float)
                    current_h /= np.sum(current_h)
                    log_h = np.log(current_h + 1e-30)
                    stale = 0
                    beta *= 0.5
                    if beta < 1.0:
                        beta = 1.0
                else:
                    eta *= 0.9
                    if eta < 1e-4:
                        break
                        
    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
