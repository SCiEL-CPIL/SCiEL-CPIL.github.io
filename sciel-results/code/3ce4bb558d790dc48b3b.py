import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
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

def _gradient_log_ratio_normalized(h):
    """
    Gradient of log(||h||_2^2) - log(sup|h*h|)
    assuming sum(h) is fixed to 1 (or we just use the unconstrained gradient 
    and project, but fixing L1=1 makes the landscape better conditioned for 
    the specific 'shape' search).
    Actually, let's just use the gradient of the ratio R(h) directly.
    R(h) = L2 / (L1 * M)
    dR = (dL2 * L1 * M - L2 * (dL1 * M + L1 * dM)) / (L1 * M)^2
    
    This is messy. Let's stick to maximizing log(R) = log(L2) - log(L1) - log(M).
    grad = 2h/L2 - 1/L1 (for h_i>0) - grad_M / M
    
    If we fix L1=1, the -log(L1) term is constant (gradient 0).
    So we maximize log(L2) - log(M).
    """
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return np.zeros_like(h)
        
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return np.zeros_like(h)
        
    # Gradient of log(L2) = 2h / L2
    grad_l2 = 2.0 * h / l2_sq
    
    # Gradient of -log(M)
    # M = max(conv). Let k* be argmax.
    # dM/dh_i = h[k*-i] if 0 <= k*-i < N else 0
    k_star = np.argmax(conv)
    
    grad_M = np.zeros_like(h)
    for i in range(N):
        j = k_star - i
        if 0 <= j < N:
            grad_M[i] = h[j]
            
    grad_sup = -grad_M / sup_conv
    
    # If we were not fixing L1, we would subtract 1/L1.
    # But we will normalize h after each step, so the L1 constraint is handled by projection/normalization.
    # The gradient of the objective function under the constraint sum(h)=1 is the projection of 
    # the unconstrained gradient onto the tangent space of the simplex.
    # Tangent space: sum(v_i) = 0.
    # v_proj = grad - mean(grad)
    
    grad = grad_l2 + grad_sup
    # Project to sum 0
    grad = grad - np.mean(grad)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Known good Sidon set
    sidon_S = [0, 1, 4, 10, 18, 23, 30]
    
    candidates = []
    # Add the known Sidon set
    candidates.append(sidon_S)
    
    # Generate some random sparse sets
    # We want sets with low collision count in sums
    for _ in range(50):
        if time.time() > deadline - 20.0:
            break
        k = np.random.randint(5, 9) # 5 to 8 elements
        idxs = np.random.choice(N, k, replace=False)
        # Sort for consistency
        idxs = np.sort(idxs)
        
        # Quick check: count number of duplicate sums
        sums = []
        for i in range(len(idxs)):
            for j in range(i, len(idxs)):
                sums.append(idxs[i] + idxs[j])
        num_unique = len(set(sums))
        # Max possible unique sums is k*(k+1)/2
        # If num_unique is close to max, it's a good Sidon candidate
        if num_unique >= 0.8 * (k*(k+1)/2):
            candidates.append(list(idxs))
            
    # Optimize weights for each candidate
    for support in candidates:
        if time.time() > deadline - 10.0:
            break
            
        h = np.zeros(N)
        for idx in support:
            h[idx] = 1.0
        h = h / np.sum(h)
        
        # Gradient ascent
        lr = 0.1
        steps = 2000
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            grad = _gradient_log_ratio_normalized(h)
            h_new = h + lr * grad
            h_new[h_new < 0] = 0
            
            # Normalize to sum 1
            s = np.sum(h_new)
            if s < 1e-12:
                break
            h_new = h_new / s
            
            # Check for improvement
            ratio_new = _compute_ratio(h_new)
            
            if ratio_new > best_ratio:
                best_ratio = ratio_new
                best_h = h_new.copy()
                
            h = h_new
            
            # Decay learning rate
            lr *= 0.9995
            # If lr is too small, break early
            if lr < 1e-4:
                break

    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
