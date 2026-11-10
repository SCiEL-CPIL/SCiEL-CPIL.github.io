# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_entropies_and_grads(P_flat):
    """
    Computes D, H0, H1, H2 and their gradients with respect to P_flat.
    P_flat is a 1D array of length 100, normalized to sum 1.
    Returns:
        D, H0, H1, H2: floats
        grad_D, grad_H0, grad_H1, grad_H2: 1D arrays of length 100
    """
    # Precomputed value indices
    # d_vals: x - y (offset 9)
    # h0_vals: x
    # h1_vals: x + y
    # h2_vals: x + 2y
    
    # We need to map each cell index i to its value index in each projection.
    # Let's precompute these mappings globally or pass them.
    # For efficiency, I'll define them in the outer scope or as global constants.
    
    # Using global precomputed arrays for speed
    global _d_idx, _h0_idx, _h1_idx, _h2_idx, _d_count, _h0_count, _h1_count, _h2_count
    
    # Pool weights
    d_pooled = np.bincount(_d_idx, weights=P_flat, minlength=_d_count)
    h0_pooled = np.bincount(_h0_idx, weights=P_flat, minlength=_h0_count)
    h1_pooled = np.bincount(_h1_idx, weights=P_flat, minlength=_h1_count)
    h2_pooled = np.bincount(_h2_idx, weights=P_flat, minlength=_h2_count)
    
    # Calculate Entropies
    def _entropy_and_grad(pooled, count):
        total = pooled.sum()
        if total <= 1e-15:
            return 0.0, np.zeros(count)
        probs = pooled / total
        # H = - sum p log2 p
        # dH/dp_i = -log2(p_i) - 1/ln(2)
        # Note: p_i = pooled_i / total.
        # dH/dpooled_i = -log2(p_i) - 1/ln(2)
        # But we want gradient w.r.t P_flat.
        # dH/dP_flat[j] = sum_{k: val(k)=j} dH/dpooled_k * (1/total) ? No.
        # H is a function of pooled. pooled_k = sum_{j: val(j)=k} P_flat[j].
        # dH/dP_flat[j] = dH/dpooled_{val(j)} * 1.
        # So we just need dH/dpooled, then scatter it back.
        
        mask = probs > 1e-15
        if not np.any(mask):
            return 0.0, np.zeros(count)
        
        H = -np.sum(probs[mask] * np.log2(probs[mask]))
        
        # Gradient w.r.t pooled
        # dH/dpooled_k = -log2(probs_k) - 1/ln(2) for probs_k > 0, else 0
        grad_pooled = np.zeros(count)
        grad_pooled[mask] = -np.log2(probs[mask]) - 1.0 / np.log(2)
        
        return H, grad_pooled

    D, grad_D_pooled = _entropy_and_grad(d_pooled, _d_count)
    H0, grad_H0_pooled = _entropy_and_grad(h0_pooled, _h0_count)
    H1, grad_H1_pooled = _entropy_and_grad(h1_pooled, _h1_count)
    H2, grad_H2_pooled = _entropy_and_grad(h2_pooled, _h2_count)
    
    # Scatter gradients back to P_flat size
    grad_D = np.zeros(100)
    np.add.at(grad_D, _d_idx, grad_D_pooled)
    
    grad_H0 = np.zeros(100)
    np.add.at(grad_H0, _h0_idx, grad_H0_pooled)
    
    grad_H1 = np.zeros(100)
    np.add.at(grad_H1, _h1_idx, grad_H1_pooled)
    
    grad_H2 = np.zeros(100)
    np.add.at(grad_H2, _h2_idx, grad_H2_pooled)
    
    return D, H0, H1, H2, grad_D, grad_H0, grad_H1, grad_H2

def _ratio(P_flat):
    P_flat = np.clip(P_flat, 0, None)
    s = P_flat.sum()
    if s <= 1e-15:
        return 0.0
    P_flat = P_flat / s
    D, H0, H1, H2, _, _, _, _ = _compute_entropies_and_grads(P_flat)
    M = max(H0, H1, H2)
    if M <= 1e-15:
        return 0.0
    return D / M

def _optimize_P():
    deadline = time.time() + 580
    
    # Precompute indices
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    
    _d_vals = x_flat - y_flat
    _h0_vals = x_flat
    _h1_vals = x_flat + y_flat
    _h2_vals = x_flat + 2 * y_flat
    
    global _d_idx, _h0_idx, _h1_idx, _h2_idx, _d_count, _h0_count, _h1_count, _h2_count
    _d_idx = _d_vals + 9
    _h0_idx = _h0_vals
    _h1_idx = _h1_vals
    _h2_idx = _h2_vals
    _d_count = 19
    _h0_count = 10
    _h1_count = 19
    _h2_count = 28
    
    rng = np.random.RandomState(42)
    
    best_P = np.ones(100) / 100.0
    best_score = _ratio(best_P)
    
    # Gradient Ascent with Simulated Annealing fallback
    # We will run multiple restarts
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize P
        if restart == 0:
            P = np.ones(100) / 100.0
        else:
            # Random initialization with some structure
            P = rng.dirichlet(np.ones(100) * 0.1)
            # Or start from best_P with noise
            if rng.random() < 0.5:
                P = best_P + rng.normal(0, 0.05, 100)
                P = np.clip(P, 0, None)
                P /= P.sum()
        
        score = _ratio(P)
        if score > best_score:
            best_score = score
            best_P = P.copy()
        
        alpha = 0.01
        T = 0.1
        
        for step in range(5000):
            if time.time() > deadline:
                break
                
            D, H0, H1, H2, grad_D, grad_H0, grad_H1, grad_H2 = _compute_entropies_and_grads(P)
            M = max(H0, H1, H2)
            if M <= 1e-15:
                break
            
            # Identify dominant M
            if H0 == M:
                grad_M = grad_H0
            elif H1 == M:
                grad_M = grad_H1
            else:
                grad_M = grad_H2
            
            # Gradient of D/M
            # d(D/M)/dP = (M * grad_D - D * grad_M) / M^2
            grad_ratio = (M * grad_D - D * grad_M) / (M * M)
            
            # Step
            P_new = P + alpha * grad_ratio
            
            # Project to simplex
            P_new = np.clip(P_new, 0, None)
            s = P_new.sum()
            if s <= 1e-15:
                continue
            P_new = P_new / s
            
            score_new = _ratio(P_new)
            
            if score_new > score:
                P = P_new
                score = score_new
                if score > best_score:
                    best_score = score
                    best_P = P.copy()
                # Increase alpha slightly if improving
                alpha = min(alpha * 1.01, 0.1)
            else:
                # Simulated Annealing accept
                if T > 0.001:
                    prob = np.exp((score_new - score) / T)
                    if rng.random() < prob:
                        P = P_new
                        score = score_new
                # Decrease alpha
                alpha *= 0.95
                T *= 0.99
            
    return best_P.reshape(10, 10)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    
    P = _optimize_P()
    
    # Final safety checks
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
    
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    return P

# EVOLVE-BLOCK-END
