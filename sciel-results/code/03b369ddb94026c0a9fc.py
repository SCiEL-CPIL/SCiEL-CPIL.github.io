import numpy as np
import time

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0

    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)

    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum

    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]

    ss_flat = ss.flatten()
    pp_flat = pp.flatten()

    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]

    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)

    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            P += p[k] * p[l] * prob_inner

    return P

def compute_gradient(s, p):
    """
    Compute the gradient of P with respect to p (unnormalized weights).
    P = sum_{k,l} p_k p_l * F(k,l) where F(k,l) = sum_{i,j: s_i+s_j < 2s_l-s_k} p_i p_j
    
    Actually, let's derive the gradient properly.
    P = sum_{k,l,i,j} p_k p_l p_i p_j * I(s_i + s_j < 2s_l - s_k)
    
    dP/dp_m = sum_{k,l,i,j} I(s_i + s_j < 2s_l - s_k) * (delta_{m,k} p_l p_i p_j + 
                                                          p_k delta_{m,l} p_i p_j + 
                                                          p_k p_l delta_{m,i} p_j + 
                                                          p_k p_l p_i delta_{m,j})
    
    This can be computed efficiently by precomputing terms.
    Let A_{k,l} = I(s_i + s_j < 2s_l - s_k) summed over i,j with weights? No, the indicator is inside.
    
    Let's use the sorted pair sums approach.
    For a fixed m, dP/dp_m involves:
    1. Terms where m=k: sum_l p_l * sum_{i,j: s_i+s_j < 2s_l-s_m} p_i p_j
    2. Terms where m=l: sum_k p_k * sum_{i,j: s_i+s_j < 2s_m-s_k} p_i p_j
    3. Terms where m=i: sum_{k,l} p_k p_l * sum_j I(s_m+s_j < 2s_l-s_k) p_j
    4. Terms where m=j: sum_{k,l} p_k p_l * sum_i I(s_i+s_m < 2s_l-s_k) p_i
    
    Note that term 3 and 4 are identical by symmetry of i,j.
    So dP/dp_m = 2 * sum_l p_l * F(l, m) + 2 * sum_k p_k * F(m, k) + 2 * sum_{k,l} p_k p_l * G(m, k, l)
    
    Where F(a, b) = sum_{i,j: s_i+s_j < 2s_a-s_b} p_i p_j
    And G(m, k, l) = sum_j I(s_m+s_j < 2s_l-s_k) p_j
    
    Let's precompute:
    ss_sorted, pp_sorted, C (cumsum of pp)
    
    For F(a, b):
    threshold = 2s_a - s_b
    idx = searchsorted(ss_sorted, threshold, 'left')
    F(a, b) = C[idx-1] if idx>0 else 0
    
    For G(m, k, l):
    threshold = 2s_l - s_k - s_m
    We need sum of p_j where s_m + s_j < 2s_l - s_k
    i.e., s_j < 2s_l - s_k - s_m
    Since s is sorted, we can find the count of j such that s_j < threshold.
    Let's precompute a prefix sum of p: P_sum_prefix[i] = sum_{j=0}^{i-1} p_j
    
    Then G(m, k, l) = P_sum_prefix[count_j] where count_j is number of j with s_j < threshold.
    """
    n = len(s)
    grad = np.zeros(n)
    
    # Precompute pair sums and sorted indices
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    ss_flat = ss.flatten()
    pp_flat = pp.flatten()
    
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]
    C = np.cumsum(pp_sorted)
    
    # Precompute prefix sums of p for G
    p_prefix = np.zeros(n + 1)
    p_prefix[1:] = np.cumsum(p)
    
    # Precompute F(a, b) matrix
    # F[a, b] = prob(s_i + s_j < 2s_a - s_b)
    F = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            threshold = 2.0 * s[a] - s[b]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                F[a, b] = C[idx - 1]
    
    # Compute gradient for each m
    for m in range(n):
        # Term 1: 2 * sum_l p_l * F(l, m)  (m=k)
        term1 = 2.0 * np.dot(p, F[:, m])
        
        # Term 2: 2 * sum_k p_k * F(m, k)  (m=l)
        term2 = 2.0 * np.dot(p, F[m, :])
        
        # Term 3 & 4: 2 * sum_{k,l} p_k p_l * G(m, k, l)
        # G(m, k, l) = sum_j p_j * I(s_j < 2s_l - s_k - s_m)
        term34 = 0.0
        for k in range(n):
            for l in range(n):
                threshold_j = 2.0 * s[l] - s[k] - s[m]
                # Count how many s_j < threshold_j
                # Since s is sorted, use searchsorted
                count_j = np.searchsorted(s, threshold_j, side='left')
                if count_j > 0:
                    g_val = p_prefix[count_j]
                else:
                    g_val = 0.0
                term34 += 2.0 * p[k] * p[l] * g_val
        
        grad[m] = term1 + term2 + term34
        
    return grad

def optimize_weights_gradient(s, time_limit):
    """
    Optimize weights using gradient ascent on the simplex.
    """
    n = len(s)
    # Initialize with geometric decay
    w = np.exp(-np.arange(n) * 0.5)
    w = w / np.sum(w)
    
    best_w = w.copy()
    best_score = evaluate_measure(s, w)
    
    start_time = time.time()
    step_size = 0.1
    
    # Gradient ascent with projection onto simplex
    # Simple projection: clip negatives and renormalize (approximate)
    # Better: use a proper projection or just careful updates
    
    for iter in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        grad = compute_gradient(s, w)
        
        # Update
        w_new = w + step_size * grad
        
        # Project onto simplex (non-negative, sum to 1)
        # Simple method: clip and renormalize
        w_new = np.maximum(w_new, 0.0)
        w_sum = np.sum(w_new)
        if w_sum > 1e-15:
            w_new = w_new / w_sum
        else:
            w_new = np.ones(n) / n
        
        score_new = evaluate_measure(s, w_new)
        
        if score_new > best_score:
            best_score = score_new
            best_w = w_new.copy()
            w = w_new
            # Decay step size if we made a big jump? Or keep it.
            # Let's adapt step size slightly
        else:
            # If no improvement, reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
            # Try a smaller step
            w_try = w + step_size * grad
            w_try = np.maximum(w_try, 0.0)
            w_try_sum = np.sum(w_try)
            if w_try_sum > 1e-15:
                w_try = w_try / w_try_sum
            else:
                continue
                
            score_try = evaluate_measure(s, w_try)
            if score_try > evaluate_measure(s, w):
                w = w_try
            else:
                # Stagnant, break or restart?
                if step_size < 1e-5:
                    break

    return best_w, best_score

def beat_the_average_measure() -> tuple:
    deadline = time.time() + 600.0
    safety_margin = 10.0
    
    # Try a few different geometric supports
    # The key insight is that geometric decay in weights works well.
    # Let's try different bases for the support itself?
    # Or just fix a good support and optimize weights.
    
    # Based on "Geometric decay concentrates mass at small values..."
    # Let's use a support that is linear or geometric.
    # The current program uses various supports. Let's pick a good one.
    # A linear support [0, 1, ..., n-1] or normalized [0..1] is standard.
    # Let's try n=20, linear support.
    
    n = 20
    s = np.linspace(0, 1, n)
    
    # Optimize weights
    w, score = optimize_weights_gradient(s, 300.0)
    
    # Try a few other supports if time permits
    # Geometric support
    s2 = np.geomspace(1e-3, 1.0, n)
    w2, score2 = optimize_weights_gradient(s2, 100.0)
    
    if score2 > score:
        s, w, score = s2, w2, score2
    
    # Another geometric
    s3 = np.geomspace(0.01, 1.0, n)
    w3, score3 = optimize_weights_gradient(s3, 100.0)
    
    if score3 > score:
        s, w, score = s3, w3, score3
        
    # Final validation
    s = np.asarray(s, dtype=np.float64).flatten()
    w = np.asarray(w, dtype=np.float64).flatten()
    
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    
    w_sum = np.sum(w)
    if w_sum > 1e-9:
        w = w / w_sum
    else:
        w = np.ones(n) / n
        
    return s, w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
