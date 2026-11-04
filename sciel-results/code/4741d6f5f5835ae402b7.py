import numpy as np
import time

N = 100

def _max_overlap(h):
    """
    Compute the maximum overlap C_k for linear shifts k=1..N-1.
    C_k = sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
    """
    max_c = 0.0
    for k in range(1, N):
        c_k = np.dot(h[:N-k], 1 - h[k:])
        if c_k > max_c:
            max_c = c_k
    return max_c

def _compute_all_overlaps(h):
    """
    Compute all C_k values.
    """
    C = np.zeros(N-1)
    for k in range(1, N):
        C[k-1] = np.dot(h[:N-k], 1 - h[k:])
    return C

def _project_to_feasible(h):
    """
    Project h to satisfy sum(h) = N/2 and 0 <= h_i <= 1.
    Uses binary search for the shift lambda.
    """
    target_sum = N / 2.0
    
    def f(lam):
        return np.sum(np.clip(h - lam, 0, 1))
    
    # f is monotonic decreasing in lam
    # We want f(lam) = target_sum
    # If sum(h) < target, we need to increase h, so lam should be negative (or small)
    # If sum(h) > target, we need to decrease h, so lam should be positive
    
    # Range for lam:
    # min h is 0, max h is 1.
    # If all h are 0, sum=0. If all h are 1, sum=N.
    # So lam range [min(h)-1, max(h)+1] is safe.
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    # Check bounds
    if f(lo) < target_sum:
        # Even with max shift, sum is too small?
        # f(lo) = sum(clip(h - (min(h)-1), 0, 1))
        # h - min(h) + 1 >= 1. So clip is 1. Sum is N.
        # So f(lo) should be N.
        pass
    if f(hi) > target_sum:
        # f(hi) = sum(clip(h - (max(h)+1), 0, 1))
        # h - max(h) - 1 <= -1. Clip is 0. Sum is 0.
        pass

    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = f(mid)
        if val > target_sum:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
            
    lam = (lo + hi) / 2.0
    h_proj = np.clip(h - lam, 0, 1)
    
    # Due to numerical errors, sum might not be exactly target_sum
    # Adjust one element slightly if necessary?
    # The binary search should be precise enough.
    # But to be safe, we can do a final small adjustment on an element not at bound.
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-9:
        mask = (h_proj > 1e-9) & (h_proj < 1 - 1e-9)
        if np.any(mask):
            idx = np.where(mask)[0][0]
            h_proj[idx] += diff
            # Clip again? If diff is small, it should stay in bounds.
            h_proj[idx] = np.clip(h_proj[idx], 0, 1)
            
    return h_proj

def construct():
    start_time = time.time()
    deadline = 110.0  # seconds
    
    # Initialize with uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    
    # Parameters for LSE optimization
    beta = 10.0
    alpha = 0.01
    n_iter = 0
    
    while time.time() - start_time < deadline:
        # Compute overlaps
        C = _compute_all_overlaps(h)
        
        # Compute softmax weights
        # Avoid overflow
        C_max = np.max(C)
        exp_C = np.exp(beta * (C - C_max))
        sum_exp = np.sum(exp_C)
        p = exp_C / sum_exp
        
        # Compute gradient of LSE
        # grad_LSE = sum_k p_k * grad_C_k
        grad = np.zeros(N)
        
        # Efficiently compute gradient
        # grad_C_k[j] = 1(j < N-k) - 1(j < N-k)h[j+k] - 1(j >= k)h[j-k]
        # We can accumulate this for each k
        
        for k_idx in range(1, N):
            p_k = p[k_idx - 1]
            if p_k < 1e-6:
                continue
                
            # Indices where j < N-k
            # j in [0, N-k-1]
            # Term 1: 1
            # Term 2: -h[j+k]
            
            # Term 3: -h[j-k] for j in [k, N-1]
            
            # Vectorized update
            # grad[0:N-k] += p_k * (1 - h[k:N])
            grad[:N-k] += p_k * (1 - h[k:])
            
            # grad[k:N] += p_k * (-h[0:N-k])
            grad[k:] += p_k * (-h[:N-k])
            
        # Update h
        h_new = h - alpha * grad
        
        # Project to feasible set
        h_new = _project_to_feasible(h_new)
        
        # Check score
        score_new = _max_overlap(h_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            # Increase beta slightly to sharpen
            beta *= 1.01
            # Maybe reduce alpha
            alpha *= 0.99
        else:
            # No improvement, reduce alpha
            alpha *= 0.95
            
        h = h_new
        n_iter += 1
        
        if time.time() - start_time > deadline - 5:
            break
            
    return best_h
