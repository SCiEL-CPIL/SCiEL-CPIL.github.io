import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0

def project_to_simplex_box(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    # Binary search for the shift tau
    # We want sum(clip(h_i - tau, 0, 1)) = target_sum
    # Let f(tau) = sum(clip(h_i - tau, 0, 1))
    # f is monotonically decreasing in tau.
    # f(-1) = sum(1) = N (if h_i >= 0)
    # f(1) = sum(0) = 0
    # We need f(tau) = target_sum
    
    lo = -1.0
    hi = 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        proj = np.clip(h - mid, 0, 1)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    proj = np.clip(h - (lo + hi) / 2, 0, 1)
    # Fine tune to ensure exact sum if possible (floating point)
    # The binary search should be close enough.
    # To be safe, we can add the residual to one element, but that might break bounds.
    # Given the continuous nature, the projection is unique.
    return proj

def compute_overlaps(h):
    # Circular autocorrelation
    # C_k = sum h_i h_{i+k}
    # Use FFT for speed? N=100 is small, direct is fine.
    # O_k = sum h_i (1 - h_{i+k})
    # We want min O_k? No, min max O_k.
    # Max O_k = S - Min C_k.
    # So we want Min C_k to be as large as possible.
    
    c = np.zeros(N)
    for k in range(N):
        c[k] = np.sum(h * np.roll(h, k))
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initialize
    h = np.full(N, 0.5)
    
    # Check if we need to search.
    # If 0.5 is optimal, we might just return it.
    # But let's run a few iterations to be sure.
    
    # Soft-min parameter
    lam = 10.0
    
    for i in range(500):
        if time.time() > deadline:
            break
            
        c = compute_overlaps(h)
        # We want to maximize min(c)
        # Gradient of min(c) is approx gradient of soft-min
        # soft_min = -1/lam * log(sum(exp(-lam * c)))
        # grad soft_min_j = - (exp(-lam * c_j) / sum(exp(-lam * c))) * grad c_j
        
        # Calculate weights for soft-min
        exp_neg_lam_c = np.exp(-lam * c)
        sum_exp = np.sum(exp_neg_lam_c)
        weights = exp_neg_lam_c / sum_exp
        
        # Gradient of c_k w.r.t h
        # c_k = sum_i h_i h_{i+k}
        # d(c_k)/d(h_j) = h_{j-k} + h_{j+k} (with circular indexing)
        # grad c = sum_k weights_k * grad(c_k)
        
        grad_c = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-6:
                # grad of c_k at j is h_{j-k} + h_{j+k}
                # vectorized: roll(h, k) + roll(h, -k)
                g_k = np.roll(h, k) + np.roll(h, -k)
                grad_c += weights[k] * g_k
        
        # We want to maximize min(c), so we go in direction of grad(min(c))
        # grad(min(c)) approx - weights * grad(c) ?
        # Let F = -1/lam log(sum exp(-lam c))
        # dF/dh = -1/lam * (1/sum) * sum (exp(-lam c_k) * (-lam) * dc_k/dh)
        #        = sum (exp(-lam c_k) / sum) * dc_k/dh
        #        = sum weights_k * dc_k/dh
        # So we want to move in direction of grad_c.
        
        # Learning rate
        lr = 0.01
        
        h_new = h + lr * grad_c
        
        # Project
        h_new = project_to_simplex_box(h_new, TARGET_SUM)
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < 1e-8:
            h = h_new
            break
        h = h_new
        
    # Final projection to ensure exact sum and bounds
    h = project_to_simplex_box(h, TARGET_SUM)
    
    # Convert to list
    return list(h)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Sum: {sum(res)}")
    print(f"Min: {min(res)}, Max: {max(res)}")
