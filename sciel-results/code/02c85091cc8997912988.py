"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Smoothed minimization using log-sum-exp approximation of the max overlap,
optimized via Projected Gradient Descent.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115


def _compute_overlaps(h):
    """Compute C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_overlaps(h)
    return float(np.max(cs))


def _project_to_feasible(y):
    """
    Project vector y onto the set {h | 0 <= h_i <= 1, sum(h) = N/2}.
    Uses bisection on the threshold tau.
    """
    n = len(y)
    target_sum = n / 2.0
    
    # h_i = clip(y_i - tau, 0, 1)
    # We need sum(h_i) = target_sum.
    # sum(clip(y_i - tau, 0, 1)) is a non-increasing function of tau.
    # If tau is very small, h_i -> 1, sum -> n.
    # If tau is very large, h_i -> 0, sum -> 0.
    
    lo = np.min(y) - 2.0
    hi = np.max(y) + 2.0
    
    for _ in range(50): # 50 bisection steps for 1e-15 precision
        mid = (lo + hi) / 2.0
        h = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h = np.clip(y - lo, 0.0, 1.0)
    # Correct any numerical drift in the sum
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Adjust elements strictly inside (0,1)
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def _soft_max_obj(h, alpha):
    """
    Compute log-sum-exp of overlaps and its gradient.
    Objective: (1/alpha) * log(sum(exp(alpha * C_k)))
    Gradient: (1/sum(exp(alpha * C_k))) * sum(alpha * exp(alpha * C_k) * grad_C_k)
    """
    cs = _compute_overlaps(h)
    
    # Shift for numerical stability
    c_max = np.max(cs)
    exp_cs = np.exp(alpha * (cs - c_max))
    sum_exp = np.sum(exp_cs)
    
    obj = (1.0 / alpha) * (np.log(sum_exp) + alpha * c_max)
    
    # Gradient of objective
    # d/dh [ (1/alpha) log(sum_exp) ] = (1/alpha) * (1/sum_exp) * sum( alpha * exp_cs_k * d/dh C_k )
    # = (1/sum_exp) * sum( exp_cs_k * d/dh C_k )
    
    grad = np.zeros(N)
    
    # We need sum_k exp_cs_k * grad_C_k
    # grad_C_k[j] = -(2/N) * (h[j-k] if j>=k else 0 + h[j+k] if j+k<N else 0)
    # Note: C_k = (2/N)(N/2 - r_k). So grad C_k = -(2/N) grad r_k.
    # r_k = sum_{i=k}^{N-1} h_i h_{i-k}.
    # grad r_k / dh_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k < N).
    
    # Let's compute the weighted sum efficiently.
    # grad_obj[j] = (1/sum_exp) * sum_k exp_cs_k * [ -(2/N) * (h[j-k] (j>=k) + h[j+k] (j+k<N)) ]
    
    # This can be computed as:
    # Term 1: sum_{k=0}^{j} exp_cs_k * h[j-k]  (for the h[j-k] part, k <= j)
    # Term 2: sum_{k=0}^{N-1-j} exp_cs_k * h[j+k] (for the h[j+k] part, k <= N-1-j)
    
    # Let w_k = exp_cs_k / sum_exp.
    w = exp_cs / sum_exp
    
    # Term 1 for each j: sum_{k=0}^j w_k h[j-k]
    # This is a convolution of w and h, but truncated.
    # Since N=100, we can just loop or use fft if needed, but N is small.
    # Actually, we can compute this for all j at once using matrix multiplication or loops.
    
    # Let's use a loop for clarity and correctness, N=100 is small.
    grad_raw = np.zeros(N)
    for j in range(N):
        # Part 1: k from 0 to j
        if j >= 0:
            # w[0..j] dot h[j..0] -> sum_{k=0}^j w[k] * h[j-k]
            part1 = np.dot(w[:j+1], h[j::-1])
        else:
            part1 = 0.0
            
        # Part 2: k from 0 to N-1-j
        if j < N:
            # w[0..N-1-j] dot h[j..N-1] -> sum_{k=0}^{N-1-j} w[k] * h[j+k]
            part2 = np.dot(w[:N-j], h[j:])
        else:
            part2 = 0.0
            
        grad_raw[j] = -(2.0 / N) * (part1 + part2)
        
    grad = grad_raw
    return obj, grad


def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses Projected Gradient Descent on a log-sum-exp smoothed max overlap.
    """
    t0 = time.time()
    np.random.seed(42)
    
    # Initialize with a structured guess
    i = np.arange(N)
    h = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N)
    h = np.clip(h, 0.0, 1.0)
    h = _project_to_feasible(h)
    
    best_h = h.copy()
    best_score = _score(best_h)
    
    # Annealing schedule for alpha
    # Start with small alpha (smooth), increase to large alpha (sharp max)
    alphas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for alpha in alphas:
        if time.time() > _DEADLINE - 5:
            break
            
        # Gradient Descent steps
        # Adaptive step size
        eta = 0.01
        for step in range(200):
            if time.time() > _DEADLINE - 5:
                break
                
            obj, grad = _soft_max_obj(h, alpha)
            
            # Gradient step
            h_new = h - eta * grad
            
            # Project back to feasible set
            h_proj = _project_to_feasible(h_new)
            
            # Check if we improved (using the actual max score, not the smoothed one)
            # Or just check the smoothed objective?
            # Using actual score is safer for final result, but might oscillate.
            # Let's use the smoothed objective for the step decision, 
            # but update best_h based on actual score.
            
            # Simple backtracking if needed? 
            # For now, fixed step with projection is often robust enough for this.
            h = h_proj
            
            # Update best
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                
            # Adaptive step size: if no improvement for a while, reduce eta
            # (Simplified: just keep eta constant or decay it)
            eta *= 0.995 # Gentle decay
    
    # Final polish: A few steps of SLSQP or similar if time permits?
    # Or just return best_h.
    # Let's do a few steps of direct gradient descent on the max overlap (using subgradient)
    # Or just return the result from the annealing.
    
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-10) & (best_h < 1 - 1e-10)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
            
    return best_h
