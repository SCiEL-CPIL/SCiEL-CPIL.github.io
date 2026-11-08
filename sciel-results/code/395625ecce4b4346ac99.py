import numpy as np
import time

N = 400
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    n = len(h)
    # Full linear autoconvolution
    c = np.convolve(h, h, mode='full')
    return np.max(np.abs(c))

def _score(h):
    """Compute the metric: max|autoconv| / sum(h^2)."""
    max_abs = _autoconv_max_abs(h)
    integral_sq = np.sum(h**2)
    return max_abs / integral_sq

def _rudin_shapiro(n):
    """Generate Rudin-Shapiro sequence of length n (padded to power of 2)."""
    # Build up to next power of 2
    m = 1
    while m < n:
        m *= 2
    # Rudin-Shapiro construction
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < m:
        a = np.concatenate([a, b])
        b = np.concatenate([a[:-len(b)], -b])
    return a[:n]

def _hill_climb(h, max_iters=20000, deadline=DEADLINE):
    """Local search: flip individual elements to reduce max|autoconv|."""
    h = h.copy()
    n = len(h)
    current_score = _score(h)
    
    # Precompute autoconvolution for fast updates
    # C(k) = sum_i h[i]*h[i+k] for k >= 0, and by symmetry C(-k) = C(k)
    # Flipping h[j] changes C(k) by: for each i where i+k=j or i=j+k
    # C(k) -= 2*h[j]*h[j-k] for k where j-k in range (k>0)
    # C(0) -= 2*h[j]^2 = -2 (since h[j]^2=1)
    # C(k) += 2*h[j]*h[j+k] for k where j+k in range
    
    # Use FFT-based approach for speed
    def _update_autoconv(h, j):
        """Return change in autoconvolution when flipping h[j]."""
        n = len(h)
        # Autoconv at lag k: sum_{i} h[i]*h[i+k]
        # Flipping h[j]: h[j] -> -h[j]
        # Delta C(k) = -2*h[j]*h[j-k] - 2*h[j]*h[j+k] + 4*h[j]^2*delta_{k=0}
        # More carefully: C_new(k) - C_old(k) = -2*h[j]*(h[j-k] + h[j+k]) + 4*h[j]^2 if k=0
        # Actually: C(k) = sum_i h[i]*h[i+k]
        # After flip at j: C'(k) = C(k) - h[j]*h[j+k] - h[j-k]*h[j] + (-h[j])*(-h[j+k]) + ... 
        # Let me just recompute for the affected lags
        
        # For lag k (k>=0): C(k) = sum_{i=0}^{n-1-k} h[i]*h[i+k]
        # If j is in [0, n-1-k], term h[j]*h[j+k] is present
        # If j-k is in [0, n-1-k], i.e., j in [k, n-1], term h[j-k]*h[j] is present
        # After flip: both terms change sign
        # Delta C(k) = -2*h[j]*h[j+k]*I(j in [0,n-1-k]) - 2*h[j-k]*h[j]*I(j in [k,n-1])
        # For k=0: Delta C(0) = -2*h[j]^2 = -2
        
        deltas = np.zeros(2*n-1)
        old_hj = h[j]
        # For each lag k from -(n-1) to (n-1):
        # C(k) = sum_i h[i]*h[i+k] where indices valid
        # Term involving h[j]: when i=j (h[j]*h[j+k]) if j+k valid, or i=j-k (h[j-k]*h[j]) if j-k valid
        # After flip, these terms get -h[j] instead of h[j]
        # Delta = -2*h[j]*h[j+k] (if j+k in [0,n-1]) - 2*h[j-k]*h[j] (if j-k in [0,n-1])
        
        for k in range(-(n-1), n):
            delta = 0.0
            # i = j, so i+k = j+k
            if 0 <= j+k < n:
                delta -= 2.0 * old_hj * h[j+k]
            # i = j-k, so i+k = j
            if 0 <= j-k < n and j-k != j:  # avoid double counting k=0
                delta -= 2.0 * h[j-k] * old_hj
            deltas[k+n-1] = delta
        return deltas
    
    best_h = h.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if it % 500 == 0:
            # Recompute to avoid drift
            pass
        
        best_gain = 0
        best_j = -1
        # Sample a subset of positions for speed
        if n > 100:
            candidates = np.random.choice(n, size=min(50, n), replace=False)
        else:
            candidates = np.arange(n)
        
        for j in candidates:
            # Estimate score change
            # We need max|C_new| where C_new = C_old + delta
            # This is expensive to compute exactly. Use a heuristic:
            # Check if flipping j reduces the current max
            old_max = np.max(np.abs(np.convolve(h, h, 'full')))
            h[j] = -h[j]
            new_max = np.max(np.abs(np.convolve(h, h, 'full')))
            h[j] = old_hj = h[j]  # restore
            h[j] = -old_hj  # flip back
            
            gain = old_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_j = j
        
        if best_j >= 0:
            h[best_j] = -h[best_j]
            current_score = _score(h)
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
    
    return best_h

def construct():
    """
    Returns a signed step sequence minimizing max|autoconvolution| / sum(h^2).
    Uses Rudin-Shapiro construction with local search refinement.
    """
    start_time = time.time()
    
    # Start with Rudin-Shapiro sequence
    h = _rudin_shapiro(N)
    
    # Run hill climbing
    h = _hill_climb(h, max_iters=5000, deadline=DEADLINE)
    
    # Ensure we return within deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Return what we have
        pass
    
    return h
