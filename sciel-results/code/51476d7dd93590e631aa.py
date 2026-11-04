import numpy as np
import time

N = 400

def _compute_acf(h):
    """Compute the autocorrelation (autoconvolution) of h."""
    # Use FFT for initial computation
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is the DC component (sum of squares = N)
    # We are interested in acf[1:N]
    return acf[:N]

def _update_acf(acf, h, i):
    """
    Update the autocorrelation array after flipping h[i].
    h[i] changes sign.
    Returns the new acf array (modified in place or new array).
    """
    # We need to update acf[k] for all k
    # acf[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
    # Let old_h_i = h[i]. new_h_i = -h[i].
    # The terms involving h[i] are:
    # 1. j = i: h[i] h[i+k] (valid if 0 <= i <= N-1-k => k <= N-1-i)
    # 2. j+k = i => j = i-k: h[i-k] h[i] (valid if 0 <= i-k <= N-1-k => k <= i <= N-1)
    
    # We will create a copy to avoid side effects during evaluation
    new_acf = acf.copy()
    old_val = h[i]
    
    for k in range(1, N):
        delta = 0.0
        # Contribution from j=i
        if i <= N - 1 - k:
            delta += old_val * h[i + k]
        # Contribution from j=i-k
        if i >= k:
            delta += h[i - k] * old_val
        
        # The new term will be -delta (because h[i] flips sign)
        # So the change is -2 * delta
        new_acf[k] -= 2.0 * delta
        
    return new_acf

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Run multiple restarts if time permits
    while time.time() < deadline:
        # Initialize with random signs
        h = np.random.choice([-1, 1], size=N)
        
        # Compute initial ACF
        acf = _compute_acf(h)
        current_score = np.max(np.abs(acf[1:N]))
        
        # Local search: Best Improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 0.0
            best_idx = -1
            
            # Evaluate all possible flips
            # To speed this up, we can vectorize the gain calculation?
            # The gain depends on the new max ACF.
            # Calculating the new ACF for each flip is O(N) per flip, so O(N^2) per pass.
            # N=400, N^2=160,000. This is fast.
            
            for i in range(N):
                new_acf = _update_acf(acf, h, i)
                new_score = np.max(np.abs(new_acf[1:N]))
                gain = current_score - new_score
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx != -1 and best_gain > 1e-9:
                # Apply flip
                h[best_idx] *= -1
                acf = _update_acf(acf, h, best_idx)
                current_score -= best_gain
                improved = True
        
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # If we found a very good score, we might stop early?
        # The theoretical minimum for binary sequences is related to sqrt(N).
        # For N=400, sqrt(400)=20. 
        # If best_score < 25, it's quite good.
        if best_score < 20:
            break

    if best_h is None:
        # Fallback to ones if something went wrong
        return np.ones(N)
        
    return best_h
