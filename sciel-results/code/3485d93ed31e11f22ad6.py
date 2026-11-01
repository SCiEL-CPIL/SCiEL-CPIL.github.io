<<<<<<< SEARCH
def _flip_delta(h, i, n):
    """
    Compute the change in the autocorrelation vector for all lags k=1..n-1
    if we flip h[i].
    
    acf[k] = sum_{t=0}^{n-1-k} h[t] * h[t+k]
    
    Flipping h[i] (h[i] -> -h[i]) changes terms where h[i] is involved.
    For lag k:
    - If i is in [0, n-1-k], then term h[i]*h[i+k] is present.
      Change: h[i]*h[i+k] -> -h[i]*h[i+k]. Delta = -2*h[i]*h[i+k].
    - If i-k is in [0, n-1-k], i.e., i is in [k, n-1], then term h[i-k]*h[i] is present.
      Change: h[i-k]*h[i] -> h[i-k]*(-h[i]). Delta = -2*h[i-k]*h[i].
    
    These two conditions are disjoint for k>0?
    Condition 1: 0 <= i <= n-1-k
    Condition 2: k <= i <= n-1
    
    If i satisfies both:
    0 <= i <= n-1-k AND k <= i <= n-1
    => k <= i <= n-1-k
    This requires k <= n-1-k => 2k <= n-1.
    
    So for small k, both terms might contribute.
    For large k, only one might contribute.
    
    Let's compute the delta vector for all k=1..n-1.
    delta[k] = -2 * h[i] * ( (h[i+k] if i+k < n else 0) + (h[i-k] if i-k >= 0 else 0) )
    
    Wait, let's verify.
    Original term at (i, i+k) in sum for lag k: exists if i+k < n. Value h[i]h[i+k].
    New term: -h[i]h[i+k]. Diff: -2 h[i] h[i+k].
    
    Original term at (i-k, i) in sum for lag k: exists if i-k >= 0. Value h[i-k]h[i].
    New term: h[i-k](-h[i]). Diff: -2 h[i-k] h[i].
    
    So delta[k] = -2 * h[i] * ( (h[i+k] if i+k < n else 0) + (h[i-k] if i-k >= 0 else 0) )
    """
    # Vectorized computation for all k
    k = np.arange(1, n)
    
    # h[i+k] part
    idx1 = i + k
    valid1 = idx1 < n
    # We need h at indices idx1 where valid1 is true, 0 otherwise
    # Create a zero array and fill
    h_plus = np.zeros(n)
    if valid1.any():
        h_plus[valid1] = h[idx1[valid1]]
        
    # h[i-k] part
    idx2 = i - k
    valid2 = idx2 >= 0
    h_minus = np.zeros(n)
    if valid2.any():
        h_minus[valid2] = h[idx2[valid2]]
        
    delta = -2.0 * h[i] * (h_plus + h_minus)
    return delta
=======
def _flip_delta(h, i, n):
    """
    Compute the change in the autocorrelation vector for all lags k=1..n-1
    if we flip h[i].
    
    acf[k] = sum_{t=0}^{n-1-k} h[t] * h[t+k]
    
    Flipping h[i] changes the terms involving h[i].
    For a specific lag k:
    - Term h[i]*h[i+k] exists if i+k < n. Change: -2*h[i]*h[i+k].
    - Term h[i-k]*h[i] exists if i-k >= 0. Change: -2*h[i-k]*h[i].
    
    delta[k] = -2 * h[i] * ( (h[i+k] if i+k < n else 0) + (h[i-k] if i-k >= 0 else 0) )
    """
    k = np.arange(1, n)
    
    # Calculate contributions from h[i+k]
    idx_plus = i + k
    # Use np.where to handle boundaries safely and efficiently
    # np.where(condition, x, y) evaluates both x and y, but we can use advanced indexing or padding.
    # Since n is small (400), simple np.where is fine, but we must ensure indices are valid for numpy if we were indexing directly.
    # Actually, np.where doesn't protect against out-of-bounds indexing if we pass h[idx_plus] directly.
    # We need to be careful.
    
    # Safer approach:
    # Create arrays of zeros, then fill valid indices.
    # But the previous error was due to size mismatch.
    # Let's use np.zeros(n-1) this time.
    
    term_plus = np.zeros(n-1)
    valid_plus = idx_plus < n
    if np.any(valid_plus):
        term_plus[valid_plus] = h[idx_plus[valid_plus]]
        
    term_minus = np.zeros(n-1)
    idx_minus = i - k
    valid_minus = idx_minus >= 0
    if np.any(valid_minus):
        term_minus[valid_minus] = h[idx_minus[valid_minus]]
        
    delta = -2.0 * h[i] * (term_plus + term_minus)
    return delta
>>>>>>> REPLACE

This fix ensures that `term_plus` and `term_minus` have size `n-1`, matching the boolean masks `valid_plus` and `valid_minus`.

Let's double check the SA logic again to ensure no other issues.
In `construct()`:
