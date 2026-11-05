"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    # Iterate over step sizes d.
    # For a given d, progressions are determined by start index i (0 <= i < d)
    # and length L.
    # An AP starting at i with step d is s[i], s[i+d], s[i+2d], ...
    # We want max |sum_{k=0}^{L-1} s[i + k*d]| for all valid i, L.
    
    for d in range(1, n + 1):
        # Number of complete chains: indices 0..d-1 are starts
        # Chain i has elements at positions i, i+d, i+2d, ...
        # Length of chain i: floor((n-1-i)/d) + 1
        # We can vectorize: for each start i in 0..d-1, get the chain
        # and compute prefix sums, then take max abs.
        
        # For efficiency, only process d where chains have length >= 1
        # (always true) but skip d > n//2 where all chains have length 1
        # (those just contribute |s[i]| = 1)
        if d > n // 2:
            # All chains have length 1, max abs is 1
            if max_abs < 1:
                max_abs = 1
            continue
        
        # Number of starts
        num_starts = d
        # For each start i, chain length L_i = floor((n-1-i)/d) + 1
        # Max chain length for this d: floor((n-1)/d) + 1
        max_len = (n - 1) // d + 1
        
        # Build a 2D array: rows = starts, cols = chain positions
        # arr[i, j] = s[i + j*d] if i + j*d < n else 0
        # Shape: (d, max_len)
        # Indices: i + j*d for i in 0..d-1, j in 0..max_len-1
        # We need i + j*d < n
        
        # Create index matrix
        # i = 0..d-1, j = 0..max_len-1
        # idx = i + j*d
        # We want idx < n
        # For j, max valid j for start i: j <= (n-1-i)//d
        # This is tricky to vectorize cleanly. Alternative:
        # Just iterate over starts but vectorize the prefix sum per start.
        
        # Actually, let's do it differently: for each start i, the chain is
        # s[i], s[i+d], s[i+2d], ...
        # We can slice: s[i::d] gives the entire chain.
        # Then compute cumulative sum and take max abs.
        # Doing this for all i in 0..d-1 in a loop is O(d * n/d) = O(n) per d,
        # total O(n^2) which for n=256 is ~65536 operations - fine.
        
        # But we can vectorize: create a matrix where row i is s[i::d] padded to max_len
        starts = np.arange(d)
        # For each start i, chain = s[i::d], length = len(s[i::d])
        # Pad to max_len with zeros
        # Use advanced indexing:
        # col_indices = np.arange(max_len)
        # idx = starts[:, None] + col_indices[None, :] * d
        # valid = idx < n
        # arr = np.where(valid, s[idx], 0)
        
        col_indices = np.arange(max_len)
        idx = starts[:, None] + col_indices[None, :] * d
        valid = idx < n
        arr = np.where(valid, s[idx], 0)
        
        # Cumulative sum along axis 1
        cumsum = np.cumsum(arr, axis=1)
        # Max abs over all positions
        m = np.max(np.abs(cumsum))
        if m > max_abs:
            max_abs = int(m)
            if max_abs >= n:  # can't get worse than n
                break
    
    return max_abs

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    n = 256
    rng = np.random.RandomState(42)
    
    # Start with random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    best_signs = signs.copy()
    best_disc = _max_ap_discrepancy(signs.tolist())
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Random flip
        idx = rng.randint(0, n)
        signs[idx] = -signs[idx]
        
        new_disc = _max_ap_discrepancy(signs.tolist())
        delta = new_disc - best_disc
        
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            best_signs = signs.copy()
            best_disc = new_disc
        else:
            # Revert
            signs[idx] = -signs[idx]
        
        T *= cooling
    
    # Final local search: try flipping each position, keep if improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            signs[i] = -signs[i]
            new_disc = _max_ap_discrepancy(signs.tolist())
            if new_disc < best_disc:
                best_signs = signs.copy()
                best_disc = new_disc
                improved = True
            else:
                signs[i] = -signs[i]
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
