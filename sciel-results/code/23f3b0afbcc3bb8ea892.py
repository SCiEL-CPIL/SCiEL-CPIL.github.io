import numpy as np
import time

N = 400

def _compute_ac(h):
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score_from_ac(ac, S):
    if abs(S) < 1e-9:
        return 1e10
    return np.max(np.abs(ac)) / (S * S)

def _update_ac(h, ac, j):
    """
    Returns new ac array after flipping h[j].
    Also returns the change in sum.
    """
    v = h[j]
    # delta_ac = -2 * v * (h[j+k] + h[j-k]) for k != 0
    # delta_ac[0] = 0
    
    # k ranges 0 to 2N-2
    # h[j+k] is non-zero if 0 <= j+k < N
    # h[j-k] is non-zero if 0 <= j-k < N
    
    # We can construct the vector using slicing or zeros.
    # Since N is small (400), we can just do it in a loop or vectorized with masks.
    # Vectorized:
    k = np.arange(2 * N - 1)
    
    # Index 1: j + k
    idx1 = j + k
    # Index 2: j - k
    idx2 = j - k
    
    # Create valid masks
    mask1 = (idx1 >= 0) & (idx1 < N)
    mask2 = (idx2 >= 0) & (idx2 < N)
    
    # Get values, padding with 0
    # Use np.take with mode='clip' or just indexing with masks?
    # Simpler: create arrays of zeros and fill valid parts.
    # But we need to add them.
    
    val1 = np.zeros(2 * N - 1)
    val2 = np.zeros(2 * N - 1)
    
    if np.any(mask1):
        val1[mask1] = h[idx1[mask1]]
    if np.any(mask2):
        val2[mask2] = h[idx2[mask2]]
        
    delta = -2 * v * (val1 + val2)
    delta[0] = 0 # k=0 correction
    
    new_ac = ac + delta
    new_S = np.sum(h) - 2 * v # Wait, h[j] is not flipped yet in this function?
    # Let's assume this function just calculates the delta.
    return new_ac, -2 * v

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    # Initialize h
    i = np.arange(N)
    h = np.sign(np.cos(2 * np.pi * i * 3 / N))
    if abs(np.sum(h)) < 1e-6:
        h = np.ones(N)
        
    S = np.sum(h)
    ac = _compute_ac(h)
    current_score = _score_from_ac(ac, S)
    
    best_h = h.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    T_start = 1e-5
    T_end = 1e-9
    n_steps = 1000000 # Try more steps with O(N) update
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
            
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        j = rng.integers(0, N)
        v = h[j]
        
        # Calculate new ac and delta S
        # Reuse logic from _update_ac but inline for speed
        k = np.arange(2 * N - 1)
        idx1 = j + k
        idx2 = j - k
        
        # Optimization: Pre-allocate val1, val2? No, N is small.
        # But creating masks every time is expensive.
        # Alternative:
        # h is length N.
        # We can just compute the sum directly?
        # No, we need the whole vector to find max.
        
        # Let's try a different vectorization:
        # delta[k] = -2*v * (h[j+k] + h[j-k])
        
        # Use np.roll?
        # h[j+k] is h shifted left by j?
        # If we consider h padded to length 2N-1?
        
        # Let's stick to the mask method but optimize it.
        # Actually, for N=400, a simple Python loop for the update might be slower than NumPy overhead.
        # Let's use the mask method.
        
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        # To avoid creating zeros arrays and filling, we can do:
        # val_sum = np.zeros(2*N-1)
        # val_sum[mask1] += h[idx1[mask1]]
        # val_sum[mask2] += h[idx2[mask2]]
        
        # This is still 4 array ops.
        
        # Let's try FFT again?
        # 1M steps * 10us = 10s.
        # O(N) update:
        # 1M steps * (mask creation + fill) ~ 1M * 20us = 20s?
        # It's close.
        
        # Let's try to make the O(N) update faster.
        # We can pre-compute the indices? No, j changes.
        
        # Let's just use FFT for robustness and simplicity, but increase steps if fast enough.
        # I'll revert to FFT in the loop but optimize the rest.
        
        h[j] = -h[j]
        S_new = S + 2 * h[j] # h[j] is new value. Old was -h[j]. S_new = S_old - 2*Old = S_old + 2*New
        
        ac_new = _compute_ac(h)
        score_new = _score_from_ac(ac_new, S_new)
        
        delta = score_new - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            S = S_new
            ac = ac_new
            current_score = score_new
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            h[j] = -h[j]
            
    return best_h
