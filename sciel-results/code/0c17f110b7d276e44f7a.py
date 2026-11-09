import numpy as np

N = 400
h = 1.0 / (2 * N)

def construct():
    rng = np.random.default_rng(42)
    f = rng.choice([-1.0, 1.0], size=N)
    
    R = np.correlate(f, f, mode='full') * h
    
    # Helper to get peak sidelobe
    def get_peak(R_arr):
        # Exclude index N-1
        # R_arr length 2N-1
        if N == 1:
            return 0.0
        return max(np.max(np.abs(R_arr[:N-1])), np.max(np.abs(R_arr[N:])))

    current_peak = get_peak(R)
    best_f = f.copy()
    best_peak = current_peak
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.99995
    steps = 300000
    
    # Preallocate buffer for R_new
    R_buf = np.zeros_like(R)
    
    for step in range(steps):
        i = rng.integers(N)
        
        delta = -2.0 * f[i]
        
        k_min = -i
        k_max = N - 1 - i
        
        idx_min = k_min + N - 1
        idx_max = k_max + N - 1
        
        # Calculate updates
        # j in 0..N-1
        # idx1 = j
        # idx2 = 2*i - j
        # We can compute updates vector once per i? No, f changes.
        
        # Efficient update:
        # R_slice = R[idx_min:idx_max+1]
        # updates = f[0:N] + f[2*i:N+2*i-1][::-1]?
        # Let's check indices.
        # idx2 = 2*i - j.
        # j=0 -> 2i
        # j=N-1 -> 2i - N + 1
        # This is a reversed slice of f.
        
        # f[2*i - N + 1 : 2*i + 1] is the slice?
        # Length N.
        # Start: 2*i - (N-1)
        # End: 2*i + 1 (exclusive)
        
        # Is this slice always valid?
        # We know 0 <= 2i - j < N for all j in 0..N-1.
        # Min idx: 2i - (N-1). Must be >= 0.
        # Max idx: 2i. Must be < N.
        
        # If i is small, 2i < N is ok. 2i - N + 1 might be < 0.
        # Example: N=10, i=0.
        # j=0..9.
        # idx2 = -j.
        # idx2 = 0, -1, ...
        # Wait, my previous logic:
        # k_min = -i.
        # i=0 -> k_min=0.
        # k=0 -> idx2 = 0-0=0.
        # k=9 -> idx2 = 0-9=-9. INVALID.
        
        # My previous logic: "Since k was valid, i-k is valid".
        # k ranges from -i to N-1-i.
        # If i=0, k ranges 0 to N-1.
        # i-k = -k.
        # For k=1, i-k = -1. INVALID.
        
        # So the condition "0 <= i-k < N" is NOT automatically satisfied for all k in [-i, N-1-i]?
        # Let's re-evaluate the valid range of k.
        # Condition 1: 0 <= i+k < N  => -i <= k <= N-1-i
        # Condition 2: 0 <= i-k < N  => i-N+1 <= k <= i
        # Intersection:
        # max(-i, i-N+1) <= k <= min(N-1-i, i)
        
        # So the range of k is NOT always length N.
        # It is length: min(N-1-i, i) - max(-i, i-N+1) + 1.
        
        # Let's find the intersection bounds.
        k_low = max(-i, i - N + 1)
        k_high = min(N - 1 - i, i)
        
        if k_low > k_high:
            continue # No valid k? Impossible for i in 0..N-1.
            
        # Length of range
        L = k_high - k_low + 1
        
        idx_min = k_low + N - 1
        idx_max = k_high + N - 1
        
        # Updates for k in [k_low, k_high]
        # k = j + k_low, j in 0..L-1
        # idx1 = i + k = i + j + k_low
        # idx2 = i - k = i - j - k_low
        
        # We need to extract these from f.
        # Since N=400, we can just loop or use small vector ops.
        # Vectorized:
        k_vals = np.arange(k_low, k_high + 1)
        idx1 = i + k_vals
        idx2 = i - k_vals
        
        updates = f[idx1] + f[idx2]
        
        R_buf[:] = R
        R_buf[idx_min:idx_max + 1] += delta * updates
        
        new_peak = get_peak(R_buf)
        
        delta_peak = new_peak - current_peak
        
        if delta_peak < 0 or (rng.random() < np.exp(-delta_peak / T) if T > 0 else False):
            f[i] = -f[i]
            R[:] = R_buf
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
        
        T = max(T_min, T * alpha)

    return best_f
