import numpy as np

N = 400

def construct():
    best_score = float('inf')
    best_x = None
    
    # Try different sparsities
    for M in [10, 15, 20, 25, 30]:
        # Run local search for this M
        x = local_search(N, M)
        
        # Calculate the metric (off-peak L_infty norm)
        # We assume the metric is the max absolute value of autocorrelation for tau != 0
        # Normalized to L2=1
        if x is None:
            continue
            
        # Compute autocorrelation
        # x is real, so autocorrelation is real
        # Use FFT for efficiency, though N=400 is small
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n_fft)
        power = X * np.conj(X)
        acf = np.fft.irfft(power, n_fft)
        
        # The ACF is circular. We need linear autocorrelation.
        # The first N values of the circular ACF correspond to lags 0 to N-1.
        # But the linear ACF has lags -(N-1) to N-1.
        # np.fft.irfft gives the circular correlation.
        # For a zero-padded signal of length n_fft >= 2N-1, the first N values are the linear correlation for lags 0 to N-1.
        # The values for negative lags are at the end.
        # acf[0] is lag 0.
        # acf[1:N] are lags 1 to N-1.
        # acf[n_fft - (N-1):] are lags -(N-1) to -1.
        
        # We want max |acf[tau]| for tau != 0
        # Since acf is symmetric for real signals, we just need to check 1..N-1
        peak = np.max(np.abs(acf[1:N]))
        
        if peak < best_score:
            best_score = peak
            best_x = x
            
    if best_x is None:
        # Fallback to a simple random sparse vector
        M = 20
        indices = np.random.choice(N, M, replace=False)
        x = np.zeros(N)
        x[indices] = 1.0 / np.sqrt(M)
        best_x = x
        
    return best_x

def local_search(N, M, iterations=500):
    # Initialize with random support
    support = np.random.choice(N, M, replace=False)
    signs = np.random.choice([-1, 1], size=M)
    
    x = np.zeros(N)
    x[support] = signs / np.sqrt(M)
    
    def get_acf_peak(x):
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n_fft)
        power = X * np.conj(X)
        acf = np.fft.irfft(power, n_fft)
        return np.max(np.abs(acf[1:N])), acf

    current_peak, current_acf = get_acf_peak(x)
    
    for _ in range(iterations):
        # Strategy 1: Flip a sign
        # Pick a random index in support
        idx_in_support = np.random.randint(M)
        pos = support[idx_in_support]
        
        # Try flipping
        signs[idx_in_support] *= -1
        x[pos] *= -1
        
        new_peak, _ = get_acf_peak(x)
        if new_peak < current_peak:
            current_peak = new_peak
            current_acf = _
        else:
            # Revert
            signs[idx_in_support] *= -1
            x[pos] *= -1
            
        # Strategy 2: Swap support element
        # Pick a random index in support to remove
        idx_in_support = np.random.randint(M)
        pos_remove = support[idx_in_support]
        
        # Pick a random index outside support to add
        # To avoid O(N) check every time, we can just pick random and check
        # But for N=400, it's fine.
        # We need a random index not in support.
        # Simple way: pick random, if in support, pick again.
        # Or use a set.
        support_set = set(support.tolist())
        pos_add = np.random.randint(N)
        while pos_add in support_set:
            pos_add = np.random.randint(N)
            
        # Perform swap
        x[pos_remove] = 0
        # Determine sign for new element. Try both? Or random?
        # Let's try random sign first
        new_sign = np.random.choice([-1, 1])
        x[pos_add] = new_sign / np.sqrt(M)
        
        new_peak, _ = get_acf_peak(x)
        if new_peak < current_peak:
            current_peak = new_peak
            current_acf = _
            # Update support and signs
            support[idx_in_support] = pos_add
            signs[idx_in_support] = new_sign
            support_set.remove(pos_remove)
            support_set.add(pos_add)
        else:
            # Revert
            x[pos_remove] = signs[idx_in_support] / np.sqrt(M)
            x[pos_add] = 0
            
    return x
