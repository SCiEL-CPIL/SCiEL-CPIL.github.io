import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe, which serves as an upper bound for C3.
    Uses local search to refine a random initialization.
    """
    # Initialize with random signs
    np.random.seed(42)  # Fixed seed for reproducibility in evaluation
    f = np.random.choice([-1, 1], N)
    
    # Function to compute autocorrelation efficiently using FFT
    def get_autocorr(f):
        # Zero-pad to 2*N to avoid circular convolution artifacts
        F = np.fft.fft(f, n=2*N)
        # Autocorrelation is inverse FFT of |F|^2
        acf = np.fft.ifft(F * np.conj(F)).real
        # The autocorrelation for lags k=0 to N-1 is in acf[0:N]
        # and lags k=-(N-1) to -1 are in acf[N:2N] (reversed)
        # We only care about non-zero lags
        return acf
    
    # Get initial autocorrelation
    acf = get_autocorr(f)
    # Extract non-zero lags
    # Lags 1 to N-1 are in acf[1:N]
    # Lags -1 to -(N-1) are in acf[N+1:2N] (actually acf[N-1:0:-1] or similar depending on fft output)
    # np.fft.ifft output: acf[k] is correlation at lag k for k=0..N-1, and acf[2N-k] is lag -k.
    # Let's just compute the max absolute value excluding lag 0.
    
    def get_peak_sidelobe(acf):
        # acf[0] is lag 0
        # acf[1:N] are positive lags 1..N-1
        # acf[N:2N] are negative lags -1..-(N-1) mapped to indices N..2N-1
        # Specifically, acf[N] is lag -N (which is 0 for N=400 if we padded to 800? No, pad to 2*N=800)
        # If N=400, pad to 800.
        # acf[0] = R[0]
        # acf[1] = R[1]
        # ...
        # acf[399] = R[399]
        # acf[400] = R[-400] = 0
        # acf[401] = R[-399]
        # ...
        # acf[799] = R[-1]
        
        # We want max(|R[k]|) for k != 0
        # R[k] for k=1..399 is in acf[1:399]
        # R[k] for k=-1..-399 is in acf[401:799] (indices 401 to 799)
        # Note: acf[400] is R[-400] which is 0.
        
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N] # Indices N+1 to 2N-1
        
        if len(pos_lags) > 0 and len(neg_lags) > 0:
            return np.max(np.abs(np.concatenate([pos_lags, neg_lags])))
        elif len(pos_lags) > 0:
            return np.max(np.abs(pos_lags))
        elif len(neg_lags) > 0:
            return np.max(np.abs(neg_lags))
        else:
            return 0.0

    current_peak = get_peak_sidelobe(acf)
    
    # Local Search: Try flipping bits to reduce the peak
    # We will iterate a few times.
    # For efficiency, we only evaluate the change in the autocorrelation at the specific lags that were contributing to the peak.
    # However, a full FFT update is O(N log N). N=400 is small. 120s allows many iterations.
    
    iterations = 2000
    for _ in range(iterations):
        # Find the lag with the maximum absolute value
        # We need to know which lag to target
        # Re-compute peak and its location? 
        # To save time, we can just pick a random index to flip and check if it improves the global peak.
        # Or, target the specific index that caused the max.
        
        # Let's target the index with the max absolute autocorrelation.
        # We need the actual values.
        all_lags = np.concatenate([acf[1:N], acf[N+1:2*N]])
        max_idx_in_concat = np.argmax(np.abs(all_lags))
        
        # Map back to actual lag
        if max_idx_in_concat < N - 1:
            lag = max_idx_in_concat + 1
            sign = 1
        else:
            lag = -(max_idx_in_concat - (N - 1))
            sign = -1
            
        # Current value at this lag
        target_val = np.abs(all_lags[max_idx_in_concat])
        
        # Try flipping a bit i.
        # Change in R[lag] when flipping f[i]:
        # R[lag] = sum f[j] f[j+lag]
        # If we flip f[i], terms involving f[i] change sign.
        # Terms are (i, i+lag) and (i-lag, i).
        # Delta R[lag] = -2 * f[i] * (f[i+lag] + f[i-lag]) with bounds check.
        
        # We want to minimize the new peak.
        # A simple heuristic: Flip the bit that most reduces the current target lag's magnitude,
        # and then check if the global peak decreased.
        
        best_i = -1
        best_new_peak = current_peak
        best_new_f = None
        
        # We don't want to try all 400 bits for every iteration if it's too slow, but 400*2000 = 800k ops is fine.
        # However, checking the *global* peak for each candidate i requires updating the whole ACF or checking all lags.
        # Updating whole ACF via FFT for each i is 400 * O(N log N) per iteration. 
        # 400 * 400 * log(400) ~ 1.6M ops per iteration. 2000 iterations ~ 3.2B ops. Might be too slow in Python.
        
        # Alternative: Just flip a random bit and accept if it improves the peak.
        # This is slower to converge but much faster per step.
        
        i = np.random.randint(N)
        old_f_i = f[i]
        f[i] = -old_f_i
        
        # Compute new ACF
        acf_new = get_autocorr(f)
        new_peak = get_peak_sidelobe(acf_new)
        
        if new_peak < current_peak:
            current_peak = new_peak
            acf = acf_new
            # Keep the flip
        else:
            # Revert
            f[i] = old_f_i
            # acf remains unchanged
            
    return f
# EVOLVE-BLOCK-END
