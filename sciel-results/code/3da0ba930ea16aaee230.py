import numpy as np

N = 400
h = 1.0 / (2 * N)  # interval width: total length 1/2, so h = (1/2)/400 = 1/800

def autocorr_peak(f):
    """Compute max |autocorrelation| for step function f on [-1/4, 1/4]."""
    # f is array of length N, values at each interval
    # Autocorrelation R(k) = h * sum_i f[i] * f[i+k] for k from -(N-1) to (N-1)
    # We compute for all lags
    R = np.correlate(f, f, mode='full') * h
    return np.max(np.abs(R))

def construct():
    # New Approach: Local Search (Greedy Flip) to minimize sidelobe peak
    # Metric: max |R(k)| for k != 0
    
    rng = np.random.default_rng(42)
    f = rng.choice([-1.0, 1.0], size=N)
    
    # Precompute autocorrelation
    # np.correlate returns lags from -(N-1) to N-1
    # Index N-1 corresponds to lag 0
    R = np.correlate(f, f, mode='full') * h
    
    # Indices for lags != 0
    # R has length 2N-1. Center is N-1.
    # Lags are -N+1 to N-1.
    # We want to exclude the center.
    mask = np.ones(len(R), dtype=bool)
    mask[N-1] = False
    
    for step in range(2000):
        # Find the lag with the maximum absolute sidelobe
        R_masked = R.copy()
        R_masked[~mask] = -np.inf
        k_idx = np.argmax(np.abs(R_masked))
        
        if k_idx == N - 1:
            break
            
        # Get the lag value
        # k_idx corresponds to lag k_idx - (N-1)
        k_lag = k_idx - (N - 1)
        
        # Current peak value
        peak_val = R[k_idx]
        
        # We want to flip an index i to reduce |peak_val|
        # Change in R(k) when flipping f[i]:
        # R(k) = sum_j f[j] f[j+k]
        # Terms involving f[i]: f[i]f[i+k] and f[i-k]f[i]
        # Delta R(k) = -2 f[i] (f[i+k] + f[i-k])
        
        # We need to find i that maximizes the reduction in |R(k)|
        # If peak_val > 0, we want Delta R(k) < 0.
        # If peak_val < 0, we want Delta R(k) > 0.
        # So we want sign(peak_val) * Delta R(k) < 0.
        # sign(peak_val) * (-2 f[i] (f[i+k] + f[i-k])) < 0
        # -sign(peak_val) f[i] (f[i+k] + f[i-k]) < 0
        # sign(peak_val) f[i] (f[i+k] + f[i-k]) > 0
        
        s = np.sign(peak_val)
        if s == 0:
            s = 1.0
            
        # Calculate the score for each possible flip i
        # We need f[i+k] and f[i-k]
        # Handle boundaries: if i+k >= N or i-k < 0, the term is 0?
        # In np.correlate mode='full', the sums are over valid indices.
        # If we flip f[i], it affects R(k) only if both i and i+k are in [0, N-1] (for the forward part)
        # or i and i-k are in [0, N-1] (for the backward part).
        
        # Let's compute the vector of "gains" for each i
        # gain[i] = s * f[i] * (f[i+k] + f[i-k])
        # We want to pick i with max gain[i]
        
        gain = np.zeros(N)
        
        # Forward part: i and i+k valid
        # i in [0, N-1-k] if k > 0
        # i in [0, N-1+ k] if k < 0?
        # Let's just use np.roll or slicing carefully.
        
        # Simpler: iterate or use vectorized ops with padding
        # f_padded = np.pad(f, (N-1, N-1), mode='constant')? No, that changes indices.
        
        # Let's use the property:
        # f[i+k] is f shifted by -k
        # f[i-k] is f shifted by k
        
        if k_lag > 0:
            # i ranges 0 to N-1-k_lag for f[i+k_lag] to be valid
            # i ranges k_lag to N-1 for f[i-k_lag] to be valid
            # Union of ranges?
            # Actually, for a specific lag k, R(k) = sum_{i=0}^{N-1-k} f[i] f[i+k]
            # If we flip f[i], it affects this sum if i is in [0, N-1-k] AND i+k is in [0, N-1] (always true if i in range)
            # AND it affects the symmetric sum? No, np.correlate handles the full sum.
            # R(k) = sum_{j} f[j] f[j+k] where indices are valid.
            
            # If we flip f[i]:
            # It appears in term j=i (if i+k valid) as f[i]f[i+k]
            # It appears in term j=i-k (if i-k valid) as f[i-k]f[i]
            
            # So Delta R(k) = -2 f[i] ( (f[i+k] if i+k < N else 0) + (f[i-k] if i-k >= 0 else 0) )
            
            f_fwd = np.zeros(N)
            f_bwd = np.zeros(N)
            
            if k_lag > 0:
                # i+k valid for i < N - k_lag
                f_fwd[:N-k_lag] = f[k_lag:]
                # i-k valid for i >= k_lag
                f_bwd[k_lag:] = f[:N-k_lag]
            else:
                # k_lag < 0, let m = -k_lag > 0
                m = -k_lag
                # i+k_lag = i-m valid for i >= m
                f_fwd[m:] = f[:N-m]
                # i-k_lag = i+m valid for i < N-m
                f_bwd[:N-m] = f[m:]
                
            term = f_fwd + f_bwd
            gain = s * f * term
            
            # Find best i
            i_best = np.argmax(gain)
            
            # If gain is negative, flipping might increase peak?
            # We only flip if it improves the metric.
            # Improvement means reducing |R(k)|.
            # If gain > 0, Delta R(k) has opposite sign to peak, so |R(k)| decreases.
            # If gain <= 0, flipping increases |R(k)| or keeps it same.
            
            if gain[i_best] > 0:
                f[i_best] *= -1
                # Update R
                # R -= 2 f_old[i] (f[i+k] + f[i-k])
                # Note: f[i] has already been flipped in f, so f[i] is new value.
                # f_old = -f_new
                # Delta = -2 f_old (term) = 2 f_new (term)
                # R_new = R_old + 2 f_new (term)
                # Wait.
                # R_old = ... + f_old f_old_k + ...
                # R_new = ... + f_new f_old_k + ... = ... - f_old f_old_k + ...
                # Change = -2 f_old f_old_k
                # f_old = -f_new
                # Change = 2 f_new f_old_k
                # So R += 2 f[i_best] * term[i_best]
                
                R += 2 * f[i_best] * term[i_best]
            else:
                # No move improves the peak lag. Try a random move or break?
                # To escape local minima, do a random flip occasionally
                if step % 10 == 0:
                    i_rand = rng.integers(N)
                    f[i_rand] *= -1
                    # Recalculate R? Or update?
                    # Updating R for random flip is same logic
                    # Recompute term for this i
                    if k_lag > 0:
                        t = 0
                        if i_rand + k_lag < N: t += f[i_rand + k_lag]
                        if i_rand - k_lag >= 0: t += f[i_rand - k_lag]
                        # Note: f[i_rand] is already flipped
                        # R += 2 f_new[i] * t
                        R += 2 * f[i_rand] * t
                    else:
                        m = -k_lag
                        t = 0
                        if i_rand - m >= 0: t += f[i_rand - m]
                        if i_rand + m < N: t += f[i_rand + m]
                        R += 2 * f[i_rand] * t

    return f
