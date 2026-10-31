"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def score(x):
        """Compute max |autocorr| / (sum)^2. Lower is better."""
        S = np.sum(x)
        if abs(S) < 1e-12:
            return float('inf')
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        X = np.fft.rfft(x, size)
        acf = np.fft.irfft(X * np.conj(X), size)[:n]
        max_abs = np.max(np.abs(acf))
        return max_abs / (S * S)

    # Strategy: Start with all +1s (dense), then flip signs to reduce peak autocorr
    # while maintaining a large sum.
    
    best_x = np.ones(N)
    best_score = score(best_x)
    
    # Try different approaches:
    # 1. Start dense, flip signs greedily to reduce max autocorr
    # 2. Try various densities with sign optimization
    
    # Approach 1: Start with all +1s, iteratively flip the sign of the element
    # that most reduces the maximum autocorrelation
    
    if time.time() < deadline - 20:
        x = np.ones(N).copy()
        s = score(x)
        
        # Precompute autocorrelation for efficiency
        def compute_acf(x):
            n = len(x)
            size = 1
            while size < 2 * n - 1:
                size *= 2
            X = np.fft.rfft(x, size)
            acf = np.fft.irfft(X * np.conj(X), size)[:n]
            return acf
        
        acf = compute_acf(x)
        max_abs = np.max(np.abs(acf))
        S = np.sum(x)
        s = max_abs / (S * S)
        
        # Iteratively flip signs to reduce max autocorr
        # Focus on the lag where max autocorr occurs
        for iteration in range(50000):
            if time.time() > deadline - 10:
                break
            
            # Find the lag with maximum absolute autocorrelation
            max_lag = np.argmax(np.abs(acf))
            max_val = acf[max_lag]
            
            # Try flipping each element and see which reduces the max autocorr most
            # This is expensive, so use a heuristic: flip elements that contribute
            # most to the peak autocorrelation at the worst lag
            
            if max_lag == 0:
                # Peak at lag 0: this is sum(x_i^2) = number of non-zeros
                # To reduce this, we'd need to zero out elements, but that also
                # reduces the sum. Skip for now.
                # Instead, try to reduce other lags
                # Find next worst lag
                abs_acf = np.abs(acf).copy()
                abs_acf[0] = 0
                max_lag = np.argmax(abs_acf)
                if abs_acf[max_lag] < 1e-10:
                    break
            else:
                # For lag k > 0, autocorr[k] = sum_i x[i] * x[i+k]
                # Flipping x[j] changes autocorr[k] by -2*x[j]*x[j+k] (if j+k < N)
                # and -2*x[j-k]*x[j] (if j-k >= 0)
                # We want to find j that most reduces |autocorr[max_lag]|
                
                # Compute the contribution vector
                contrib = np.zeros(N)
                k = max_lag
                # For each j, flipping x[j] changes acf[k] by:
                # -2 * x[j] * x[j+k] (from term i=j)
                # -2 * x[j-k] * x[j] (from term i=j-k, if j-k >= 0)
                
                # Term 1: i = j, so x[j] * x[j+k]
                valid1 = np.arange(N - k)
                if len(valid1) > 0:
                    contrib[valid1] += -2 * x[valid1] * x[valid1 + k]
                
                # Term 2: i = j - k, so x[j-k] * x[j]
                valid2 = np.arange(k, N)
                if len(valid2) > 0:
                    contrib[valid2] += -2 * x[valid2 - k] * x[valid2]
                
                # We want to minimize |acf[k] + delta| where delta is the change
                # Current acf[k] = max_val
                # After flip: new_acf[k] = max_val + contrib[j]
                # We want to minimize |max_val + contrib[j]|
                
                # Find the j that most reduces the absolute value
                new_vals = max_val + contrib
                # We want the most negative contribution if max_val > 0,
                # or most positive if max_val < 0
                if max_val > 0:
                    best_j = np.argmin(new_vals)
                else:
                    best_j = np.argmax(new_vals)
                
                # Check if this flip actually helps
                old_score = s
                x[best_j] = -x[best_j]
                acf = compute_acf(x)
                S = np.sum(x)
                if abs(S) < 1e-12:
                    x[best_j] = -x[best_j]
                    continue
                max_abs = np.max(np.abs(acf))
                s = max_abs / (S * S)
                
                if s < best_score:
                    best_score = s
                    best_x = x.copy()
                else:
                    # Revert
                    x[best_j] = -x[best_j]
                    acf = compute_acf(x)
                    s = old_score
        
        # Verify best
        final_score = score(best_x)
        if final_score < best_score:
            best_score = final_score
    
    # Approach 2: Try sparse sequences with optimized signs
    if time.time() < deadline - 10:
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 10:
                break
            for trial in range(20):
                if time.time() > deadline - 10:
                    break
                rng = np.random.RandomState(trial * 1000 + int(density * 10000))
                mask = rng.random(N) < density
                # Start with all +1 on mask
                x = np.zeros(N)
                x[mask] = 1.0
                s = score(x)
                
                # Local search: flip signs
                for _ in range(5000):
                    if time.time() > deadline - 10:
                        break
                    idx = np.random.randint(N)
                    if x[idx] == 0:
                        continue
                    x[idx] = -x[idx]
                    new_s = score(x)
                    if new_s < s:
                        s = new_s
                        if s < best_score:
                            best_score = s
                            best_x = x.copy()
                    else:
                        x[idx] = -x[idx]
    
    return best_x
