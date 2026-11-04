"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr_peak(f):
    """
    Compute the maximum absolute value of the autocorrelation at non-zero lags.
    f is a vector of length N with values in {-1, 1}.
    """
    n = len(f)
    # Use FFT for efficient autocorrelation
    # Size must be at least 2*N to avoid circular convolution aliasing
    size = 1
    while size < 2 * n:
        size *= 2
        
    # Zero-pad f
    f_pad = np.zeros(size)
    f_pad[:n] = f
    
    # FFT
    F = np.fft.fft(f_pad)
    
    # Power spectrum
    S = np.abs(F) ** 2
    
    # Inverse FFT to get autocorrelation
    # Note: np.fft.ifft returns complex, take real part
    acf = np.fft.ifft(S).real
    
    # acf[k] is the autocorrelation at lag k.
    # For real signals, acf is symmetric.
    # We want max |acf[k]| for k in 1..N-1.
    # acf[0] is the DC term (sum of squares), which is N for +-1.
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _get_peak_lag(f):
    """Return the lag index k (1..N-1) with the maximum absolute autocorrelation."""
    n = len(f)
    size = 1
    while size < 2 * n:
        size *= 2
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad)
    S = np.abs(F) ** 2
    acf = np.fft.ifft(S).real
    if n <= 1:
        return 0
    # Find max abs in 1..n-1
    max_val = -1
    max_idx = 0
    for k in range(1, n):
        val = abs(acf[k])
        if val > max_val:
            max_val = val
            max_idx = k
    return max_idx

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Strategy: Full support signed step function (Ising spin chain)
    # Minimize peak autocorrelation
    
    # Initialize with random signs
    np.random.seed(42)
    f = np.random.choice([-1, 1], size=N)
    
    best_f = f.copy()
    best_score = _compute_autocorr_peak(best_f)
    
    # Local Search: Greedy descent targeting the peak lag
    # We repeatedly find the lag with the max autocorrelation and try to flip signs
    # to reduce it.
    
    while time.time() < deadline - 5.0:
        # Find the current peak lag
        k_peak = _get_peak_lag(f)
        if k_peak == 0:
            break
            
        improved = False
        # Try flipping each sign to see if it reduces the peak at k_peak
        # We check the actual global peak after the flip to ensure we don't create a new larger peak elsewhere.
        # To speed up, we can first check if the flip reduces |R[k_peak]|.
        
        # Pre-compute terms for R[k_peak] update
        # R[k] = sum f[i] f[i+k]
        # If we flip f[i], R[k] changes by -2 * f[i] * (f[i-k] + f[i+k])
        
        # We will iterate through indices. 
        # To avoid O(N^2) if we recompute FFT for every flip, we can be smart.
        # But N=400 is small. FFT is ~400*log(800) ~ 4000 ops.
        # 400 flips * 4000 ops = 1.6M ops per epoch.
        # 1.6M ops is very fast in Python/NumPy if vectorized, but here we are doing scalar updates in a loop.
        # Let's optimize: 
        # 1. Identify candidate flips that reduce |R[k_peak]|.
        # 2. Among those, pick the one that reduces it the most.
        # 3. Apply it.
        
        best_gain = -1e-9
        best_idx = -1
        
        # Current R[k_peak]
        # We can compute R[k_peak] directly or from FFT. 
        # Let's just compute the "delta" for each i.
        
        # f[i-k] and f[i+k] need boundary handling (wrap around? No, linear autocorrelation).
        # The FFT-based autocorrelation assumes zero-padding, so it's linear.
        # So f[i-k] is 0 if i-k < 0 or i-k >= N.
        
        # Let's vectorize the calculation of the change in R[k_peak] for all i.
        # delta_R[i] = -2 * f[i] * (f[i-k] + f[i+k])
        
        # Create shifted arrays
        # Note: np.roll wraps around, we need zero-padding logic.
        # Since N is small, we can do it with a loop or simple slicing.
        
        # f_shifted_minus: f[i-k]
        # f_shifted_plus: f[i+k]
        
        # For linear autocorrelation, if i-k < 0, the term is 0.
        # If i+k >= N, the term is 0.
        
        term_minus = np.zeros(N)
        term_plus = np.zeros(N)
        
        if k_peak > 0:
            # i-k >= 0 => i >= k_peak
            term_minus[k_peak:] = f[:N-k_peak]
            # i+k < N => i < N-k_peak
            term_plus[:N-k_peak] = f[k_peak:]
            
        # delta_R[i] = -2 * f[i] * (term_minus[i] + term_plus[i])
        delta_R = -2.0 * f * (term_minus + term_plus)
        
        # We want to flip f[i] if it reduces |R[k_peak]|.
        # Current R[k_peak] can be computed as sum(f * f_shifted_plus) ... 
        # Actually, let's just get the current value from the FFT we did in _get_peak_lag? 
        # We didn't return it. Let's recompute or store it.
        # For efficiency, let's just check if the flip is "good" for the peak.
        # A flip is good if it moves R[k_peak] towards 0.
        # i.e., sign(R[k_peak]) * delta_R[i] < 0.
        
        # We need R[k_peak].
        # R[k_peak] = sum(f[i] * f[i+k_peak])
        R_k_peak = np.sum(f[:N-k_peak] * f[k_peak:])
        
        # Condition for improvement in R[k_peak]:
        # If R_k_peak > 0, we want delta_R < 0.
        # If R_k_peak < 0, we want delta_R > 0.
        # So: R_k_peak * delta_R < 0
        
        candidates = np.where(R_k_peak * delta_R < 0)[0]
        
        if len(candidates) > 0:
            # Among candidates, pick the one with the largest magnitude of change (steepest descent)
            # Magnitude of change is |delta_R[i]|
            # We want to maximize |delta_R[i]| subject to the sign condition.
            gains = np.abs(delta_R[candidates])
            max_gain_idx_local = np.argmax(gains)
            best_idx = candidates[max_gain_idx_local]
            best_gain = gains[max_gain_idx_local]
            
            if best_gain > 1e-9:
                # Apply the flip
                f[best_idx] *= -1
                improved = True
                
                # Check if global peak improved
                new_score = _compute_autocorr_peak(f)
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                else:
                    # If global peak didn't improve (maybe another lag got worse), 
                    # we might still keep the flip if it's part of a sequence, 
                    # but for greedy, we usually revert if the main objective (global peak) doesn't improve.
                    # However, reducing the current peak is a good heuristic.
                    # Let's be strict: only keep if global peak improves or stays same?
                    # The task says "lower is better".
                    # If new_score == best_score, we might keep it to escape local minima? 
                    # Let's revert if it doesn't strictly improve the global peak to be safe,
                    # unless we are stuck.
                    if new_score >= best_score:
                        f[best_idx] *= -1
                        improved = False
                        # If we can't improve global peak, try random flips?
                        # For now, break out of this epoch to avoid infinite loops on plateaus.
                        # Or try a different strategy: random restart if stuck.
        
        if not improved:
            # Stuck in local minimum for greedy peak reduction.
            # Try a few random flips to escape
            escaped = False
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                idx = np.random.randint(N)
                f[idx] *= -1
                new_score = _compute_autocorr_peak(f)
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    escaped = True
                    break
                else:
                    f[idx] *= -1 # Revert
            
            if not escaped:
                # If we can't escape, restart with a new random seed?
                # Or just break and return best_f.
                # Given the time, let's do a few random restarts if we have time.
                if time.time() < deadline - 20.0:
                    # Random restart
                    np.random.seed(int(time.time()))
                    f = np.random.choice([-1, 1], size=N)
                    current_score = _compute_autocorr_peak(f)
                    if current_score < best_score:
                        best_score = current_score
                        best_f = f.copy()
                    # Continue loop to refine this new start
                else:
                    break

    return best_f
# EVOLVE-BLOCK-END
