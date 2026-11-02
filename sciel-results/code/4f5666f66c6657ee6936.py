"""Third autocorrelation inequality — Simulated Annealing on Random Init."""
import numpy as np
import time

N = 400
DEADLINE = 110.0  # seconds, safety margin below 150s evaluator limit


def _compute_autocorr(h):
    """
    Compute the third-order autocorrelation (autoconvolution of f with f*f).
    For a discrete signal h of length N, C3[k] = sum_i h[i] * h[i+k] * h[i+2k]
    over valid i for each lag k.
    
    The "upper bound on C3" is the maximum absolute value of this triple product sum.
    """
    # Use FFT-based convolution for efficiency
    # f * f = conv(h, h), then C3 = conv(f*f, h)
    h_float = h.astype(np.float64)
    # Zero-pad to avoid circular convolution
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    # autoconv = conv(h, h)
    autoconv_fft = h_fft * h_fft
    # C3 = conv(autoconv, h)
    c3_fft = autoconv_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    # The linear convolution result of length 3N-1; c3_full[:3*N-1] is valid
    # But we only care about lags k=0,1,...,N-1 where all three h[i], h[i+k], h[i+2k] are in range
    # For lag k, the sum is over i=0 to N-1-2k, so valid for k = 0 to (N-1)//2
    # Take the first N elements as a conservative set of lags
    c3 = c3_full[:N]
    return c3


def _score(h):
    """The evaluator measures the raw value: max |C3| (lower is better)."""
    c3 = _compute_autocorr(h)
    return np.max(np.abs(c3))


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Returns a sequence of signed step heights.
    Score is the upper bound on C3 (max |C3|), lower is better.
    """
    start_time = time.time()
    
    # Initialize with random signs
    # Random sequences typically have much lower peak autocorrelation than structured ones like sines
    np.random.seed(42) # Reproducibility for debugging, remove for final? No, keep for consistency in trials
    best_h = np.random.choice([-1, 1], N)
    best_score = _score(best_h)
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # Simulated Annealing
    # Temperature schedule: Start high, cool down
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    current_h = best_h.copy()
    current_score = best_score
    
    while elapsed_ok():
        # Generate a move: flip a random block of signs
        # Block size varies to allow large jumps and fine tuning
        block_size = np.random.choice([1, 2, 4, 8, 16, 32])
        start_idx = np.random.randint(0, N - block_size + 1)
        
        new_h = current_h.copy()
        new_h[start_idx:start_idx + block_size] *= -1
        
        new_score = _score(new_h)
        
        # Acceptance probability
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / T):
                current_h = new_h
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            # Reheat if stuck? Or just stop? 
            # Let's restart with a random sequence if we have time
            if elapsed_ok():
                # Try a few random restarts
                for _ in range(10):
                    if not elapsed_ok():
                        break
                    cand_h = np.random.choice([-1, 1], N)
                    cand_score = _score(cand_h)
                    if cand_score < best_score:
                        best_score = cand_score
                        best_h = cand_h.copy()
                        current_h = best_h.copy()
                        current_score = best_score
                        T = 5.0 # Reset temp
                break

    # Final: ensure output is ±1 (signed step heights)
    best_h = np.sign(best_h).astype(np.float64)
    # If all zeros, default to ones
    if np.all(best_h == 0):
        best_h = np.ones(N)
    return best_h


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
