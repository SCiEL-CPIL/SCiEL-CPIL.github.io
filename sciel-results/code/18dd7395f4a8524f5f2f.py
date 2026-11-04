"""Combined program: Thue-Morse init + FFT-based PSL scoring + greedy refinement."""
import numpy as np
import time

N = 400
TIME_LIMIT = 145  # safety margin before 150s kill


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n: (-1)^{popcount(i)}"""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count bits
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1.0 if count % 2 == 0 else -1.0
    return seq


def _compute_psl(heights):
    """Compute peak sidelobe level: max |autocorrelation| for lags 1..N-1."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # FFT size: next power of 2 >= 2n
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is zero-lag, acf[1..n-1] are positive lags
    # acf[fft_size - (n-1) .. fft_size-1] are negative lags (by symmetry, same as positive)
    # For real input, acf is symmetric, so max over 1..n-1 suffices
    return np.max(np.abs(acf[1:n]))


def _greedy_refine(h, max_rounds=50, time_budget=None):
    """Greedy sign-flip refinement: flip the sign that most reduces PSL."""
    h = np.array(h, dtype=np.float64)
    current_psl = _compute_psl(h)
    
    start = time.time()
    for round_idx in range(max_rounds):
        if time_budget is not None and time.time() - start > time_budget:
            break
        
        best_gain = 0.0
        best_idx = -1
        
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_psl = _compute_psl(h)
            h[i] = old_val
            
            gain = current_psl - new_psl
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx < 0:
            break
        
        h[best_idx] = -h[best_idx]
        current_psl -= best_gain
    
    return h, current_psl


def construct():
    start_time = time.time()
    
    # Initialize with Thue-Morse sequence
    h = _thue_morse(N)
    current_psl = _compute_psl(h)
    
    best_h = h.copy()
    best_psl = current_psl
    
    # Greedy refinement from Thue-Morse
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        h_refined, psl_refined = _greedy_refine(h, max_rounds=100, time_budget=remaining - 15)
        if psl_refined < best_psl:
            best_psl = psl_refined
            best_h = h_refined.copy()
    
    # Try a few random restarts with greedy refinement if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 30:
        np.random.seed(42)
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            h_rand = np.random.choice([-1.0, 1.0], size=N)
            psl_rand = _compute_psl(h_rand)
            if psl_rand < best_psl:
                h_rand, psl_rand = _greedy_refine(h_rand, max_rounds=50, time_budget=15)
                if psl_rand < best_psl:
                    best_psl = psl_rand
                    best_h = h_rand.copy()
    
    # Final check
    final_psl = _compute_psl(best_h)
    if final_psl < best_psl:
        best_psl = final_psl
    
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"PSL: {_compute_psl(h)}")
