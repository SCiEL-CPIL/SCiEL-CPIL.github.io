"""Third autocorrelation inequality — DWT-inspired Block Flip Annealing."""
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


def _make_sine_pattern(freq, phase, amp=1.0):
    """Generate a signed step function from a sine wave."""
    x = np.linspace(0, 1, N, endpoint=False)
    raw = amp * np.sin(2 * np.pi * freq * x + phase)
    return np.sign(raw)


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing with Block Flips (DWT-inspired multi-resolution moves).
    """
    start_time = time.time()
    
    # Initialization: Start with a random sign sequence or a good sine pattern
    # Sine patterns often provide a better starting point than pure random
    x = np.linspace(0, 1, N, endpoint=False)
    best_freq = 5
    best_phase = 0.0
    best_init_score = np.inf
    best_init_h = np.ones(N)
    
    # Quick init search
    for freq in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32]:
        for phase in np.linspace(0, np.pi, 20, endpoint=False):
            h = _make_sine_pattern(freq, phase)
            s = _score(h)
            if s < best_init_score:
                best_init_score = s
                best_init_h = h.copy()
                best_freq = freq
                best_phase = phase

    h_current = best_init_h.copy()
    h_best = best_init_h.copy()
    score_current = best_init_score
    score_best = best_init_score

    # Simulated Annealing Parameters
    # Start high to allow large changes, cool down to refine
    T_start = 1.0
    T_end = 1e-6
    # Number of steps: We have ~110s. 
    # Each step involves a block flip and a score calculation.
    # Score calc is ~0.1ms. We can do ~100k steps in 10s.
    # Let's target 500k steps, checking time every 1000 steps.
    max_steps = 500000
    
    # Block sizes for "DWT-inspired" moves:
    # Small blocks (1-4) for fine tuning, Large blocks (16-100) for structural changes
    block_sizes = [1, 2, 3, 4, 8, 16, 32, 64, 128]
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if (time.time() - start_time) > DEADLINE:
            break
        
        # Cooling schedule: Geometric
        # T = T_start * (T_end / T_start) ** (step / max_steps)
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Select a block size: Weighted towards smaller blocks as we cool?
        # Or just random. Larger blocks are more disruptive.
        # Let's pick a size. Early on, larger blocks. Late on, smaller.
        if T > 0.1:
            blen = rng.choice(block_sizes, p=[0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15])
        else:
            blen = rng.choice([1, 2, 4, 8], p=[0.4, 0.3, 0.2, 0.1])
            
        # Select start index
        if blen >= N:
            start_idx = 0
            blen = N
        else:
            start_idx = rng.integers(0, N - blen + 1)
            
        # Perform block flip
        h_try = h_current.copy()
        h_try[start_idx:start_idx + blen] *= -1
        
        # Calculate score
        score_try = _score(h_try)
        
        # Acceptance criterion
        delta = score_try - score_current
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h_current = h_try
            score_current = score_try
            
            if score_try < score_best:
                score_best = score_try
                h_best = h_try.copy()
                
        # Optional: Greedy polish every 1000 steps?
        # The inspirations used greedy polish. Let's add a light greedy polish
        # if we have time, but SA with block flips is the main driver.
        # To save time, we skip explicit greedy polish inside the loop and rely
        # on the SA's ability to find low-energy states. 
        # However, a final greedy polish is cheap.
        
    # Final Greedy Polish: Try flipping single bits to see if we can improve further
    # This is fast and often helps clean up the SA result.
    h_work = h_best.copy()
    s_work = score_best
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            h_try = h_work.copy()
            h_try[i] *= -1
            s_try = _score(h_try)
            if s_try < s_work:
                s_work = s_try
                h_work = h_try
                h_best = h_work.copy()
                score_best = s_work
                improved = True

    return h_best.astype(np.float64)


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
