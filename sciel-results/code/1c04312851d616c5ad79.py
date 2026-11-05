import numpy as np
import time

N = 400

def construct():
    """
    Construct a signed step function (400 intervals) to minimize the upper bound on C3.
    
    This implementation uses Simulated Annealing with contiguous block flips.
    The metric is the L1 norm of the aperiodic autocorrelation sidelobes.
    Block flips allow for larger structural changes compared to single-bit flips,
    potentially escaping local minima more effectively in the autocorrelation landscape.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit)

    rng = np.random.RandomState(42)

    def compute_l1_sidelobe(h):
        """
        Compute the L1 norm of the aperiodic autocorrelation sidelobes.
        h: array of +/- 1 of length N
        Returns: sum_{tau=1}^{N-1} |r(tau)|
        """
        # Aperiodic autocorrelation
        # r(tau) = sum_{i=0}^{N-1-tau} h[i] h[i+tau]
        # We can compute this using FFT for speed, or direct summation.
        # FFT approach:
        # Let h_ext be h padded with zeros to length 2N.
        # Convolve h_ext with reversed h_ext?
        # Actually, linear correlation can be computed via FFT:
        # H = fft(h, 2*N)
        # R = ifft(H * conj(H))
        # The first N elements of R (shifted appropriately) give the correlation.
        # Specifically, for real h:
        # R[k] for k=0..N-1 corresponds to r(k).
        # R[N+1..] corresponds to r(-k).
        
        H = np.fft.rfft(h, n=2 * N)
        R = np.fft.irfft(H * np.conj(H), n=2 * N)
        
        # R[0] = sum h[i]^2 = N
        # R[1] = sum h[i]h[i+1] ... wait, the order from irfft might be different.
        # For autocorrelation, if we do ifft(|fft(h)|^2), the result is circular autocorrelation.
        # To get aperiodic, we zero-pad to at least 2N-1.
        # With n=2*N, the first N points are the aperiodic autocorrelation r(0) to r(N-1).
        # Let's verify.
        # h = [1, 1, 0, 0] (padded to 4? No, pad to 2N).
        # Let's just use the direct O(N^2) if N is small, but N=400 is 160k ops, which is fast.
        # O(N^2) is ~1.6e5 operations per evaluation.
        # If we do 10,000 evaluations, that's 1.6e9 operations. Might be slow in Python.
        # FFT is O(N log N) ~ 400 * 9 = 3600 ops. Much faster.
        
        # The irfft result R[0:N] gives r(0), r(1), ..., r(N-1).
        # We want sum_{tau=1}^{N-1} |r(tau)|.
        sidelobes = np.abs(R[1:N])
        return np.sum(sidelobes)

    def get_block_flip_indices(idx, length):
        """
        Generate start and end indices for a contiguous block flip.
        """
        # Random start
        start = rng.randint(0, N)
        # Random length (1 to N/2)
        blen = rng.randint(1, N // 2 + 1)
        end = (start + blen) % N
        return start, end

    # Initialize with a few different strategies
    # 1. Random
    h_rand = rng.choice([-1.0, 1.0], size=N)
    # 2. Thue-Morse (low autocorrelation properties)
    # Thue-Morse sequence: t(n) = sum bits of n mod 2
    h_tm = np.array([(-1) ** np.bitwise_count(np.arange(N)) for i in range(N)], dtype=np.float64)
    # 3. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    
    # Evaluate initial candidates
    scores = {
        'rand': compute_l1_sidelobe(h_rand),
        'tm': compute_l1_sidelobe(h_tm),
        'alt': compute_l1_sidelobe(h_alt)
    }
    
    best_name = min(scores, key=scores.get)
    best_h = {'rand': h_rand, 'tm': h_tm, 'alt': h_alt}[best_name].copy()
    best_score = scores[best_name]
    
    # Simulated Annealing with Block Flips
    # Temperature schedule
    T = 50.0  # Higher initial temperature to allow large changes
    cooling_rate = 0.995
    min_T = 1.0
    
    steps_per_temp = 200
    
    while time.time() < deadline - 10.0 and T > min_T:
        for _ in range(steps_per_temp):
            if time.time() > deadline - 10.0:
                break
            
            # Perform a block flip
            start, end = get_block_flip_indices(0, 0) # dummy call to use rng
            # Correct generation
            start = rng.randint(0, N)
            blen = rng.randint(1, N // 2 + 1)
            
            # Create a mask for the flip
            # We need to flip indices [start, start+blen) modulo N
            idxs = np.arange(start, start + blen) % N
            
            # Apply flip
            best_h[idxs] *= -1.0
            
            new_score = compute_l1_sidelobe(best_h)
            delta = new_score - best_score
            
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                best_score = new_score
            else:
                # Reject: undo the flip
                best_h[idxs] *= -1.0
        
        T *= cooling_rate
        
    return best_h
