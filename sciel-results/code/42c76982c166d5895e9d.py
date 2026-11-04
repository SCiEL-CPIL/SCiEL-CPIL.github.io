"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_peak_autocorr(h):
    """Compute the maximum absolute value of the linear autocorrelation at non-zero lags."""
    n = len(h)
    # Linear autocorrelation: r[k] = sum_i h[i] * h[i-k] for k > 0
    # Use FFT for efficiency
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    # Power spectrum
    power = np.abs(H) ** 2
    # Inverse FFT gives circular autocorrelation
    auto = np.fft.irfft(power, n=fft_size)
    # The first n values are r[0], r[1], ..., r[n-1]
    # r[0] is the DC term (sum of squares), exclude it
    # We want max |r[k]| for k = 1, ..., n-1
    # Also need to check negative lags, but for real sequences r[-k] = r[k]
    peak = np.max(np.abs(auto[1:n]))
    return peak


def _local_search(h, time_limit):
    """Local search: flip individual bits to reduce peak autocorrelation."""
    n = len(h)
    h = h.copy()
    current_peak = _compute_peak_autocorr(h)
    best_peak = current_peak
    best_h = h.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping each position
        for i in range(n):
            h[i] = -h[i]
            new_peak = _compute_peak_autocorr(h)
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
                no_improve_count = 0
                break
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_h, best_peak


def _convex_relaxation_init(rng, time_limit):
    """
    Use a convex relaxation approach to find a good initial sign pattern.
    We solve a quadratic program: minimize ||A h - b||^2 subject to h_i in {-1, 1}
    where A is a Toeplitz matrix representing autocorrelation constraints.
    Instead of solving the QP directly (which is hard), we use a soft-thresholding
    approach on the continuous relaxation, then round to signs.
    
    A simpler and effective heuristic: start with a random continuous vector,
    project onto the hypercube, and use gradient descent on the peak autocorrelation
    (smoothed with a soft-max).
    """
    n = N
    # Start with random continuous values in [-1, 1]
    x = rng.uniform(-1.0, 1.0, size=n)
    
    # We want to minimize the peak autocorrelation.
    # Let's use a simple iterative method:
    # 1. Compute autocorrelation.
    # 2. Identify the lag with maximum absolute value.
    # 3. Adjust x to reduce that lag.
    
    # However, for a "structural change" that might yield better global optima,
    # let's try a frequency-domain based initialization.
    # We want |H(k)|^2 to be flat for k != 0.
    # This is related to Barker sequences or Zadoff-Chu sequences.
    
    # Zadoff-Chu sequences are optimal for constant modulus and low autocorrelation.
    # They are defined as z_n = exp(-j * pi * n * (n+1) / N) for N odd, or similar.
    # For real sequences, we can take the real part or use a different construction.
    
    # Let's try to construct a sequence based on a Zadoff-Chu-like phase.
    # For N=400, which is even, we can use:
    # h_n = sign(cos(2 * pi * n * u / N)) for some integer u coprime to N.
    # Or simply use a random phase and take the sign.
    
    # A better structural idea: Use the fact that the optimal sequence for peak
    # autocorrelation minimization is often a "balanced" sequence.
    # Let's try a few deterministic patterns that are known to have low autocorrelation.
    
    patterns = []
    
    # 1. Random
    patterns.append(rng.choice([-1.0, 1.0], size=n))
    
    # 2. Alternating (worst case for autocorrelation usually, but let's see)
    patterns.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)]))
    
    # 3. Block pattern: +1 for first half, -1 for second half
    p3 = np.ones(n)
    p3[n//2:] = -1.0
    patterns.append(p3)
    
    # 4. Zadoff-Chu like (real part)
    # For N even, use u=1
    u = 1
    t = np.arange(n)
    phase = -np.pi * t * (t + 1) / N
    h_zc = np.sign(np.cos(phase))
    # Ensure it's not all zeros (sign(0)=0 in numpy, so replace 0 with 1)
    h_zc[h_zc == 0] = 1.0
    patterns.append(h_zc)
    
    # 5. Random phase cosine
    phi = rng.uniform(0, 2 * np.pi)
    h_cos = np.sign(np.cos(2 * np.pi * t / N + phi))
    h_cos[h_cos == 0] = 1.0
    patterns.append(h_cos)
    
    # Evaluate each pattern and return the best one
    best_h = None
    best_peak = np.inf
    
    for p in patterns:
        peak = _compute_peak_autocorr(p)
        if peak < best_peak:
            best_peak = peak
            best_h = p.copy()
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_peak = np.inf
    
    # Use the convex relaxation / structured initialization
    h_init = _convex_relaxation_init(rng, 10.0)
    if h_init is not None:
        peak = _compute_peak_autocorr(h_init)
        if peak < best_peak:
            best_peak = peak
            best_h = h_init.copy()
    
    # Multiple random restarts
    n_restarts = 20
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize with random signs
        h = rng.choice([-1.0, 1.0], size=N)
        
        # Run local search with time limit
        remaining = deadline - time.time() - 5.0  # 5s buffer
        if remaining <= 0:
            break
        time_limit = min(remaining, time_per_restart)
        
        h_opt, peak = _local_search(h, time_limit)
        
        if peak < best_peak:
            best_peak = peak
            best_h = h_opt
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
