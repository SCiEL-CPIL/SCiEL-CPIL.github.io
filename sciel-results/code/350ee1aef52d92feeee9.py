"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_l1_sidelobe(x):
    """
    Compute L1 norm of autocorrelation sidelobes.
    This is a proxy for 'autoconvolution upper bound'.
    Lower is better.
    """
    # Autocorrelation
    # x is length N. Autocorrelation length 2N-1.
    # Use FFT for speed
    n_fft = 2 * N
    fft_x = np.fft.fft(x, n=n_fft)
    power = np.abs(fft_x) ** 2
    auto = np.fft.ifft(power).real
    # The autocorrelation is centered at N-1 in the 'full' correlate sense,
    # but in FFT circular convolution sense, it's shifted.
    # np.correlate(x, x, 'full') gives [gamma(-N+1)...gamma(N-1)]
    # FFT method:
    # gamma[k] = sum x[i] x[i+k]
    # Let's just use np.correlate for simplicity and correctness, N=400 is small enough.
    auto_corr = np.correlate(x, x, mode='full')
    center = N - 1
    # Sidelobes are all except the center
    sidelobes = np.concatenate([auto_corr[:center], auto_corr[center+1:]])
    return np.sum(np.abs(sidelobes))

def _generate_band_limited(N, num_freqs=10, max_freq=5, seed=42):
    """
    Generate a signed step function based on a sum of low-frequency sines.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, N, endpoint=False)
    signal = np.zeros(N)
    
    # Choose random low frequencies
    # Frequencies should be integers to ensure periodicity or low freqs
    freqs = rng.randint(1, max_freq + 1, size=num_freqs)
    amplitudes = rng.rand(num_freqs)
    phases = rng.rand(num_freqs) * 2 * np.pi
    
    for k in range(num_freqs):
        signal += amplitudes[k] * np.sin(2 * np.pi * freqs[k] * t + phases[k])
        
    # Threshold to +1/-1
    x = np.sign(signal)
    # Replace 0s with 1s (or random)
    x[x == 0] = 1
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate initial band-limited candidate
    # Try a few different seeds/frequencies to find a good starting point
    best_x = None
    best_score = float('inf')
    
    for seed in range(20):
        x = _generate_band_limited(N, num_freqs=10, max_freq=5, seed=seed)
        score = _compute_l1_sidelobe(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    # 2. Local Search Refinement
    # Flip bits to reduce the L1 sidelobe sum
    x = best_x
    current_score = best_score
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline and improved:
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Flip bit
            x[i] *= -1
            new_score = _compute_l1_sidelobe(x)
            
            if new_score < current_score - 1e-10:
                current_score = new_score
                improved = True
                no_improve_count = 0
            else:
                # Revert
                x[i] *= -1
                no_improve_count += 1
                
        if not improved:
            break
            
    return x

# EVOLVE-BLOCK-END
