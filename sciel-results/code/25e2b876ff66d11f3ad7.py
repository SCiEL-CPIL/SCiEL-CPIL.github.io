"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_psl(h):
    """Compute Peak Sidelobe Level: max(|R[k]|) for k != 0 using FFT."""
    # Use FFT for faster autocorrelation computation
    # Zero-pad to at least 2*N to avoid circular convolution aliasing
    size = 1 << (2 * N).bit_length()
    # Pad h to size
    h_padded = np.zeros(size)
    h_padded[:N] = h
    # FFT
    H = np.fft.fft(h_padded)
    # Power spectrum
    power = np.abs(H) ** 2
    # Inverse FFT gives circular autocorrelation
    ac = np.fft.ifft(power).real
    # The zero lag is at index 0
    # The positive lags are 1 to N-1
    # The negative lags are N to 2N-2 (due to padding size being power of 2 >= 2N)
    # Specifically, ac[k] for k in 1..N-1 corresponds to lag k
    # ac[size-k] corresponds to lag -k
    # We want max |ac[k]| for k != 0, but within the valid range of linear autocorrelation
    # Valid lags are -(N-1) to N-1.
    # In the FFT output of length size:
    # Index 0 is lag 0.
    # Indices 1 to N-1 are lags 1 to N-1.
    # Indices size-(N-1) to size-1 are lags -(N-1) to -1.
    
    # Extract valid sidelobes
    # Positive lags: ac[1:N]
    # Negative lags: ac[size-N+1:size]
    # Note: ac is real due to symmetry, but ifft might have tiny imaginary parts, we take .real
    
    sidelobes_pos = ac[1:N]
    sidelobes_neg = ac[size-N+1:]
    
    # Combine and find max abs
    # We can just check the max of abs of these two segments
    max_psl = np.max(np.abs(sidelobes_pos))
    if len(sidelobes_neg) > 0:
        max_psl = max(max_psl, np.max(np.abs(sidelobes_neg)))
        
    return max_psl

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize with Thue-Morse sequence for better initial structure
    # Thue-Morse is 0, 1, 1, 0, 1, 0, 0, 1, ...
    # Map to -1, 1
    def thue_morse(n):
        # Generate first n bits of Thue-Morse sequence
        bits = [0]
        while len(bits) < n:
            bits += [1 - b for b in bits]
        return bits[:n]
    
    tm_bits = thue_morse(N)
    h = np.array([2*b - 1 for b in tm_bits], dtype=np.float64)
    
    current_psl = _compute_psl(h)
    best_psl = current_psl
    best_h = h.copy()
    
    # Also try a random initialization to see if it's better
    h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
    psl_rand = _compute_psl(h_rand)
    if psl_rand < best_psl:
        best_psl = psl_rand
        best_h = h_rand.copy()
        h = h_rand.copy()
        current_psl = psl_rand
    else:
        h = best_h.copy()
        current_psl = best_psl

    # Simulated Annealing / Hill Climbing with restarts
    max_restarts = 10
    steps_per_restart = 200000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        if r > 0:
            # Restart with new random sequence
            h = np.random.choice([-1, 1], size=N).astype(np.float64)
            current_psl = _compute_psl(h)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        stuck_count = 0
        max_stuck = 500  # If no improvement for 500 steps, break restart loop
        
        # Use a simple acceptance criterion: accept if better, or with small probability if worse (simulated annealing lite)
        # But for now, stick to hill climbing to be safe and fast
        
        while time.time() < deadline and stuck_count < max_stuck:
            # Pick a random index to flip
            i = np.random.randint(N)
            
            # Flip
            old_val = h[i]
            h[i] = -old_val
            
            # Compute new PSL
            new_psl = _compute_psl(h)
            
            if new_psl < current_psl:
                current_psl = new_psl
                stuck_count = 0
                if new_psl < best_psl:
                    best_psl = new_psl
                    best_h = h.copy()
            else:
                # Revert
                h[i] = old_val
                stuck_count += 1
                
    return best_h
# EVOLVE-BLOCK-END
