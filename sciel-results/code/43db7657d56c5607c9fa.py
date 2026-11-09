"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autocorr_sidelobe_max(h):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes R[k] for k=1..N-1.
    h is an array of length N with values in {-1, 1}.
    """
    # Use FFT to compute autocorrelation efficiently
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[0] is R[0] = N. We want max of |acf[1:]|
    return np.max(np.abs(acf[1:]))

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n mapped to {-1, 1}."""
    # Thue-Morse sequence t_k = (-1)^(popcount(k))
    # We can generate it iteratively or using bit counts.
    # For n=400, this is fast.
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits set in i
        seq[i] = 1 - 2 * (bin(i).count('1') % 2)
    return seq

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Strategy:
    1. Start with Thue-Morse sequence as a strong baseline (low sidelobes).
    2. Perform local search (greedy hill climbing) with random restarts.
    3. Use a deadline guard to ensure termination within 150s.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = None
    best_val = np.inf
    
    # Initial candidate: Thue-Morse
    h = _thue_morse(N)
    current_val = _autocorr_sidelobe_max(h)
    
    if current_val < best_val:
        best_val = current_val
        best_h = h.copy()
    
    # Local search function
    def local_search(h, current_val):
        nonlocal best_h, best_val
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias in flip order
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    return
                
                # Try flipping bit i
                old_bit = h[i]
                h[i] = -h[i]
                
                # Compute new value
                new_val = _autocorr_sidelobe_max(h)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
                else:
                    # Revert
                    h[i] = old_bit
                    
                # Early exit if we reach theoretical minimum? 
                # For N=400, min max sidelobe is likely > 0.
                # If current_val == 0, we can stop.
                if current_val == 0:
                    return
                    
    # Run local search from Thue-Morse
    local_search(h, current_val)
    
    # Random restarts if time permits
    restarts = 0
    while time.time() < deadline - 5.0:  # Leave 5s for final checks
        # Generate a random sequence
        h = np.random.choice([-1, 1], size=N).astype(np.int8)
        current_val = _autocorr_sidelobe_max(h)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Run local search from this random start
        local_search(h, current_val)
        restarts += 1
        
        # Optional: Perturb best solution and search
        if best_h is not None and time.time() < deadline - 5.0:
            h_perturbed = best_h.copy()
            # Flip a few random bits
            num_flips = random.randint(1, 5)
            flip_indices = random.sample(range(N), num_flips)
            for i in flip_indices:
                h_perturbed[i] = -h_perturbed[i]
            current_val_p = _autocorr_sidelobe_max(h_perturbed)
            if current_val_p < best_val:
                best_val = current_val_p
                best_h = h_perturbed.copy()
            else:
                # Try local search from perturbed state
                local_search(h_perturbed, current_val_p)
    
    return best_h
# EVOLVE-BLOCK-END
