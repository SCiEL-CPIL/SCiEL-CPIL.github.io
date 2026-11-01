"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(h):
    """
    Compute the C3 autoconvolution upper bound.
    For a ±1 sequence, this is based on the third moment of the autocorrelation.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3


def _parallel_tempering(h_init, deadline, rng, n_replicas=10):
    """
    Parallel Tempering (Replica Exchange) with Block Flips.
    """
    # Temperatures: geometric progression from low to high
    # Low T: precise search, High T: broad exploration
    T_min = 0.01
    T_max = 50.0
    temperatures = np.geomspace(T_min, T_max, n_replicas)
    
    # Initialize replicas
    # Replica 0 starts with the best initial guess
    # Others start random
    replicas = []
    energies = []
    
    h0 = h_init.copy()
    e0 = _compute_c3_bound(h0)
    replicas.append(h0)
    energies.append(e0)
    
    for i in range(1, n_replicas):
        h_rand = rng.choice([-1.0, 1.0], size=N)
        e_rand = _compute_c3_bound(h_rand)
        replicas.append(h_rand)
        energies.append(e_rand)
        
    best_h = replicas[0].copy()
    best_e = e0
    
    # Precompute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Main Loop
    # We need to balance exploration (swaps) and exploitation (moves)
    # Block flip size distribution: mix of small and large blocks
    # Small blocks (1-5) for fine tuning, Large blocks (10-50) for structural changes
    
    while time.time() < deadline:
        # 1. Perform moves in each replica
        for i in range(n_replicas):
            # Check time frequently
            if time.time() > deadline:
                return best_h, best_e
            
            h = replicas[i]
            e = energies[i]
            T = temperatures[i]
            
            # Generate a block flip move
            # Random start
            start = rng.integers(0, N)
            # Random length: biased towards smaller blocks but allow larger
            # Use a power law or similar distribution
            # Let's try: 50% chance small (1-10), 50% chance medium (10-50)
            if rng.random() < 0.5:
                length = rng.integers(1, 11)
            else:
                length = rng.integers(11, 51)
                
            end = (start + length) % N
            
            # Apply flip
            if start + length <= N:
                h[start:start+length] *= -1
            else:
                h[start:] *= -1
                h[:end] *= -1
                
            new_e = _compute_c3_bound(h)
            delta_e = new_e - e
            
            # Metropolis acceptance
            if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
                energies[i] = new_e
                if new_e < best_e:
                    best_e = new_e
                    best_h = h.copy()
            else:
                # Revert
                if start + length <= N:
                    h[start:start+length] *= -1
                else:
                    h[start:] *= -1
                    h[:end] *= -1
        
        # 2. Attempt swaps between adjacent replicas
        # Swap probability based on temperature difference
        for i in range(n_replicas - 1):
            if time.time() > deadline:
                return best_h, best_e
            
            i1 = i
            i2 = i + 1
            
            T1 = temperatures[i1]
            T2 = temperatures[i2]
            E1 = energies[i1]
            E2 = energies[i2]
            
            # Delta E for swapping states between T1 and T2
            # P(accept) = exp(-(E2/T1 - E1/T1) + (E2/T2 - E1/T2)) 
            #           = exp(-(E2-E1)/T1 + (E2-E1)/T2)
            #           = exp((E2-E1) * (1/T2 - 1/T1))
            
            delta_swap = (E2 - E1) * (1.0/T2 - 1.0/T1)
            
            if delta_swap < 0 or (rng.random() < np.exp(delta_swap)):
                # Swap
                replicas[i1], replicas[i2] = replicas[i2], replicas[i1]
                energies[i1], energies[i2] = energies[i2], energies[i1]
                
                # Update best if the swapped-in replica is better
                if energies[i1] < best_e:
                    best_e = energies[i1]
                    best_h = replicas[i1].copy()
                if energies[i2] < best_e:
                    best_e = energies[i2]
                    best_h = replicas[i2].copy()

    return best_h, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initial guess: Random sequence
    h_init = rng.choice([-1.0, 1.0], size=N)
    
    # Quick local search on the initial guess to get a decent starting point
    # Single flip greedy
    h_best = h_init.copy()
    e_best = _compute_c3_bound(h_best)
    
    # Do a few passes of single-flip local search to warm up
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N):
            h_best[i] *= -1
            e_new = _compute_c3_bound(h_best)
            if e_new < e_best:
                e_best = e_new
                improved = True
            else:
                h_best[i] *= -1
                
    # Now run Parallel Tempering
    h_final, e_final = _parallel_tempering(h_best, deadline, rng)
    
    # Final check: if PT didn't improve, use the warmed-up one
    if e_final > e_best:
        return h_best
    return h_final

# EVOLVE-BLOCK-END
