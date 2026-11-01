"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _compute_c3_bound_from_acf(acf):
    """Compute C3 bound from precomputed autocorrelation."""
    # Sum of cubes of absolute values of sidelobes
    # Normalized by N^3 as per task description "upper bound on C3"
    # The task says "score is the upper bound on C3" and "evaluator measures the raw value (lower is better)".
    # The seed program used: np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)

def _compute_c3_bound(h):
    """Compute C3 bound from sequence."""
    acf = _compute_acf(h)
    return _compute_c3_bound_from_acf(acf)

def _gs_iteration(h, rng, max_iter=100, target_power=None):
    """
    Gerchberg-Saxton style iteration to minimize sidelobes.
    We want the power spectrum to be flat.
    """
    current_h = h.copy()
    if target_power is None:
        # Target power is the mean power of the current spectrum, or just 1.0
        # For a binary sequence, the average power is 1.
        target_power = np.ones(N)
    
    # We optimize the complex spectrum
    H = np.fft.fft(current_h)
    
    for _ in range(max_iter):
        # 1. Update magnitude to target (flat)
        H_mag = np.abs(H)
        # Avoid division by zero
        H_phase = np.angle(H)
        H_new = target_power * np.exp(1j * H_phase)
        
        # 2. Inverse FFT
        h_cont = np.fft.ifft(H_new).real
        
        # 3. Threshold to binary
        h_bin = np.sign(h_cont)
        h_bin[h_bin == 0] = 1.0 # Handle zeros
        
        # Check for convergence or improvement
        # We don't strictly need to check improvement in GS, but we can stop if it stalls
        if np.array_equal(h_bin, current_h):
            # Small perturbation to escape
            idx = rng.integers(N)
            h_bin[idx] = -h_bin[idx]
            
        current_h = h_bin
        H = np.fft.fft(current_h)
        
    return current_h

def _local_search_sa(h, acf, current_val, deadline, rng, T_init=0.5, cooling=0.999):
    """
    Simulated Annealing local search.
    Uses O(N) update for autocorrelation changes.
    """
    best_h = h.copy()
    best_val = current_val
    best_acf = acf.copy()
    
    T = T_init
    
    # Precompute indices for shuffling
    indices = np.arange(N)
    
    while time.time() < deadline and T > 1e-6:
        # Shuffle indices to avoid bias
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                return best_h, best_val
            
            # Calculate change in acf if we flip h[i]
            d_acf = np.zeros(N)
            
            # Right part: k from 1 to N-1-i
            if i < N - 1:
                right_len = N - 1 - i
                d_acf[1:right_len+1] += -2 * h[i] * h[i+1:]
                
            # Left part: k from 1 to i
            if i > 0:
                left_len = i
                d_acf[1:left_len+1] += -2 * h[i] * h[:i][::-1]
            
            new_acf = acf + d_acf
            new_val = _compute_c3_bound_from_acf(new_acf)
            
            delta = new_val - current_val
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
                # Accept
                h[i] = -h[i]
                acf = new_acf
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                    best_acf = acf.copy()
        
        T *= cooling

    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = np.inf
    
    # Strategy 1: Spectral Initialization (GS) + SA
    # Run a few GS iterations from different random starts
    for restart in range(10):
        if time.time() > deadline - 15:
            break
            
        # Random start
        h = rng.choice([-1.0, 1.0], size=N)
        
        # Run GS to get a low-sidelobe structure
        h_gs = _gs_iteration(h, rng, max_iter=50)
        
        # Evaluate
        acf = _compute_acf(h_gs)
        val = _compute_c3_bound_from_acf(acf)
        
        if val < best_val:
            best_val = val
            best_h = h_gs.copy()
            
        # Polish with SA
        if time.time() < deadline - 10:
            res_h, res_val = _local_search_sa(h_gs, acf, val, deadline, rng)
            if res_val < best_val:
                best_val = res_val
                best_h = res_h

    # Strategy 2: Pure Random + SA (backup)
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            h = rng.choice([-1.0, 1.0], size=N)
            acf = _compute_acf(h)
            val = _compute_c3_bound_from_acf(acf)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            if time.time() < deadline - 5:
                res_h, res_val = _local_search_sa(h, acf, val, deadline, rng)
                if res_val < best_val:
                    best_val = res_val
                    best_h = res_h

    # Final polish on the best result
    if best_h is not None and time.time() < deadline - 2:
        acf = _compute_acf(best_h)
        val = _compute_c3_bound_from_acf(acf)
        res_h, res_val = _local_search_sa(best_h, acf, val, deadline, rng, T_init=0.1, cooling=0.995)
        if res_val < best_val:
            best_h = res_h

    return best_h

# EVOLVE-BLOCK-END
