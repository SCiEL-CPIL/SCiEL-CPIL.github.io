"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_psl_fast(h, n_fft, F, power, ac):
    """
    Compute PSL using pre-allocated buffers.
    """
    np.fft.rfft(h, n=n_fft, out=F)
    np.multiply(F, np.conj(F), out=power)
    np.fft.irfft(power, n=n_fft, out=ac)
    return np.max(np.abs(ac[1:N]))

def _local_search(h_init, time_limit):
    """
    Local search to minimize PSL.
    """
    start_time = time.time()
    current_h = h_init.copy()
    n_fft = 2 * N
    
    # Pre-allocate buffers
    F = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    power = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    ac = np.zeros(n_fft, dtype=np.float64)
    
    # Initial PSL
    current_psl = _compute_psl_fast(current_h, n_fft, F, power, ac)
    
    # Simple random restart logic inside if stuck?
    # No, just keep flipping.
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        i = np.random.randint(N)
        old_val = current_h[i]
        current_h[i] = -old_val
        
        new_psl = _compute_psl_fast(current_h, n_fft, F, power, ac)
        
        if new_psl < current_psl:
            current_psl = new_psl
        else:
            current_h[i] = old_val
            
    return current_h, current_psl

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = None
    best_psl = float('inf')
    
    # Strategy 1: Spectral Initialization (Random Phases, Flat Magnitude)
    # This tends to produce sequences with low sidelobes.
    num_spectral_restarts = 20
    
    for i in range(num_spectral_restarts):
        if time.time() > deadline:
            break
            
        # Generate random phases
        phases = np.random.rand(N) * 2 * np.pi
        spec = np.exp(1j * phases)
        
        # Inverse FFT to get time domain signal
        # We want a sequence of length N.
        # Using n=N in irfft gives N real values.
        h_init = np.real(np.fft.irfft(spec, n=N))
        
        # Threshold to +/- 1
        h_init = np.sign(h_init)
        h_init[h_init == 0] = 1.0
        
        # Allocate time for local search
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        # Distribute remaining time among remaining restarts
        # Use a fraction of the remaining time for this restart
        t_limit = time_left / (num_spectral_restarts - i) * 0.8
        
        h_opt, psl_opt = _local_search(h_init, t_limit)
        
        if psl_opt < best_psl:
            best_psl = psl_opt
            best_h = h_opt
            
    # Strategy 2: Random Initialization
    num_random_restarts = 10
    
    for i in range(num_random_restarts):
        if time.time() > deadline:
            break
            
        h_init = np.random.choice([-1.0, 1.0], size=N)
        
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        t_limit = time_left / (num_random_restarts - i) * 0.8
        
        h_opt, psl_opt = _local_search(h_init, t_limit)
        
        if psl_opt < best_psl:
            best_psl = psl_opt
            best_h = h_opt

    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
