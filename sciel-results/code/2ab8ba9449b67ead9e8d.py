"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_psl(h):
    """
    Compute the Peak Sidelobe Level (PSL) of the aperiodic autocorrelation
    of the binary sequence h.
    PSL = max_{k != 0} |R[k]| / R[0]
    """
    # Pad to length 2N to compute aperiodic autocorrelation via FFT
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    H = np.fft.fft(h_pad)
    R = np.fft.ifft(H * np.conj(H)).real
    
    # R[0] is the peak (energy)
    peak = R[0]
    if peak < 1e-12:
        return 0.0
        
    # Sidelobes are R[1] to R[N-1]
    # Note: R[N] to R[2N-1] are the negative lags, symmetric for real signals
    sidelobes = np.abs(R[1:N])
    max_side = np.max(sidelobes)
    
    return max_side / peak

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 110.0
    
    best_psl = float('inf')
    best_h = None
    
    # We will perform multiple random restarts with local search
    # Each restart initializes a random binary sequence and refines it
    
    while time.time() < deadline:
        # Initialize with random +/- 1
        h = np.random.choice([-1, 1], N)
        current_psl = _compute_psl(h)
        
        # Local Search: First-improvement hill climbing
        # We iterate up to a max number of flips or until time is up
        # We can also add a simple Simulated Annealing component if stuck,
        # but first-improvement with random restarts is often sufficient for N=400.
        
        # To be more effective, we can try to flip bits that contribute most to the peak sidelobe.
        # However, random flipping with acceptance if it improves is simple and robust.
        
        # Adaptive iteration count: run for a fixed time slice per restart?
        # Or fixed iterations?
        # Let's do fixed iterations per restart, say 2000.
        # If N=400, 2000 flips is 5 flips per bit on average.
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(N)
            h[idx] *= -1
            
            new_psl = _compute_psl(h)
            
            if new_psl < current_psl:
                current_psl = new_psl
                # If we improve, we stay. If not, we revert.
            else:
                h[idx] *= -1
                
            # Early termination if PSL is extremely low
            if current_psl < 0.02:
                break
        
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h.copy()
            
            # If we find a very good solution, we might want to spend more time polishing it
            # But for now, we just record it and continue searching for better.
            
    return best_h
# EVOLVE-BLOCK-END
