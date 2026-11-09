"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialize with Thue-Morse
    def thue_morse(n):
        t = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            t[i] = t[i >> 1] ^ (i & 1)
        return 2 * t - 1
        
    s = thue_morse(N).astype(np.float64)
    
    # Helper to compute R
    def calc_R(s):
        f = np.fft.fft(s, 2*N)
        r = np.fft.ifft(f * np.conj(f))
        return r[:N].real

    R = calc_R(s)
    current_cost = np.max(np.abs(R[1:]))
    best_cost = current_cost
    best_s = s.copy()
    best_R = R.copy()
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.5
    alpha = 0.999
    
    # Pre-allocate delta
    delta = np.zeros(N)
    
    while time.time() < deadline:
        if T < T_min:
            # Reheat and restart from best with perturbation
            T = 2.0
            s = best_s.copy()
            R = best_R.copy()
            current_cost = best_cost
            
            # Perturb
            for _ in range(5):
                j = random.randint(0, N-1)
                s[j] *= -1
            
            # Recalculate R for perturbed state
            R = calc_R(s)
            current_cost = np.max(np.abs(R[1:]))
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_s = s.copy()
                best_R = R.copy()
            continue

        j = random.randint(0, N-1)
        
        # Calculate delta for flip at j
        delta[:] = 0.0
        val = s[j]
        
        # Update delta
        if N - j > 0:
            delta[0:N-j] -= 2 * val * s[j:N]
        if j >= 0:
            delta[0:j+1] -= 2 * val * s[j::-1]
            
        R_new = R + delta
        new_cost = np.max(np.abs(R_new[1:]))
        delta_cost = new_cost - current_cost
        
        # Acceptance criterion
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            s[j] *= -1
            R = R_new
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_s = s.copy()
                best_R = R.copy()
        else:
            pass # Revert
            
        T *= alpha
        
    return best_s.astype(np.int8)
# EVOLVE-BLOCK-END
