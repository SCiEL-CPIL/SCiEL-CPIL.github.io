"""
Strategy: Simulated Annealing with restarts.
The greedy local search in the seed program often gets stuck. SA allows
escaping local minima by accepting worse solutions with a probability that
decreases over time. We use the fast FFT score for the main loop to allow
many iterations, and periodically verify with the exact score.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    n_coeffs = 24
    N_fft = 1 << 14  # 16384 points

    # 1. Initial Candidate: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)

    def fast_score(coeffs):
        """Fast sup-norm estimation using FFT."""
        padded = np.zeros(N_fft, dtype=complex)
        padded[:n_coeffs] = coeffs
        vals = np.fft.fft(padded)
        return np.max(np.abs(vals))

    def exact_score(coeffs):
        """
        Refine the peaks found by fast_score using golden-section search.
        Returns the normalized score (max |P(z)| / sqrt(24)).
        """
        # Find top peaks on the FFT grid
        padded = np.zeros(N_fft, dtype=complex)
        padded[:n_coeffs] = coeffs
        vals = np.abs(np.fft.fft(padded))
        
        # Get indices of top 5 peaks
        top_indices = np.argsort(vals)[-5:]
        
        max_val = 0.0
        gr = (math.sqrt(5) - 1) / 2
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / N_fft
            width = 2 * np.pi / N_fft
            
            a = theta_center - width
            b = theta_center + width
            
            def f(theta):
                z = np.exp(1j * theta)
                val = 0.0
                z_pow = 1.0 + 0.0j
                for k in range(24):
                    val += coeffs[k] * z_pow
                    z_pow *= z
                return abs(val)
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            fc = f(c)
            fd = f(d)
            
            for _ in range(50):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
            
            peak_val = max(fc, fd)
            if peak_val > max_val:
                max_val = peak_val
                
        return max_val / math.sqrt(24)

    # Initial evaluation
    current_coeffs = best_coeffs.copy()
    current_fast_val = fast_score(current_coeffs)
    best_val = exact_score(current_coeffs)
    best_coeffs = current_coeffs.copy()

    # Simulated Annealing Parameters
    T_start = 0.5  # Initial temperature (scale of score difference)
    T_min = 0.001
    alpha = 0.9995  # Cooling rate
    N_steps = 200000  # Target number of SA steps (will be limited by time)
    
    # We will run multiple restarts if time permits
    # But first, a long SA chain from the initial good point
    
    T = T_start
    
    # Pre-compute indices for random selection
    indices = np.arange(n_coeffs)
    
    steps_done = 0
    no_improve_count = 0
    max_no_improve = 5000  # Restart if stuck for this many steps
    
    while time.time() < deadline - 5 and steps_done < N_steps:
        # Pick a random index to flip
        i = random.randint(0, n_coeffs - 1)
        
        # Flip
        current_coeffs[i] = -current_coeffs[i]
        new_fast_val = fast_score(current_coeffs)
        delta_fast = new_fast_val - current_fast_val
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_fast < 0 or random.random() < math.exp(-delta_fast / T):
            # Accept
            current_fast_val = new_fast_val
            no_improve_count = 0
        else:
            # Reject: flip back
            current_coeffs[i] = -current_coeffs[i]
            no_improve_count += 1
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        steps_done += 1
        
        # Periodically check exact score
        if steps_done % 1000 == 0:
            exact_v = exact_score(current_coeffs)
            if exact_v < best_val:
                best_val = exact_v
                best_coeffs = current_coeffs.copy()
                no_improve_count = 0
                
        # Check for stagnation
        if no_improve_count > max_no_improve:
            # Restart from best_coeffs with some random perturbations
            current_coeffs = best_coeffs.copy()
            # Perturb 5 random coefficients
            for _ in range(5):
                k = random.randint(0, n_coeffs - 1)
                current_coeffs[k] = -current_coeffs[k]
            current_fast_val = fast_score(current_coeffs)
            T = T_start * 0.5  # Restart with slightly lower temperature
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 0.8) # Tighten stagnation criterion

    # Final verification
    final_val = exact_score(best_coeffs)
    if final_val < best_val:
        best_val = final_val

    return best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
