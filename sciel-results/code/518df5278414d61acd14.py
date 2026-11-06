import numpy as np
import time

def construct():
    # Initial sequence: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=int)
    
    # Fast sup norm estimator using FFT
    # We use a large N to approximate the continuous sup norm well.
    def fast_sup_norm(coeffs):
        N = 4096
        padded = np.zeros(N, dtype=complex)
        padded[:24] = coeffs
        fft_vals = np.fft.fft(padded)
        return np.max(np.abs(fft_vals))
    
    best_score = fast_sup_norm(best_coeffs)
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 0.5 # Start temperature
    T_min = 0.01
    cooling = 0.99995
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin for safety
    
    # Optional: Random restarts if stuck? 
    # For now, just run SA.
    
    while time.time() - start_time < timeout:
        # Pick random index to flip
        idx = np.random.randint(24)
        
        # Flip
        current_coeffs[idx] *= -1
        new_score = fast_sup_norm(current_coeffs)
        
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = current_coeffs.copy()
                # If we found a very good solution, we can lower temperature faster?
                # Or just keep going.
        else:
            # Revert
            current_coeffs[idx] *= -1
            
        T = max(T_min, T * cooling)
        
        # Early exit if we hit the theoretical lower bound or known optimum
        # Known optimum sup norm is approx 4.9925
        # If we get below 4.993, we are likely optimal.
        if best_score < 4.993:
            break

    return list(best_coeffs)
