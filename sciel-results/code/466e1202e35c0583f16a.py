import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    n = 24
    sqrt_n = np.sqrt(n)
    
    # Precompute points on the unit circle
    # Use a high number of samples for accurate sup norm estimation
    S = 4096
    angles = 2 * np.pi * np.arange(S) / S
    zs = np.exp(1j * angles)
    
    def eval_poly(coeffs):
        """Evaluate polynomial at all zs."""
        # coeffs is a numpy array of shape (n,)
        # Result is shape (S,)
        # Horner's method vectorized
        result = np.zeros(S, dtype=complex)
        for c in coeffs:
            result = result * zs + c
        return result
    
    def sup_norm(coeffs):
        """Compute sup norm of P(z) over |z|=1."""
        vals = eval_poly(coeffs)
        mag = np.abs(vals)
        # Refine the top few peaks to get a more accurate estimate
        # Find indices of top 5 peaks
        if S > 10:
            top_indices = np.argsort(mag)[-5:][::-1]
            best_val = float(np.max(mag))
            window = 2 * np.pi / S
            for idx in top_indices:
                center = angles[idx]
                # Golden section search in [center - window, center + window]
                # We can use scipy or implement a simple local search
                # For speed, let's just check a few points around the peak
                # Or use minimize_scalar from scipy
                # But minimize_scalar is slow if called too often.
                # Let's use a small grid around the peak
                local_angles = np.linspace(center - window, center + window, 10)
                local_zs = np.exp(1j * local_angles)
                local_vals = np.zeros(len(local_angles), dtype=complex)
                for c in coeffs:
                    local_vals = local_vals * local_zs + c
                local_mag = np.abs(local_vals)
                peak_val = float(np.max(local_mag))
                if peak_val > best_val:
                    best_val = peak_val
            return best_val
        return float(np.max(mag))
    
    def score(coeffs):
        return sup_norm(coeffs) / sqrt_n

    def score_fast(coeffs):
        """Faster score for SA steps: no refinement, maybe fewer samples?"""
        # Use all S samples, just max
        vals = eval_poly(coeffs)
        return float(np.max(np.abs(vals))) / sqrt_n

    # Initialize with Golay-Rudin-Shapiro sequence
    def grs_sequence(length=24):
        p, q = [1], [1]
        while len(p) < length:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:length])
        
    best_coeffs = grs_sequence(n)
    best_score = score(best_coeffs)
    
    # Simulated Annealing
    # Parameters
    T_start = 0.5  # Initial temperature (in units of score difference)
    T_end = 0.001
    alpha = 0.999  # Cooling rate
    max_steps = 500000  # Adjust based on time
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    T = T_start
    
    # Precompute powers of z for faster eval? 
    # eval_poly is already vectorized.
    
    steps_done = 0
    while time.time() < deadline - 10 and steps_done < max_steps:
        # Generate neighbor
        # Flip 1 or 2 bits
        if random.random() < 0.5:
            # Flip 1 bit
            i = random.randint(0, n - 1)
            new_coeffs = current_coeffs.copy()
            new_coeffs[i] *= -1
        else:
            # Flip 2 bits
            i, j = random.sample(range(n), 2)
            new_coeffs = current_coeffs.copy()
            new_coeffs[i] *= -1
            new_coeffs[j] *= -1
            
        new_score = score_fast(new_coeffs)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_coeffs = new_coeffs
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_coeffs = new_coeffs.copy()
                
        T *= alpha
        steps_done += 1
        
    # Final polish with accurate score
    final_score = score(best_coeffs)
    
    return best_coeffs.tolist()
