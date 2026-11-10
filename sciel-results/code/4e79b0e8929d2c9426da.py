import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 24
    S = 10000  # Increase samples for better precision in peak location
    
    # Precompute z powers for fast evaluation
    angles = 2 * np.pi * np.arange(S) / S
    z_powers = np.zeros((n, S), dtype=complex)
    for k in range(n):
        z_powers[k] = np.exp(1j * k * angles)
    
    # Initial sequence: Golay-Rudin-Shapiro
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    # Helper to compute peak and peak index
    def get_peak_info(seq):
        P_vals = np.sum(np.array(seq)[:, None] * z_powers, axis=0)
        abs_vals = np.abs(P_vals)
        peak_idx = np.argmax(abs_vals)
        peak_val = abs_vals[peak_idx]
        return P_vals, peak_idx, peak_val

    # Greedy Local Search
    def local_search(seq):
        seq = list(seq)
        P_vals, peak_idx, current_peak = get_peak_info(seq)
        
        max_steps = 5000
        
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Find the best flip to reduce the peak at the current peak_idx
            # P(peak) = sum c_k z_k(peak)
            # New P(peak) if flip k: P(peak) - 2 c_k z_k(peak)
            # We want to minimize |New P(peak)|
            
            z_peak = z_powers[:, peak_idx] # Vector of z^k at peak angle
            P_peak = P_vals[peak_idx]
            
            # Calculate potential new magnitudes for all k
            # delta_k = -2 * seq[k] * z_peak[k]
            # new_P_k = P_peak + delta_k
            # cost_k = |new_P_k|
            
            deltas = -2 * np.array(seq) * z_peak
            new_Ps = P_peak + deltas
            costs = np.abs(new_Ps)
            
            best_k = np.argmin(costs)
            best_cost = costs[best_k]
            
            if best_cost < current_peak:
                # Accept the flip if it reduces the peak at the current peak location
                # We must also check if it increases the global peak
                # But calculating global peak is O(S).
                # Let's just accept if it reduces the local peak, and then check global.
                
                old_c = seq[best_k]
                seq[best_k] = -old_c
                
                # Update P_vals
                change_vec = -2 * old_c * z_powers[best_k]
                P_vals += change_vec
                
                # Update peak info
                abs_vals = np.abs(P_vals)
                new_peak_idx = np.argmax(abs_vals)
                new_peak_val = abs_vals[new_peak_idx]
                
                # Only accept if the new global peak is not larger than the old one
                # This ensures monotonic decrease of the objective
                if new_peak_val <= current_peak:
                    current_peak = new_peak_val
                    peak_idx = new_peak_idx
                else:
                    # Revert
                    seq[best_k] = old_c
                    P_vals -= change_vec
                    # If we can't reduce the peak without increasing another, we are stuck
                    break
            else:
                # No flip reduces the peak at the current location
                break
        
        return seq, current_peak

    # Main loop
    best_peak = float('inf')
    best_seq = None
    
    # Start with GR
    current_seq = list(gr_sequence(n))
    
    # Run local search
    while time.time() < deadline - 10:
        improved_seq, local_peak = local_search(current_seq)
        
        if local_peak < best_peak:
            best_peak = local_peak
            best_seq = list(improved_seq)
            current_seq = list(improved_seq)
        else:
            # Perturb to escape local minimum
            # Flip 2-3 random bits
            num_flips = random.randint(2, 3)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                current_seq[idx] = -current_seq[idx]
                
            # If we've been perturbing without success for a while, restart from GR?
            # Simple counter
            # For now, just keep perturbing the current stuck solution
            
    return best_seq
