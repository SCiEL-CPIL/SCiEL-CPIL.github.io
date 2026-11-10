import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n = 24
    S = 10000  # Dense sampling for search
    
    # Precompute z powers
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
            
    best_seq = list(gr_sequence(n))
    best_peak = np.inf
    
    # Function to get peak from P_vals
    def get_peak(P_vals):
        return np.max(np.abs(P_vals))
        
    # Local search function
    def local_search(init_seq, time_limit):
        seq = list(init_seq)
        # Compute initial P_vals
        P_vals = np.sum(np.array(seq)[:, None] * z_powers, axis=0)
        current_peak = get_peak(P_vals)
        local_best_peak = current_peak
        local_best_seq = list(seq)
        
        improved = True
        # Limit steps to avoid infinite loops in case of numerical issues
        max_steps = 1000
        steps = 0
        
        while improved and time.time() < time_limit and steps < max_steps:
            improved = False
            best_new_peak = current_peak
            best_idx = -1
            
            # Try all single flips to find the steepest descent
            for k in range(n):
                c_k = seq[k]
                # P_new = P_old - 2 * c_k * z^k
                P_new = P_vals - 2 * c_k * z_powers[k]
                new_peak = get_peak(P_new)
                
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_idx = k
            
            if best_idx != -1 and best_new_peak < current_peak:
                # Accept the best flip
                c_old = seq[best_idx]
                P_vals = P_vals - 2 * c_old * z_powers[best_idx]
                seq[best_idx] = -seq[best_idx]
                current_peak = best_new_peak
                improved = True
                steps += 1
                
                if current_peak < local_best_peak:
                    local_best_peak = current_peak
                    local_best_seq = list(seq)
                    
        return local_best_seq, local_best_peak

    # Multi-start loop
    num_restarts = 0
    while time.time() < deadline - 5:
        # Start from random or GR
        if num_restarts % 5 == 0:
            init_seq = list(gr_sequence(n))
        else:
            init_seq = [random.choice([1, -1]) for _ in range(n)]
            
        # Run local search
        seq, peak = local_search(init_seq, deadline - 2)
        
        if peak < best_peak:
            best_peak = peak
            best_seq = seq
            # Early exit if we hit the theoretical minimum closely
            # Theoretical min peak is approx 6.1517
            if best_peak < 6.152:
                break
                
        num_restarts += 1
        
    return best_seq
