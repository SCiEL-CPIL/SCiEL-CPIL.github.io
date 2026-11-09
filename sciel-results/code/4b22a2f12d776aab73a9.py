"""Rewritten program for low-discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute the list of (start, step) pairs and their indices
    # This avoids re-calculating indices every time
    ap_data = []
    for step in range(1, n + 1):
        for start in range(n):
            indices = list(range(start, n, step))
            if indices:
                ap_data.append(indices)
    
    # Function to compute the maximum absolute signed sum on any AP
    # Returns max_disc, index of the AP that achieves it
    def max_discrepancy(signs):
        max_disc = 0
        best_ap_idx = 0
        for ap_idx, indices in enumerate(ap_data):
            s = 0
            for i in indices:
                s += signs[i]
            abs_s = abs(s)
            if abs_s > max_disc:
                max_disc = abs_s
                best_ap_idx = ap_idx
        return max_disc, best_ap_idx

    rng = random.Random(42)
    
    best_signs = None
    best_disc = float('inf')
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Initialization: Use a structured sequence to start low
        if restart == 0:
            # Thue-Morse sequence: t(n) = parity of number of 1s in binary representation of n
            # This has low discrepancy for many sequences
            signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
        elif restart == 1:
            # Balanced blocks of 4: 1,1,-1,-1
            signs = []
            for i in range(n):
                block_idx = i % 4
                if block_idx < 2:
                    signs.append(1)
                else:
                    signs.append(-1)
        elif restart == 2:
            # Balanced blocks of 8: 1,1,1,-1,-1,-1,1,-1 (approx)
            # Or just random
            signs = [rng.choice([-1, 1]) for _ in range(n)]
        else:
            signs = [rng.choice([-1, 1]) for _ in range(n)]
        
        current_disc, cur_ap_idx = max_discrepancy(signs)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
        
        # Simulated Annealing
        temperature = 2.0
        cooling_rate = 0.9995
        min_temp = 0.1
        
        stall_counter = 0
        max_stall = 2000 # Reduced because moves are more expensive/directed
        
        while time.time() - start_time < time_limit - 2.0:
            # Determine move strategy
            # 80% chance: Targeted move on the worst AP
            # 20% chance: Random move to explore
            
            if rng.random() < 0.8:
                worst_indices = ap_data[cur_ap_idx]
                # Find the current sum of the worst AP
                current_sum = sum(signs[i] for i in worst_indices)
                
                # We want to reduce the absolute value of current_sum
                # If current_sum > 0, we should flip +1 to -1
                # If current_sum < 0, we should flip -1 to +1
                # If current_sum == 0, pick random from AP
                
                candidates = []
                for i in worst_indices:
                    if current_sum > 0 and signs[i] == 1:
                        candidates.append(i)
                    elif current_sum < 0 and signs[i] == -1:
                        candidates.append(i)
                    elif current_sum == 0:
                        candidates.append(i)
                
                if candidates:
                    idx = rng.choice(candidates)
                else:
                    # No candidates (e.g., sum is positive but all are -1? Impossible unless sum is negative)
                    # This case shouldn't happen if logic is correct, but fallback to random
                    idx = rng.randint(0, n - 1)
            else:
                idx = rng.randint(0, n - 1)
            
            # Perform the flip
            signs[idx] = -signs[idx]
            
            # Evaluate new state
            new_disc, new_ap_idx = max_discrepancy(signs)
            
            delta = new_disc - current_disc
            
            if delta < 0:
                # Accept
                current_disc = new_disc
                cur_ap_idx = new_ap_idx
                stall_counter = 0
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                # Accept with probability exp(-delta / temperature)
                # delta is integer, usually small (1 or 2)
                if delta == 0:
                    # Accept neutral moves to wander
                    current_disc = new_disc
                    cur_ap_idx = new_ap_idx
                else:
                    prob = math.exp(-delta / temperature)
                    if rng.random() < prob:
                        current_disc = new_disc
                        cur_ap_idx = new_ap_idx
                    else:
                        # Reject: flip back
                        signs[idx] = -signs[idx]
            
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
            
            stall_counter += 1
            if stall_counter >= max_stall:
                break
    
    return best_signs

# EVOLVE-BLOCK-END
