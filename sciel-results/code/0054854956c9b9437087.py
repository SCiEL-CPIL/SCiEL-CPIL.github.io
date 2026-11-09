import time
import random
import math
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55  # Safety margin for the 60s/90s limit
    
    # Thue-Morse initialization
    def thue_morse(n):
        seq = []
        for i in range(n):
            c = bin(i).count('1')
            if c % 2 == 0:
                seq.append(-1)
            else:
                seq.append(1)
        return seq

    seq = thue_morse(n)
    arr = np.array(seq, dtype=np.int8)
    
    def get_cost(arr):
        """
        Compute max absolute signed sum over all arithmetic progressions.
        Returns the max discrepancy and the (start, step) of the worst AP.
        """
        max_disc = 0
        worst_start = 0
        worst_step = 1
        nn = n
        for d in range(1, nn + 1):
            for r in range(d):
                idx = np.arange(r, nn, d)
                if len(idx) == 0:
                    continue
                cumsum = np.cumsum(arr[idx])
                cs = np.concatenate(([0], cumsum))
                disc = cs.max() - cs.min()
                if disc > max_disc:
                    max_disc = disc
                    worst_start = r
                    worst_step = d
        return max_disc, worst_start, worst_step

    best_seq = seq[:]
    best_cost, best_start, best_step = get_cost(arr)
    
    rng = random.Random(42)
    
    # Constraint-directed search parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.05
    
    # We will run multiple restarts if time permits
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        T = 5.0
        current_seq = best_seq[:]
        current_arr = np.array(current_seq, dtype=np.int8)
        current_cost, current_start, current_step = best_cost, best_start, best_step
        
        no_improve = 0
        patience = 2000
        
        while time.time() < deadline:
            # Identify the worst AP indices
            # r, r+d, r+2d, ...
            idxs = np.arange(current_start, n, current_step)
            if len(idxs) == 0:
                # Fallback to random
                flip_idx = rng.randrange(n)
            else:
                # Pick an index to flip from within the worst AP
                # Weighted towards the middle or just uniform within the AP
                flip_pos = rng.randrange(len(idxs))
                flip_idx = idxs[flip_pos]
                
            current_seq[flip_idx] *= -1
            current_arr[flip_idx] = current_seq[flip_idx]
            
            new_cost, new_start, new_step = get_cost(current_arr)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                current_start = new_start
                current_step = new_step
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = current_seq[:]
                    best_start = new_start
                    best_step = new_step
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if delta == 0:
                    current_cost = new_cost
                    current_start = new_start
                    current_step = new_step
                    no_improve += 1
                elif rng.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    current_start = new_start
                    current_step = new_step
                    no_improve += 1
                else:
                    current_seq[flip_idx] *= -1
                    current_arr[flip_idx] = current_seq[flip_idx]
                    no_improve += 1
            
            T *= cooling
            if T < min_T:
                break
            
            if no_improve > patience:
                break
        
        if time.time() >= deadline:
            break
        
        restarts += 1
        if restarts > max_restarts:
            break
    
    # Final greedy polish pass: try flipping each position, keep if it improves
    polish_arr = np.array(best_seq, dtype=np.int8)
    polish_cost, _, _ = get_cost(polish_arr)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            polish_arr[i] *= -1
            new_c, _, _ = get_cost(polish_arr)
            if new_c < polish_cost:
                polish_cost = new_c
                best_seq = polish_arr.tolist()
                improved = True
            else:
                polish_arr[i] *= -1
    
    best_cost = polish_cost
    return best_seq
