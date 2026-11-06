import numpy as np
import random
import time
import math

def construct():
    start_time = time.time()
    best_score = 23 # Max possible is 2*12-1 = 23
    best_c = np.zeros(41, dtype=int)
    # Initialize best_c with a random 12-term polynomial
    for _ in range(100):
        idxs = random.sample(range(41), 12)
        c = np.zeros(41, dtype=int)
        for i in idxs:
            c[i] = random.randint(1, 5)
        best_c = c
        break

    def get_score(c):
        # c is numpy array of length 41
        # Square is convolution of c with c
        # Length of square is 41+41-1 = 81
        conv = np.convolve(c, c)
        return np.count_nonzero(conv)

    current_c = best_c.copy()
    current_score = get_score(current_c)
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        if time.time() - start_time > 55:
            break
            
        steps += 1
        
        # Generate a neighbor
        # 90% of the time: change a value
        # 10% of the time: move a term (change position)
        c = current_c.copy()
        
        if random.random() < 0.9:
            # Change value of a nonzero term
            nonzero_idxs = np.nonzero(c)[0]
            if len(nonzero_idxs) == 0:
                continue
            idx = random.choice(nonzero_idxs)
            old_val = c[idx]
            # Try small changes or random jumps
            if random.random() < 0.5:
                new_val = old_val + random.choice([-2, -1, 1, 2])
            else:
                new_val = random.randint(-10, 10)
            
            if new_val == 0:
                # If we set to 0, we lose a term. 
                # To keep exactly 12 terms, we must add another one.
                # Or we can allow temporary violations? No, contract says exactly 12.
                # So if we set to 0, we must pick a zero index and set it to non-zero.
                zero_idxs = np.nonzero(c == 0)[0]
                if len(zero_idxs) == 0:
                    continue
                new_idx = random.choice(zero_idxs)
                c[idx] = 0
                c[new_idx] = new_val if new_val != 0 else 1
            elif new_val == old_val:
                continue
            else:
                c[idx] = new_val
        else:
            # Move a term: pick a nonzero index, clear it, pick a zero index, set it.
            nonzero_idxs = np.nonzero(c)[0]
            if len(nonzero_idxs) < 2:
                continue
            idx_out = random.choice(nonzero_idxs)
            val_out = c[idx_out]
            zero_idxs = np.nonzero(c == 0)[0]
            if len(zero_idxs) == 0:
                continue
            idx_in = random.choice(zero_idxs)
            
            c[idx_out] = 0
            c[idx_in] = val_out
            
        # Ensure exactly 12 nonzero terms (safety check, should be maintained by logic above)
        n_nonzero = np.count_nonzero(c)
        if n_nonzero != 12:
            # Repair
            if n_nonzero < 12:
                while n_nonzero < 12:
                    zero_idxs = np.nonzero(c == 0)[0]
                    if len(zero_idxs) == 0: break
                    idx = random.choice(zero_idxs)
                    c[idx] = 1
                    n_nonzero += 1
            elif n_nonzero > 12:
                while n_nonzero > 12:
                    nonzero_idxs = np.nonzero(c)[0]
                    idx = random.choice(nonzero_idxs)
                    c[idx] = 0
                    n_nonzero -= 1
            
        # Ensure no coefficient exceeds bounds [-64, 64]
        c = np.clip(c, -64, 64)
        
        # If clipping created zeros or changed values, re-check count
        n_nonzero = np.count_nonzero(c)
        if n_nonzero != 12:
             # If we clipped a value to 0, we need to fix it.
             # If we clipped a value to 64, it's still nonzero.
             # The main risk is clipping -65 to -64 (still nonzero) or 0 to 0.
             # Since we only change to [-10, 10] or small steps, clipping to 0 is unlikely unless we set to 0 explicitly.
             # But let's be safe.
             if n_nonzero < 12:
                 zero_idxs = np.nonzero(c == 0)[0]
                 for i in range(12 - n_nonzero):
                     if len(zero_idxs) == 0: break
                     idx = random.choice(zero_idxs)
                     c[idx] = 1
                     zero_idxs = np.nonzero(c == 0)[0]
             elif n_nonzero > 12:
                 nonzero_idxs = np.nonzero(c)[0]
                 for i in range(n_nonzero - 12):
                     if len(nonzero_idxs) == 0: break
                     idx = random.choice(nonzero_idxs)
                     c[idx] = 0
                     nonzero_idxs = np.nonzero(c)[0]

        new_score = get_score(c)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            current_c = c
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_c = c.copy()
                if best_score <= 3: # Theoretical min?
                    break
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                current_c = c
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final cleanup: ensure best_c is valid
    # best_c is a numpy array of length 41
    # Convert to list of ints
    res = [int(x) for x in best_c]
    
    # Verify constraints on the returned list
    # 1. Length <= 41 (Yes, 41)
    # 2. Nonzero final coefficient? 
    #    The contract says "nonzero final coefficient". 
    #    This usually means the highest degree term is non-zero.
    #    My array is length 41 (indices 0..40).
    #    If best_c[40] is 0, we need to fix it.
    if res[-1] == 0:
        # Find a non-zero term and move it to the end? Or just set end to 1 and zero out another?
        # To maintain 12 terms, we should move a term.
        # But this is a heuristic repair.
        nonzero_idxs = [i for i, x in enumerate(res) if x != 0]
        if len(nonzero_idxs) > 0:
            # Pick the last nonzero index
            last_idx = nonzero_idxs[-1]
            if last_idx != 40:
                val = res[last_idx]
                res[last_idx] = 0
                res[40] = val
            else:
                # If last_idx is 40, it's not 0. Contradiction.
                pass
    
    # Re-check 12 terms
    if len([x for x in res if x != 0]) != 12:
        # Fallback: construct a valid one
        res = [0] * 41
        for i in range(12):
            res[i] = 1
        res[-1] = 1 # Ensure last is non-zero? If 12 terms are 0..11, last (40) is 0.
        # So we need to place one term at 40.
        res = [0] * 41
        res[40] = 1
        for i in range(11):
            res[i] = 1
        # This has 12 terms.

    return res
