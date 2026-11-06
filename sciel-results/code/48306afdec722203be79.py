import time
import random
import math

def _score_mask(mask):
    """
    Fast score using Python big integers as bitsets.
    mask: int where bit i is 1 if (i+1) is in the set.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    count = 0
    # d can range from 1 to 255 (since 1 + 2*255 = 511 <= 512)
    # For a given d, we need to check if there exists a such that
    # bit a, bit a+d, bit a+2d are all set.
    # This is equivalent to (mask & (mask >> d) & (mask >> 2*d)) != 0
    # We iterate d from 1 to 255.
    # Optimization: Python int bit ops are very fast.
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _mask_to_list(mask):
    """Convert mask to list of integers (1..512)."""
    res = []
    m = mask
    i = 0
    while m:
        if m & 1:
            res.append(i + 1)
        m >>= 1
        i += 1
    return res

def _list_to_mask(S):
    """Convert list of integers (1..512) to mask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 110s.
    deadline = start_time + 110.0
    
    best_score = 0
    best_mask = 0
    
    # Strategy:
    # 1. Quick random initializations to get a decent start.
    # 2. Simulated Annealing with the fast scorer.
    
    # We can afford many SA steps because _score_mask is very fast (~10-20us).
    # Let's run multiple SA chains.
    
    num_chains = 5
    
    for chain_idx in range(num_chains):
        # Check time
        if time.time() > deadline:
            break
        
        # Initialize a random set of 32 distinct integers
        S = random.sample(range(1, 513), 32)
        mask = _list_to_mask(S)
        current_score = _score_mask(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        # We don't know how many steps to take, so we loop until time or T_min
        while time.time() < deadline:
            if T < T_min:
                break
            
            # Pick a random element to remove
            # To pick a random set bit from mask efficiently:
            # We can just pick a random index from 0..511 and check if it's set?
            # No, better to maintain the list S for easy removal.
            # But maintaining list and mask in sync is overhead.
            # Let's just pick a random position in 0..511. If it's not in S, retry?
            # Or better: maintain a list of active indices.
            
            # For simplicity and speed, let's just pick a random x in 1..512.
            # If x is in mask, remove it. Then pick y.
            # Probability of x being in mask is 32/512 = 1/16.
            # This is inefficient.
            
            # Better approach: Maintain the list S.
            # We already have S from init.
            # Let's update S in place.
            
            # Pick random index to remove
            idx_out = random.randint(0, 31)
            x = S[idx_out]
            
            # Pick random y to add
            y = random.randint(1, 512)
            
            if y == x:
                continue
            
            # Check if y is already in S
            # To check if y is in S quickly, we can check the bit in mask.
            if (mask >> (y - 1)) & 1:
                continue
            
            # Apply move
            # Remove x
            mask &= ~(1 << (x - 1))
            # Add y
            mask |= (1 << (y - 1))
            
            # Update S list
            S[idx_out] = y
            
            new_score = _score_mask(mask)
            delta = new_score - current_score
            
            # Acceptance probability
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
            else:
                # Revert
                mask &= ~(1 << (y - 1))
                mask |= (1 << (x - 1))
                S[idx_out] = x
            
            T *= cooling
    
    return _mask_to_list(best_mask)
