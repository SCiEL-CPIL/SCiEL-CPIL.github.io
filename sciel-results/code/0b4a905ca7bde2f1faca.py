"""Program for reciprocal weight of three-progression-free sets on [1,256]."""
import time
import random
import math

# EVOLVE-BLOCK-START
N = 256
RECIPS = [0.0] + [1.0 / i for i in range(1, N + 1)]

def _is_3ap_free_add(mask, x):
    """Check if adding x to set represented by mask keeps it 3-AP free.
    x is 1-indexed. mask is int where bit i-1 is set if i is in set.
    """
    if (mask >> (x - 1)) & 1:
        return True
    
    m = mask
    while m:
        lsb = m & -m
        a = lsb.bit_length() # 1-indexed
        m ^= lsb
        
        # x is middle: c = 2x - a
        c = 2 * x - a
        if c > 0 and c <= N and c != a and c != x:
            if (mask >> (c - 1)) & 1:
                return False
        
        # x is end: b = (x + a) / 2
        if (x + a) % 2 == 0:
            b = (x + a) // 2
            if b > 0 and b <= N and b != a and b != x:
                if (mask >> (b - 1)) & 1:
                    return False
    return True

def _get_sum(mask):
    s = 0.0
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        s += RECIPS[i]
        m ^= lsb
    return s

def _greedy_ascending():
    mask = 0
    for x in range(1, N + 1):
        if _is_3ap_free_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Start with greedy ascending
    best_mask = _greedy_ascending()
    best_score = _get_sum(best_mask)
    
    # Simulated Annealing
    current_mask = best_mask
    current_score = best_score
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if T < 1e-5:
            break
            
        # Pick a random move: flip a bit
        x = random.randint(1, N)
        
        is_in_set = (current_mask >> (x - 1)) & 1
        
        if is_in_set:
            # Remove x
            new_mask = current_mask ^ (1 << (x - 1))
            new_score = current_score - RECIPS[x]
            delta = new_score - current_score
            # Always valid to remove
            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Add x
            if _is_3ap_free_add(current_mask, x):
                new_mask = current_mask | (1 << (x - 1))
                new_score = current_score + RECIPS[x]
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            # else: invalid move, reject
            
        T *= cooling_rate
        
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        result.append(i)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
