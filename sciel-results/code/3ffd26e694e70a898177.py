import random
import time

def _score_set_bits(B):
    """
    Count distinct d > 0 such that a, a+d, a+2d all in S.
    B is an integer bitset where bit i is set if (i+1) is in S.
    """
    count = 0
    # Max d such that 2d <= 511 (since indices 0..511)
    for d in range(1, 256):
        if (B & (B >> d) & (B >> (2 * d))) != 0:
            count += 1
    return count

def _dense_seeds(n=32):
    """Return n integers from [1,512] that participate in many 3-APs."""
    counts = []
    for x in range(1, 513):
        c_mid = min(x - 1, 512 - x)
        c_first = (512 - x) // 2
        c_last = (x - 1) // 2
        total = c_mid + c_first + c_last
        counts.append((total, x))
    counts.sort(reverse=True)
    return [x for _, x in counts[:n]]

def construct():
    seeds = _dense_seeds(32)
    
    # Initialize bitset and list
    B = 0
    active_list = []
    for x in seeds:
        B |= (1 << (x - 1))
        active_list.append(x - 1) # store 0-indexed
        
    current_score = _score_set_bits(B)
    best_B = B
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110
    
    random.seed(42)
    
    temp = 5.0
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        # Pick a random element to remove from active_list
        if not active_list:
            break
        idx_to_remove = random.randint(0, len(active_list) - 1)
        val_to_remove = active_list[idx_to_remove]
        
        # Pick a random element to add (not in active_list)
        val_to_add = random.randint(0, 511)
        # Ensure val_to_add is not already in set
        while (B >> val_to_add) & 1:
            val_to_add = random.randint(0, 511)
            
        # Perform move
        B &= ~(1 << val_to_remove)
        B |= (1 << val_to_add)
        
        # Update active_list (swap)
        active_list[idx_to_remove] = val_to_add
        
        new_score = _score_set_bits(B)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_B = B
        else:
            # Revert
            B &= ~(1 << val_to_add)
            B |= (1 << val_to_remove)
            active_list[idx_to_remove] = val_to_remove
            
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    # Convert best_B back to list of 1..512
    result = []
    for i in range(512):
        if (best_B >> i) & 1:
            result.append(i + 1)
    return result
