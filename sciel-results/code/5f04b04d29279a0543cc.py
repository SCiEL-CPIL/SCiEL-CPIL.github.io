import random
import time

def _score_mask(mask):
    """Count distinct d > 0 such that ∃a: a, a+d, a+2d all in S represented by mask."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
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
    current_elems = seeds[:]
    current_mask = 0
    for x in current_elems:
        current_mask |= (1 << (x - 1))
        
    current_score = _score_mask(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110
    
    temp = 10.0
    cooling = 0.9999
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, len(current_elems) - 1)
        to_remove = current_elems[idx]
        
        # Find a random element to add that is not in the current set
        while True:
            to_add = random.randint(1, 512)
            if not (current_mask & (1 << (to_add - 1))):
                break
        
        new_mask = current_mask
        new_mask &= ~(1 << (to_remove - 1))
        new_mask |= (1 << (to_add - 1))
        
        new_score = _score_mask(new_mask)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            current_elems[idx] = to_add
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_mask = current_mask
                best_score = current_score
                
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    # Convert best_mask to list of integers
    result = []
    for i in range(512):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return result
