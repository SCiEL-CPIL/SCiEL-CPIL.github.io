import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    N = 512
    K = 32
    
    def get_score(m):
        score = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
        return score
    
    # Initialize with a good starting point: use a structured set
    # Try a few different initialization strategies
    best_mask = 0
    best_score = 0
    
    # Strategy 1: Random initialization
    for attempt in range(5):
        indices = random.sample(range(N), K)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        s = get_score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
    
    # Strategy 2: Structured initialization - use arithmetic progressions
    # Try sets that are unions of small APs
    for base in range(0, 50, 5):
        mask = 0
        # Create a set with elements spread out
        used = set()
        for i in range(K):
            # Try to place elements to cover many differences
            val = (base + i * 16) % N
            while val in used:
                val = (val + 1) % N
            used.add(val)
            mask |= (1 << val)
        s = get_score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
    
    current_mask = best_mask
    current_score = best_score
    
    # Maintain lists of set and unset bits
    set_bits = [i for i in range(N) if (current_mask >> i) & 1]
    unset_bits = [i for i in range(N) if not ((current_mask >> i) & 1)]
    
    # Local search with simulated annealing-like acceptance
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random i from set_bits, j from unset_bits
        idx_i = random.randint(0, len(set_bits) - 1)
        idx_j = random.randint(0, len(unset_bits) - 1)
        
        i = set_bits[idx_i]
        j = unset_bits[idx_j]
        
        # Swap
        new_mask = current_mask ^ (1 << i) ^ (1 << j)
        
        new_score = get_score(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_mask = new_mask
            current_score = new_score
            set_bits[idx_i] = j
            unset_bits[idx_j] = i
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        else:
            # Accept with probability exp(delta/temp) if delta < 0
            if temp > 0.01:
                import math
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_mask = new_mask
                    current_score = new_score
                    set_bits[idx_i] = j
                    unset_bits[idx_j] = i
        
        temp *= cooling
    
    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
