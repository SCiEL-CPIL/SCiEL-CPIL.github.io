import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    N = 512
    K = 32
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set (bit x-1 for x)."""
        count = 0
        # d ranges 1..255 (since max element 512, 2d <= 511)
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def set_to_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def bitset_to_set(b):
        return {i + 1 for i in range(N) if (b >> i) & 1}
    
    def random_set():
        return set(random.sample(range(1, N + 1), K))
    
    # Simulated Annealing
    current_set = set(range(1, K + 1))
    current_b = set_to_bitset(current_set)
    current_score = score_bitset(current_b)
    
    best_set = set(current_set)
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Pick a random element to replace
        elem_to_replace = random.choice(list(current_set))
        
        # Pick a random new value
        new_val = random.randint(1, N)
        if new_val in current_set:
            continue
        
        # Try replacement
        old_bit = 1 << (elem_to_replace - 1)
        new_bit = 1 << (new_val - 1)
        new_b = (current_b & ~old_bit) | new_bit
        
        new_score = score_bitset(new_b)
        
        # Acceptance criterion
        if new_score >= current_score:
            accept = True
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = min(1.0, 1.0 / (1.0 + abs(delta)))
                accept = (random.random() < prob)
            else:
                accept = True
        
        if accept:
            current_set = set(current_set)
            current_set.remove(elem_to_replace)
            current_set.add(new_val)
            current_b = new_b
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        
        T *= cooling
    
    return sorted(best_set)

def get_score(S):
    S_set = set(S)
    S_sorted = sorted(S)
    d_set = set()
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i+1, n):
            b = S_sorted[j]
            d = b - a
            c = b + d
            if c in S_set:
                d_set.add(d)
    return len(d_set)
