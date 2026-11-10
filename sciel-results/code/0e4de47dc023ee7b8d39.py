import time
import random

def construct():
    deadline = time.time() + 50.0
    
    def is_valid(S):
        n = len(S)
        if n < 3:
            return True
        
        # Sort to potentially break early? Not strictly needed for bitset sum.
        # But checking multiples k*a requires k up to n-1.
        
        for i, a in enumerate(S):
            others = [x for j, x in enumerate(S) if j != i]
            
            # Bitset for subset sums of 'others'
            # Initialize with 0 sum (empty subset)
            bitset = 1 
            
            for x in others:
                bitset |= bitset << x
                
            # Check if any k*a is a subset sum
            # k ranges from 2 to len(others)
            max_k = len(others)
            for k in range(2, max_k + 1):
                target = k * a
                if (bitset >> target) & 1:
                    return False
                    
        return True

    best_set = []
    
    # Strategy 1: Deterministic Greedy (Seed-like but full range)
    # This might find a good dense set
    current = []
    for x in range(1, 513):
        candidate = current + [x]
        if is_valid(candidate):
            current = candidate
    if len(current) > len(best_set):
        best_set = current
        
    # Strategy 2: Randomized Greedy
    while time.time() < deadline:
        current = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            candidate = current + [x]
            if is_valid(candidate):
                current = candidate
        
        if len(current) > len(best_set):
            best_set = current
            # Optional: break if we hit a theoretical upper bound? 
            # 512 is max. 48 is current best. 
            # If we find 50, keep searching for 51?

    return best_set
