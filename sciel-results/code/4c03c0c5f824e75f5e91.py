# EVOLVE-BLOCK-START
import time
import random
from math import log

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    time_limit = 115
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def try_set(U):
        nonlocal best_score, best_set
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_set = sorted(set(U))
            
    # Strategy 1: Structured candidates
    # Powers of 2
    for k in range(3, 20):
        U = [2**i for i in range(k)]
        try_set(U)
        
    # 2^i - 1
    for k in range(3, 20):
        U = [(2**i) - 1 for i in range(1, k)]
        U.append(0)
        try_set(U)
        
    # Powers of 3
    for k in range(3, 15):
        U = [3**i for i in range(k)]
        try_set(U)
        
    # Mixed: 0, 1, then geometric
    for base in range(2, 10):
        for k in range(4, 15):
            U = [0, 1]
            val = 1
            for i in range(k-2):
                val *= base
                U.append(val)
            try_set(U)
            
    # Local Search / Hill Climbing on the best set found so far
    if time.time() - start_time < time_limit - 10:
        # Start local search from the best set
        current_set = list(best_set)
        current_score = best_score
        
        # Ensure 0 is at index 0
        current_set.sort()
        
        # We will perturb elements (excluding 0)
        # To avoid duplicates, we work on a list and convert to set for scoring
        
        for iteration in range(5000):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random element to change (index 1 to end)
            if len(current_set) < 2:
                break
                
            idx = random.randint(1, len(current_set) - 1)
            old_val = current_set[idx]
            
            # Propose a new value
            # Random jump or small step
            if random.random() < 0.5:
                # Small step
                step = random.randint(-10, 10)
                new_val = old_val + step
            else:
                # Random restart within range
                max_range = max(current_set) * 2
                new_val = random.randint(1, max_range)
                
            if new_val <= 0:
                continue
                
            # Create new set
            new_set_list = current_set.copy()
            new_set_list[idx] = new_val
            new_set = sorted(set(new_set_list))
            
            # Ensure 0 is present and first
            if 0 not in new_set:
                new_set.append(0)
                new_set.sort()
                
            # Check if size changed (if duplicate was created)
            # If size decreased, it might be bad, but let's score it anyway
            # However, if we remove an element, we might lose the "0" if 0 was removed? 
            # We ensured 0 is added back.
            
            new_score = compute_score(new_set)
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                best_score = current_score
                best_set = list(current_set)
            else:
                # Optionally, accept worse moves with small probability (Simulated Annealing style)
                # But for simplicity and speed, stick to Hill Climbing
                pass
                
    return best_set
# EVOLVE-BLOCK-END
