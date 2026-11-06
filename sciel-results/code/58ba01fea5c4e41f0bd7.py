import time
import random
import math

def _score(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d in S.
    Uses bitmask for fast checking.
    S is a set of integers in 1..512.
    """
    mask = 0
    for x in S:
        mask |= (1 << x)
    
    valid_ds = 0
    # d can be at most (512-1)/2 = 255.5, so max d is 255.
    # Actually, if a=1, a+2d=512, 2d=511, d=255.5 not int.
    # Max a=1, max a+2d=512. 2d <= 511. d <= 255.
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are in S
        # Bit a in mask is 1
        # Bit a in (mask >> d) is 1 iff bit a+d in mask is 1
        # Bit a in (mask >> 2d) is 1 iff bit a+2d in mask is 1
        # So we check if mask & (mask >> d) & (mask >> 2d) != 0
        if mask & (mask >> d) & (mask >> 2*d):
            valid_ds += 1
    return valid_ds

def construct():
    start_time = time.time()
    # Strict deadline: 100s to be safely under 120s limit
    deadline = start_time + 100.0
    
    # Initial best: Simple AP [1..32]
    # Score is number of d such that a, a+d, a+2d in S.
    # For 1..32, d can be 1..15 (1, 1+15, 1+30=31 is in set. 1, 1+16, 1+32=33 not in set).
    initial_set = set(range(1, 33))
    best_set = set(initial_set)
    best_score = _score(best_set)
    
    # Simulated Annealing
    while time.time() < deadline:
        # Start SA run
        if random.random() < 0.1:
            current_set = set(random.sample(range(1, 513), 32))
        else:
            current_set = set(best_set)
            
        current_score = _score(current_set)
        
        T = 10.0  # Higher initial temperature for better exploration
        T_min = 0.05
        cooling_rate = 0.9995
        
        # Inner loop with strict time check
        while time.time() < deadline:
            # Pick a move: swap one element
            # Pick element to remove
            # Using a list for fast random access
            # We maintain a list representation of current_set for speed
            # But current_set is a set. Let's just pick a random element from it.
            # To avoid O(N) list conversion every time, we can keep a list in sync.
            # However, for 32 elements, list(current_set) is very fast.
            
            # Optimization: Keep a list of elements to avoid list() conversion overhead?
            # Let's stick to set for simplicity but be aware of overhead.
            # Actually, let's just use the set.
            
            elem_to_remove = random.choice(tuple(current_set))
            
            # Pick new value
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
                
            # Perform move
            current_set.remove(elem_to_remove)
            current_set.add(new_val)
            
            new_score = _score(current_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                # Revert
                current_set.remove(new_val)
                current_set.add(elem_to_remove)
                
            T *= cooling_rate
            if T < T_min:
                break
                
    return list(best_set)
