import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 57.0  # Leave 3s buffer before 60s kill
    
    # Precompute conflict masks
    # conflicts[x] is a list of bitmasks representing the other 3 elements in a 4-AP containing x
    conflicts = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            
            # Mask for the other 3 elements for each position in the AP
            # For a: other are b, c, dd
            mask_a = (1 << b) | (1 << c) | (1 << dd)
            # For b: other are a, c, dd
            mask_b = (1 << a) | (1 << c) | (1 << dd)
            # For c: other are a, b, dd
            mask_c = (1 << a) | (1 << b) | (1 << dd)
            # For dd: other are a, b, c
            mask_dd = (1 << a) | (1 << b) | (1 << c)
            
            conflicts[a].append(mask_a)
            conflicts[b].append(mask_b)
            conflicts[c].append(mask_c)
            conflicts[dd].append(mask_dd)
            
    # Helper: Check if x can be added to S (bitset)
    def can_add(S, x):
        for mask in conflicts[x]:
            if (S & mask) == mask:
                return False
        return True

    # Helper: Greedy fill
    def greedy_fill(S, order):
        S = S | 0  # Ensure int
        for x in order:
            if not (S & (1 << x)):  # If x not in S
                if can_add(S, x):
                    S |= (1 << x)
        return S

    best_S = 0
    best_size = 0
    random.seed(42)
    
    # Generate base-3 set as a potential seed
    base3_set = 0
    for i in range(1, n + 1):
        # Check if base 3 representation has only 0s and 1s
        num = i
        ok = True
        while num > 0:
            if num % 3 == 2:
                ok = False
                break
            num //= 3
        if ok:
            base3_set |= (1 << i)
    
    # Try starting from base3_set and filling
    order = list(range(1, n + 1))
    random.shuffle(order)
    S = greedy_fill(base3_set, order)
    best_S = S
    best_size = bin(S).count('1')
    
    # Random restarts
    while time.time() - start_time < time_limit * 0.7:
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_fill(0, order)
        
        # Local search / Kick
        # Try to add more
        while True:
            improved = False
            # Find candidates
            for x in order:
                if not (S & (1 << x)):
                    if can_add(S, x):
                        S |= (1 << x)
                        improved = True
            if not improved:
                break
        
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
            
        # Kick if stuck
        if size < best_size:
             # Try to kick from best_S
             pass # We restart from scratch mostly, but kicking from best is better
        
    # More aggressive local search on best_S
    S = best_S
    # Repeatedly try to add, and if stuck, kick
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try adding
        for x in range(1, n + 1):
            if not (S & (1 << x)):
                if can_add(S, x):
                    S |= (1 << x)
                    improved = True
        if not improved:
            # Kick: remove 1-3 random elements from S
            if bin(S).count('1') > 3:
                # Get list of set bits
                bits = [i for i in range(1, n + 1) if (S & (1 << i))]
                kick_size = random.randint(1, 3)
                to_remove = random.sample(bits, kick_size)
                for x in to_remove:
                    S &= ~(1 << x)
                # Try to fill again
                for x in range(1, n + 1):
                    if not (S & (1 << x)):
                        if can_add(S, x):
                            S |= (1 << x)
                
                if bin(S).count('1') > best_size:
                    best_size = bin(S).count('1')
                    best_S = S

    # Convert bitset to list
    result = [i for i in range(1, n + 1) if (best_S & (1 << i))]
    return result
