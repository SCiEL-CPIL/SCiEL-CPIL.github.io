import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    
    # Initial set: Upper half [65..128]
    initial_mask = (1 << n) - 1
    lower_mask = (1 << (n//2)) - 1
    current_mask = initial_mask & ~lower_mask
    
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    def get_size(mask):
        return bin(mask).count('1')
        
    def is_valid_addition(mask, x):
        """
        Check if adding element x (1-indexed) to the set represented by mask is valid.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is a vertex (c=x).
        # Need a, b in S distinct from x, a != b, such that:
        # a+x in S, b+x in S, and (a+b in S or a+b == x)
        
        # Find all a in S such that a+x in S
        candidates_a = []
        for a in range(1, min(x, n - x + 1) + 1):
            if a == x:
                continue
            if (mask >> (a - 1)) & 1:
                if (mask >> (a + x - 1)) & 1:
                    candidates_a.append(a)
        
        k = len(candidates_a)
        if k > 1:
            for i in range(k):
                a = candidates_a[i]
                for j in range(i + 1, k):
                    b = candidates_a[j]
                    if a == b:
                        continue
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and (mask >> (s - 1)) & 1:
                        return False
        
        # Case 2: x is a sum. x = a + b
        # Need c in S \ {a,b} such that a+c in S and b+c in S
        
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                # Check if there exists c in S \ {a,b} with a+c in S and b+c in S
                C_a_mask = mask & (mask >> a)
                C_b_mask = mask & (mask >> b)
                potential_c_mask = C_a_mask & C_b_mask
                potential_c_mask &= ~(1 << (a - 1))
                potential_c_mask &= ~(1 << (b - 1))
                if potential_c_mask != 0:
                    return False
        
        return True
    
    def try_greedy_add(mask):
        """Try to add missing elements in ascending order. Returns new_mask."""
        added = False
        for x in range(1, n + 1):
            if not (mask >> (x - 1)) & 1:
                if is_valid_addition(mask, x):
                    mask |= (1 << (x - 1))
                    added = True
        return mask
    
    def random_local_search(mask, time_limit):
        """Perform randomized local search with add/remove moves."""
        current = mask
        best_local = mask
        best_local_size = get_size(mask)
        attempts = 0
        max_attempts = 200000
        
        while time.time() < time_limit and attempts < max_attempts:
            attempts += 1
            
            # Try to add a random missing element
            x = random.randint(1, n)
            if not (current >> (x - 1)) & 1:
                if is_valid_addition(current, x):
                    current |= (1 << (x - 1))
                    new_size = get_size(current)
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local = current
            
            # Occasionally remove a random element to escape local optima
            if random.random() < 0.12:
                idx = random.randint(0, n - 1)
                if (current >> idx) & 1:
                    current &= ~(1 << idx)
            
            # If current is much smaller than best, reset to best
            if get_size(current) < best_local_size - 4:
                current = best_local
        
        return best_local, best_local_size
    
    # Phase 1: Greedy addition from initial set
    current_mask = try_greedy_add(current_mask)
    best_size = get_size(current_mask)
    best_mask = current_mask
    
    # Phase 2: Multiple rounds of random local search with restarts
    # Use adaptive time allocation
    remaining_time = deadline - time.time()
    num_rounds = max(5, int(remaining_time / 3))  # At least 5 rounds, ~3s each
    
    for round_idx in range(num_rounds):
        if time.time() >= deadline - 3:
            break
        
        # Restart from best_mask with random perturbation
        current = best_mask
        # Randomly remove 0-2 elements to create diversity
        for _ in range(random.randint(0, 2)):
            bits = [i for i in range(n) if (current >> i) & 1]
            if bits:
                idx = random.choice(bits)
                current &= ~(1 << idx)
        
        # Run local search with time limit
        round_deadline = min(time.time() + 2.5, deadline - 3)
        local_best, local_size = random_local_search(current, round_deadline)
        
        if local_size > best_size:
            best_size = local_size
            best_mask = local_best
            # After improvement, do another greedy pass
            best_mask = try_greedy_add(best_mask)
            best_size = get_size(best_mask)
    
    # Final greedy pass
    best_mask = try_greedy_add(best_mask)
    best_size = get_size(best_mask)
    
    # Convert best_mask to list of ints
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
