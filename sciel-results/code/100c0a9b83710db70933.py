import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = 55.0  # 5s margin for 60s limit
    
    # Initial valid set: [171, 256]
    # Size 86
    base_S = list(range(171, n + 1))
    best_S = set(base_S)
    best_size = len(base_S)
    
    candidates = list(range(1, 171))
    
    # Helper to compute sums set for a given list S
    # Sums of b+c for b <= c in S
    def compute_sums(S_list):
        sums = set()
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                sums.add(b + c)
        return sums

    # Initial sums for base_S
    base_sums = compute_sums(base_S)
    
    # Greedy search with random restarts
    # We try to add candidates 1..170 to the set.
    # Since we add smaller numbers, the new number is always the minimum 'a'.
    # We only need to check if 'a' divides any sum of two existing elements.
    
    num_restarts = 0
    while time.time() < deadline:
        # Random permutation of candidates
        perm = candidates[:]
        random.shuffle(perm)
        
        current_S = base_S[:]
        current_sums = set(base_sums) # Copy the set
        
        for x in perm:
            # Check if x divides any sum in current_sums
            # If yes, cannot add x
            # Optimization: check if any multiple of x is in current_sums
            # Range of sums: min ~ 342, max ~ 512
            # Multiples of x in that range
            
            can_add = True
            # Iterate through sums? Or multiples?
            # Sums set size is ~5000. Iterating is fast.
            # But we can be smarter.
            min_sum = 2 * current_S[0] # current_S[0] is min element
            max_sum = 2 * current_S[-1]
            
            # Find first multiple of x >= min_sum
            start_mult = (min_sum + x - 1) // x * x
            
            for m in range(start_mult, max_sum + 1, x):
                if m in current_sums:
                    can_add = False
                    break
            
            if can_add:
                # Add x to current_S
                current_S.append(x)
                current_S.sort() # Keep sorted for min/max? 
                # Actually, we just need to update sums.
                # New sums: x + y for all y in OLD current_S (before adding x)
                # Note: x is smaller than all y, so x <= y is satisfied.
                for y in current_S[:-1]: # Iterate old elements
                    current_sums.add(x + y)
                
                # Update best if size increased?
                # Size increases by 1.
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
        
        num_restarts += 1
        # Early exit if we hit a very high size? 
        # Max possible is 256. 
        # If we get > 100, it's a huge improvement.
        # Let's run until deadline.
        
    return list(best_S)
