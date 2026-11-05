import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    
    # State
    A = set()
    # counts[s] = number of ordered pairs (i,j) in A x A such that i+j=s
    counts = np.zeros(max_sum + 1, dtype=int)
    
    def add_element(x):
        # Update counts
        counts[2*x] += 1
        for y in A:
            counts[x+y] += 1
        A.add(x)
        
    def remove_element(x):
        counts[2*x] -= 1
        for y in A:
            counts[x+y] -= 1
        A.remove(x)
            
    def get_cost():
        # Calculate max unordered multiplicity in target range
        # Also check for coverage
        max_mult = 0
        uncovered = 0
        for s in target_sums:
            ordered = counts[s]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= N and half in A:
                    equal = 1
                else:
                    equal = 0
            else:
                equal = 0
            
            unordered = (ordered + equal) // 2
            
            if unordered == 0:
                uncovered += 1
            else:
                if unordered > max_mult:
                    max_mult = unordered
        
        if uncovered > 0:
            return 1000 + uncovered
        return max_mult

    # Phase 1: Initialize
    random.seed(42)
    # Start with a set that likely covers
    # Greedy addition
    for _ in range(20):
        x = random.randint(0, N)
        if x not in A:
            add_element(x)
            
    # Ensure coverage
    # Simple loop: while uncovered, add best
    while time.time() < deadline - 5:
        cost = get_cost()
        if cost < 1000: # Covered
            break
            
        # Find best x to add
        best_x = None
        best_gain = -1
        
        # Sample candidates to speed up if A is large, but N=128 is small
        candidates = [x for x in range(N+1) if x not in A]
        if not candidates:
             break
             
        # To speed up, we can just pick a few random candidates and check gain
        # Or iterate all if time permits.
        # Let's iterate all 128, it's fast.
        for x in candidates:
            # Calculate gain
            gain = 0
            # Check if 2x is uncovered
            # We need to know if sum is currently 0.
            # counts[2x] == 0?
            # If counts[2x] == 0, then 2x is uncovered (unless 2x > 256 or < 128)
            if 128 <= 2*x <= 256 and counts[2*x] == 0:
                gain += 1
                
            for y in A:
                s = x + y
                if 128 <= s <= 256 and counts[s] == 0:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_x = x
                
        if best_x is None or best_gain == 0:
            break
            
        add_element(best_x)

    # Phase 2: Local Search
    current_cost = get_cost()
    best_cost = current_cost
    best_A = set(A)
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Swap
        if len(A) > 0:
            x_in = random.choice(list(A))
            x_out = random.randint(0, N)
            if x_out == x_in:
                continue
                
            # Swap
            remove_element(x_in)
            add_element(x_out)
            
            new_cost = get_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        remove_element(x_out)
                        add_element(x_in)
                else:
                    current_cost = new_cost
        else:
            # A is empty? Unlikely, but handle
            x_out = random.randint(0, N)
            add_element(x_out)
            new_cost = get_cost()
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
            else:
                remove_element(x_out)

        T *= cooling
        if T < 0.1:
            T = 0.1

        if best_cost <= 2:
            break

    return list(best_A)
