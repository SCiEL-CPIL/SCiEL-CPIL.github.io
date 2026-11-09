import time
import random
import numpy as np

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)  # in_S[i] is True if i is in S
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    # Max sum is 2000
    counts = [0] * (2 * max_val + 1)
    
    # Helper to add a number to the set and update counts
    # Returns True if successful, False if it would violate B2[2]
    def try_add(x):
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # If we get here, it's valid.
        # Perform addition
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    # Helper to remove a number
    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    # Helper to check if a swap (remove a, add c) is valid
    # Does not modify state, just checks
    def is_valid_swap(a, c):
        # Sums affected:
        # Removed: 2*a, a+y (y in S, y!=a)
        # Added: 2*c, c+y (y in S\{a})
        
        # Let's use a temporary dictionary for deltas.
        deltas = {}
        
        # Remove a
        deltas[2*a] = deltas.get(2*a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
        
        # Add c
        deltas[2*c] = deltas.get(2*c, 0) + 1
        for y in S:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
        
        # Check all affected sums
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False # Should not happen
        
        return True

    # 1. Greedy Initialization
    for i in range(1, max_val + 1):
        if try_add(i):
            pass

    # 2. Local Search with Simulated Annealing-like acceptance
    # We will try to add random unused numbers.
    # If it fails, we try to swap an existing number with the candidate.
    
    # To speed up random selection of unused numbers, we can keep a list of unused numbers.
    # But maintaining it is overhead. 
    # Random picking from 1..1000 and checking in_S is O(1).
    
    # Track best solution
    best_S = list(S)
    best_size = len(S)
    
    # Temperature for simulated annealing (accepting worse moves sometimes)
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute list of all numbers for random selection
    all_numbers = list(range(1, max_val + 1))
    
    while time.time() < deadline:
        c = random.choice(all_numbers)
        if in_S[c]:
            continue
            
        # Try to add c
        if try_add(c):
            # Improvement
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
            T = max(0.1, T * cooling_rate)
            continue
            
        # If add failed, try to find a swap
        if not S:
            continue
            
        # Try up to 10 random swaps per failed addition
        swapped = False
        for _ in range(10):
            a = random.choice(S)
            if is_valid_swap(a, c):
                # Perform swap
                try_remove(a)
                try_add(c)
                swapped = True
                # Swap keeps size same, but might enable future additions
                T = max(0.1, T * cooling_rate)
                break
        
        # If we didn't swap, we just discard c for now.
        # The loop continues.
        
        # Occasionally, try a "remove" move to escape local minima?
        # The current program doesn't do this, but inspiration 2 does.
        # Let's add a small probability of removing a random element to allow rearrangement.
        if random.random() < 0.05 and S:
            a = random.choice(S)
            try_remove(a)
            T = max(0.1, T * cooling_rate)

    # Return the best solution found
    return best_S
