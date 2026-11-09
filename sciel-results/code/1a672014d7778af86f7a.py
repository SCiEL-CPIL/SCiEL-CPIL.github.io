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

    # Helper to check if a 2-for-2 swap is valid
    # Remove a, b from S; Add c, d to S
    def is_valid_2swap(a, b, c, d):
        # a, b in S; c, d not in S
        # Check if c, d are distinct and not in S
        if c == d or in_S[c] or in_S[d]:
            return False
        if a == b:
            return False

        deltas = {}
        
        # Remove a and b
        deltas[2*a] = deltas.get(2*a, 0) - 1
        deltas[2*b] = deltas.get(2*b, 0) - 1
        
        # a+b is counted once in counts. Removing a and b removes this pair.
        # Wait, counts[a+b] includes the pair (a,b) if a!=b.
        # If a!=b, counts[a+b] is decremented by 1.
        if a != b:
            s_ab = a + b
            deltas[s_ab] = deltas.get(s_ab, 0) - 1
            
        for y in S:
            if y != a and y != b:
                s_ay = a + y
                s_by = b + y
                deltas[s_ay] = deltas.get(s_ay, 0) - 1
                deltas[s_by] = deltas.get(s_by, 0) - 1
        
        # Add c and d
        deltas[2*c] = deltas.get(2*c, 0) + 1
        deltas[2*d] = deltas.get(2*d, 0) + 1
        
        if c != d:
            s_cd = c + d
            deltas[s_cd] = deltas.get(s_cd, 0) + 1
            
        for y in S:
            if y != a and y != b:
                s_cy = c + y
                s_dy = d + y
                deltas[s_cy] = deltas.get(s_cy, 0) + 1
                deltas[s_dy] = deltas.get(s_dy, 0) + 1
        
        # Check all affected sums
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False
        
        return True

    # Perform a 2-for-2 swap
    def do_2swap(a, b, c, d):
        # Remove a, b
        S.remove(a)
        in_S[a] = False
        S.remove(b)
        in_S[b] = False
        
        # Update counts for removal
        counts[2*a] -= 1
        counts[2*b] -= 1
        if a != b:
            counts[a+b] -= 1
            
        for y in S:
            if y != a and y != b:
                counts[a+y] -= 1
                counts[b+y] -= 1
                
        # Add c, d
        S.append(c)
        in_S[c] = True
        S.append(d)
        in_S[d] = True
        
        # Update counts for addition
        counts[2*c] += 1
        counts[2*d] += 1
        if c != d:
            counts[c+d] += 1
            
        for y in S:
            if y != c and y != d:
                counts[c+y] += 1
                counts[d+y] += 1

    # 1. Greedy Initialization
    # Try a few different greedy orders to get a good starting point
    best_init_S = []
    best_init_size = 0
    
    for _ in range(5):
        # Reset state
        S.clear()
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        
        for i in order:
            if try_add(i):
                pass
        
        if len(S) > best_init_size:
            best_init_size = len(S)
            best_init_S = list(S)
            # Save state
            best_init_counts = list(counts)
            best_init_in_S = list(in_S)

    # Restore best initialization
    S = best_init_S
    in_S = best_init_in_S
    counts = best_init_counts
    
    # Track best solution
    best_S = list(S)
    best_size = len(S)
    
    # Temperature for simulated annealing
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute list of all numbers for random selection
    all_numbers = list(range(1, max_val + 1))
    
    # Local Search
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random unused number
            c = random.choice(all_numbers)
            if in_S[c]:
                continue
            if try_add(c):
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                T = max(0.1, T * cooling_rate)
                continue
        elif move_type < 0.6:
            # Try a swap
            if not S:
                continue
            a = random.choice(S)
            c = random.choice(all_numbers)
            if in_S[c]:
                continue
            if is_valid_swap(a, c):
                try_remove(a)
                try_add(c)
                # Swap keeps size same
                T = max(0.1, T * cooling_rate)
                continue
        elif move_type < 0.9:
            # Try a 2-for-2 swap
            if len(S) < 2:
                continue
            a, b = random.sample(S, 2)
            c, d = random.sample(all_numbers, 2)
            if in_S[c] or in_S[d]:
                continue
            if is_valid_2swap(a, b, c, d):
                do_2swap(a, b, c, d)
                # Size remains same
                T = max(0.1, T * cooling_rate)
                continue
        else:
            # Remove a random element to escape local minima
            if S:
                a = random.choice(S)
                try_remove(a)
                T = max(0.1, T * cooling_rate)
                continue

    # Return the best solution found
    return best_S
