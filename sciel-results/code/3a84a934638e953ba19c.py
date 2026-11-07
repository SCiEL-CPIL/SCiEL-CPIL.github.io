"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    n = 128
    
    # Start with a random or heuristic initial set
    # Heuristic: Upper half is dense but sums are often > 128, so safe.
    # Let's try starting with a random subset of size ~60 to allow more flexibility?
    # Or stick to the proven upper half start but randomize the greedy order?
    
    def is_valid_addition(x, S):
        """Check if adding x to S keeps S valid."""
        # Case 1: x is a member (a=x)
        # Need b,c in S distinct, b!=x, c!=x
        # x+b in S, x+c in S, b+c in S
        A_x = []
        for b in S:
            if b != x and (x + b) in S:
                A_x.append(b)
        
        if len(A_x) >= 2:
            # Check if any pair in A_x sums to something in S
            for i in range(len(A_x)):
                for j in range(i + 1, len(A_x)):
                    if A_x[i] + A_x[j] in S:
                        return False
        
        # Case 2: x is a sum (a+b=x)
        # Need a,b,c in S distinct
        # a+b=x, a+c in S, b+c in S
        # Note: a+c cannot be x (implies c=b), b+c cannot be x (implies c=a)
        # So a+c, b+c must be strictly in S
        
        for a in S:
            b = x - a
            if b <= a or b not in S:
                continue
            # Found pair (a,b) with a+b=x
            # Check if there is c in S, c!=a, c!=b
            # such that a+c in S and b+c in S
            
            # Optimization: Precompute sets? 
            # For now, iterate c
            for c in S:
                if c == a or c == b:
                    continue
                if (a + c) in S and (b + c) in S:
                    return False
        
        return True

    # Initial Set: Upper Half
    S = set(range(65, n + 1))
    
    # Greedy Add
    # Try adding smaller numbers
    # Randomize order to find different local optima?
    # Let's do multiple restarts with different random seeds/orders if time permits.
    
    best_size = len(S)
    best_S = set(S)
    
    # Greedy Pass 1: Deterministic (1 to 64)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S):
            S.add(x)
    
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Hill Climbing / Local Search
    # We try to swap elements or add elements
    # To improve size, we mainly want to ADD elements.
    # If we can't add, we might need to remove one to allow adding two?
    
    # Let's do a random restart strategy with randomized greedy order
    # and then local search.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize the greedy order for the lower half
        lower_half = list(range(1, 65))
        random.shuffle(lower_half)
        
        S_try = set(range(65, n + 1))
        
        for x in lower_half:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_try):
                S_try.add(x)
        
        # Local Search: Try to add any remaining number
        # Iterate over all numbers not in S_try
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x not in S_try:
                    if is_valid_addition(x, S_try):
                        S_try.add(x)
                        improved = True
                        break # Restart scan from 1? Or just continue?
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_S = set(S_try)
            
        # If we have time, try to swap to increase size?
        # Swap: remove y, add z. Size stays same.
        # To increase size, we need a sequence of swaps that ends with a net add.
        # This is complex. The "Add if possible" loop above should find most improvements.
        
    return sorted(list(best_S))

# EVOLVE-BLOCK-END
