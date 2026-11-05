import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 256
    # Precompute squares up to 2*256 = 512
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Precompute conflicts
    # conflicts[x] = set of y such that x+y is a square
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for s in squares:
            y = s - x
            if 1 <= y <= N:
                conflicts[x].add(y)
    
    # Precompute diagonal bad numbers (2x is square)
    diag_bad = [False] * (N + 1)
    for x in range(1, N + 1):
        if 2 * x in squares:
            diag_bad[x] = True

    def is_valid_add(x, in_set):
        if diag_bad[x]:
            return False
        # Check if any neighbor is in the set
        for y in conflicts[x]:
            if in_set[y]:
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Initial greedy with random order
    # We will run multiple random greedy passes and local searches
    
    for _ in range(200): # Max 200 restarts, break if time runs out
        if time.time() > deadline:
            break
            
        # Random permutation of valid candidates
        candidates = [i for i in range(1, N + 1) if not diag_bad[i]]
        random.shuffle(candidates)
        
        in_set = [False] * (N + 1)
        current_set = set()
        
        # Greedy phase
        for x in candidates:
            if is_valid_add(x, in_set):
                in_set[x] = True
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # Local Search Phase
        # We try to improve by adding/removing/swapping
        # To keep it simple and fast, we do a few passes of "try to add" and "try to swap"
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any valid number
            # We can iterate through all not-in-set numbers
            # To speed up, we can just iterate through a random subset or all
            # Given N=256, iterating all is cheap (256 * 15 checks)
            
            # Sort candidates by "potential" (fewer conflicts with current set?) 
            # Or just random order to avoid bias
            not_in_set = [i for i in range(1, N + 1) if not in_set[i] and not diag_bad[i]]
            random.shuffle(not_in_set)
            
            for x in not_in_set:
                if is_valid_add(x, in_set):
                    in_set[x] = True
                    current_set.add(x)
                    improved = True
                    # Don't break, try to add more
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
            
            # 2. Try to swap
            # Pick a random element u from current_set
            # Remove it
            # Try to add as many new elements as possible
            # If size increases, keep; else revert
            
            if current_set:
                u = random.choice(list(current_set))
                
                # Save state
                old_in_set = in_set.copy()
                old_set = set(current_set)
                
                # Remove u
                in_set[u] = False
                current_set.remove(u)
                
                # Try to add others
                # We can try to add all valid ones greedily
                not_in_set_now = [i for i in range(1, N + 1) if not in_set[i] and not diag_bad[i]]
                random.shuffle(not_in_set_now)
                
                added_count = 0
                for x in not_in_set_now:
                    if is_valid_add(x, in_set):
                        in_set[x] = True
                        current_set.add(x)
                        added_count += 1
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                
                # Check if we improved
                if len(current_set) > len(old_set):
                    improved = True
                    # Keep the new state
                else:
                    # Revert
                    in_set = old_in_set
                    current_set = old_set

        # Check time
        if time.time() > deadline:
            break

    # Convert to list for return
    return list(best_set)

# EVOLVE-BLOCK-END
