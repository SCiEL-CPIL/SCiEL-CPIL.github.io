import random
import time

# Global cache for scaled values to avoid recomputation if construct is called multiple times?
# The task says "Each evaluation must finish within 60s". 
# Usually construct is called once per evaluation.
# But to be safe, I'll compute it inside.

def construct():
    start_time = time.time()
    timeout = 85.0
    
    universe = list(range(1, 23))
    LCM = 232792560
    scaled_vals = {x: LCM // x for x in universe}
    
    def is_dissociated_fast(S):
        if not S:
            return True
        sums = {0}
        for x in S:
            v = scaled_vals[x]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    best_set = set()
    best_size = 0
    
    # Randomized Greedy
    # We want to maximize cardinality.
    # Smaller denominators (larger values) might be more "conflicting"?
    # Or larger denominators (smaller values) might be more "conflicting" because they are smaller?
    # Actually, dissociated sets often look like powers of 2 or similar.
    # Here we have fixed values.
    
    # Strategy 1: Random Greedy
    for _ in range(10000):
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(universe)
        current_set = set()
        
        for x in universe:
            candidate = current_set | {x}
            if is_dissociated_fast(candidate):
                current_set = candidate
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we hit 12, we can stop searching for 11s, but keep looking for 13?
            # Max possible is 22. But likely 11 or 12.
            # If best_size >= 12, we can reduce the number of restarts or just continue.
            
    # Strategy 2: Local Search / Hill Climbing
    # Try to improve the best_set found.
    # We will try to swap elements to find a configuration that allows adding more.
    
    visited = set()
    visited.add(frozenset(best_set))
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > timeout:
            break
            
        S = set(best_set)
        remaining = set(universe) - S
        
        # Try all swaps
        for i in list(S):
            for j in remaining:
                if time.time() - start_time > timeout:
                    break
                
                candidate = (S - {i}) | {j}
                if candidate in visited:
                    continue
                
                if is_dissociated_fast(candidate):
                    visited.add(candidate)
                    
                    # Check if we can add any new element to this candidate
                    added = False
                    for k in (set(universe) - candidate):
                        if is_dissociated_fast(candidate | {k}):
                            # Found a better set!
                            best_set = candidate | {k}
                            best_size = len(best_set)
                            improved = True
                            added = True
                            break
                    
                    if added:
                        break
                        
            if improved:
                break

    return list(best_set)
