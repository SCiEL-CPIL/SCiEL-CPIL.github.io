# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute neighbors for fast conflict checking
    # neighbors[u] is a set of v such that (u-v) % m is in forbidden
    neighbors = [set() for _ in range(m)]
    for u in range(m):
        for v in range(u + 1, m):
            d1 = (u - v) % m
            if d1 in forbidden:
                neighbors[u].add(v)
                neighbors[v].add(u)
                
    def is_compatible(v, selected_set):
        # Check if v has no neighbors in selected_set
        # Using set intersection for speed
        if selected_set & neighbors[v]:
            return False
        return True

    # 1. Greedy Initialization
    def greedy_init():
        selected = set()
        # Try adding in natural order
        for candidate in range(m):
            if is_compatible(candidate, selected):
                selected.add(candidate)
        return selected

    best_set = greedy_init()
    best_size = len(best_set)
    
    # 2. Local Search: 1-out, 2-in moves
    # We iterate until no improvement is found or time runs out
    start_time = time.time()
    timeout = 500.0 # Leave 130s margin for 630s limit
    
    while time.time() - start_time < timeout:
        improved = False
        current_set = set(best_set)
        current_size = len(current_set)
        
        # Try to add single elements first (should be none if maximal, but check anyway)
        # Actually, greedy_init produces a maximal set. 
        # But after swaps, it might not be? No, we only accept improvements.
        # However, a 1-out, 2-in move results in a set that might allow more single additions?
        # Yes, so we should loop.
        
        # Step A: Add any possible single element
        added = False
        for v in range(m):
            if v not in current_set and is_compatible(v, current_set):
                current_set.add(v)
                current_size += 1
                added = True
                # Restart the inner loop to check for more additions
                # We can just break and restart the whole while loop logic?
                # Or just continue checking?
                # Let's just break and restart the main loop to be safe and clean.
                break
        
        if added:
            if current_size > best_size:
                best_set = current_set
                best_size = current_size
            continue # Restart while loop to check for more additions or new swaps
        
        # Step B: If no single addition, try 1-out, 2-in swaps
        found_swap = False
        # Shuffle removal order for randomness? Or deterministic?
        # Deterministic is fine for now, but random might help escape.
        # Let's iterate through all u in current_set
        for u in list(current_set):
            # Temporarily remove u
            temp_set = current_set - {u}
            
            # Find candidates compatible with temp_set
            candidates = []
            for v in range(m):
                if v in temp_set:
                    continue
                if is_compatible(v, temp_set):
                    candidates.append(v)
            
            # Check if any pair in candidates is mutually compatible
            # candidates size can be large (up to 205)
            # O(N^2) check
            swap_found = False
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    v1 = candidates[i]
                    v2 = candidates[j]
                    # Check if v1 and v2 are compatible
                    if v2 not in neighbors[v1]:
                        # We can add both
                        new_set = temp_set | {v1, v2}
                        # Size is current_size - 1 + 2 = current_size + 1
                        if current_size + 1 > best_size:
                            best_set = new_set
                            best_size = current_size + 1
                        current_set = new_set
                        current_size += 1
                        found_swap = True
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if found_swap:
                improved = True
                break # Restart main loop to check for more additions
        
        if not improved:
            # No single addition, no 1-out-2-in swap found.
            # We are stuck in a local optimum.
            # Try a random restart with perturbation?
            # Or try 2-out, 3-in?
            # For now, let's try a random restart from a perturbed state
            # Perturb best_set by removing a random element and re-greedy?
            
            # Simple random restart:
            # Pick a random subset of best_set to keep, then greedy fill
            if random.random() < 0.1: # Occasional restart
                # Keep a random 50% of best_set
                keep = set(random.sample(list(best_set), max(1, len(best_set)//2)))
                new_set = set(keep)
                # Greedy fill
                for v in range(m):
                    if v not in new_set and is_compatible(v, new_set):
                        new_set.add(v)
                if len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(new_set)
            else:
                # If we are stuck and not restarting, we might be done?
                # Or we can try 2-out, 3-in?
                # Given time, let's just break if we've been stuck for a while?
                # The loop condition is time-based. If we make no progress, we waste time.
                # Let's break if no improvement in a full scan?
                # Actually, the scan above is exhaustive for 1-out-2-in.
                # So if no swap found, we are at a local max for this neighborhood.
                # Break to save time or do a restart.
                break

    # Convert to array
    return np.array(list(best_set), dtype=np.int64)

# EVOLVE-BLOCK-END
