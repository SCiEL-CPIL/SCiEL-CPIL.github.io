import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

def _get_weights():
    """Precompute scaled weights for exact integer arithmetic."""
    # LCM of 1 to 22
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
    # Prime powers:
    # 2^4 = 16
    # 3^2 = 9
    # 5
    # 7
    # 11
    # 13
    # 17
    # 19
    # LCM = 16 * 9 * 5 * 7 * 11 * 13 * 17 * 19
    lcm_val = 16 * 9 * 5 * 7 * 11 * 13 * 17 * 19
    # lcm_val = 232792560
    weights = [lcm_val // i for i in range(1, 23)]
    return weights

def _is_dissociated_fast(S, weights):
    """
    Checks if S is dissociated using exact integer sums.
    S is a list of indices (0-based) or values (1-based)? 
    Let's use 1-based values for clarity, but map to 0-based for weights.
    Actually, passing the list of weights directly is faster.
    """
    # S contains values from 1 to 22.
    # We need the corresponding weights.
    # To avoid lookup overhead, let's pass the list of weights directly to this function?
    # Or precompute a dict.
    # Let's just use a list of weights corresponding to the elements in S.
    
    seen = {0}
    for val in S:
        w = weights[val - 1]
        # Generate new sums
        new_sums = []
        for s in seen:
            ns = s + w
            # If ns is in seen, collision
            if ns in seen:
                return False
            new_sums.append(ns)
        seen.update(new_sums)
    return True

def _greedy_build(order, weights):
    """Greedily build a dissociated set following the given order."""
    current_set = []
    for x in order:
        candidate = current_set + [x]
        if _is_dissociated_fast(candidate, weights):
            current_set = candidate
    return current_set

def _local_search_add_and_swap(S, weights, deadline):
    """
    Local search to maximize cardinality.
    1. Try to add any element.
    2. If stuck, try systematic swaps to explore the space of same-size dissociated sets.
    Uses a more aggressive strategy: try multiple swap candidates per stuck state.
    """
    current = list(S)
    current_set = set(current)
    universe = set(range(1, 23))
    
    # Track best size found during this local search
    best_local = list(current)
    best_size = len(current)
    
    while time.time() < deadline:
        # Phase 1: Try to add - try all available elements in a smart order
        added = False
        available = [x for x in universe if x not in current_set]
        
        # Sort by weight descending (larger elements first) as they're more likely to fit
        # but also try a random mix for diversity
        available_sorted = sorted(available, key=lambda x: weights[x-1], reverse=True)
        # Interleave sorted and random for better exploration
        random.shuffle(available)
        mixed_order = available_sorted[:len(available)//2] + available[len(available)//2:]
        random.shuffle(mixed_order)
        
        for x in mixed_order:
            candidate = current + [x]
            if _is_dissociated_fast(candidate, weights):
                current = candidate
                current_set.add(x)
                if len(current) > best_size:
                    best_size = len(current)
                    best_local = list(current)
                added = True
                break
        
        if added:
            continue # Restart add phase
        
        # Phase 2: Stuck, try multiple swaps to escape local optimum
        if not current:
            break
            
        x_in_candidates = [x for x in universe if x not in current_set]
        if not x_in_candidates:
            break # Full set, nothing to do
            
        # Try up to 10 swap attempts before giving up on this state
        swap_found = False
        for _ in range(10):
            x_out = random.choice(current)
            x_in = random.choice(x_in_candidates)
            
            # Perform swap
            new_set_list = [x for x in current if x != x_out] + [x_in]
            if _is_dissociated_fast(new_set_list, weights):
                current = new_set_list
                current_set = set(current)
                swap_found = True
                break
        
        if not swap_found:
            # Try a more aggressive move: remove 2, add 2 (if possible)
            if len(current) >= 2 and len(x_in_candidates) >= 2:
                # Try removing a random pair and adding a random pair
                pair_out = random.sample(current, 2)
                pair_in = random.sample(x_in_candidates, 2)
                new_set_list = [x for x in current if x not in pair_out] + pair_in
                if _is_dissociated_fast(new_set_list, weights):
                    current = new_set_list
                    current_set = set(current)
                    swap_found = True
        
        if not swap_found:
            # No swap worked, break out to avoid infinite loop
            break
            
    return best_local

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s safety margin
    
    weights = _get_weights()
    universe = list(range(1, 23))
    
    # Initial best
    # Try a few deterministic greedy orders first
    best_set = []
    best_size = 0
    
    # Order 1: Descending
    s1 = _greedy_build(list(range(22, 0, -1)), weights)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # Order 2: Ascending
    s2 = _greedy_build(list(range(1, 23)), weights)
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # Order 3: Randomized permutations
    random.seed(42)
    
    # Run local search from the best initial set
    if time.time() < deadline - 5:
        improved = _local_search_add_and_swap(best_set, weights, deadline - 5)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
            
    # If we have time, try multiple random restarts
    # We can try to find sets of increasing size by random sampling + local search
    # But local search from a good seed is usually sufficient.
    
    # Let's do a few more random greedy starts + local search
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Random order
        random_order = list(universe)
        random.shuffle(random_order)
        
        seed_set = _greedy_build(random_order, weights)
        
        # Local search from this seed
        if time.time() < deadline - 1:
            improved = _local_search_add_and_swap(seed_set, weights, deadline - 1)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(best_set)
                
        # Early exit if we hit max possible? 
        # Max possible is 22, but unlikely. 
        # If we hit 18 or 19, it's very good.
        if best_size >= 19:
             break
             
    if not best_set:
        # Fallback to a known good small set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    # Final verification: ensure the returned set is valid
    if _is_dissociated_fast(best_set, weights):
        return best_set
    else:
        # If somehow invalid, return a known good set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
