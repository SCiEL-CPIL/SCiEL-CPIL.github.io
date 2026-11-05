"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def has_4ap_set(s):
    """
    Check if a set of integers contains a non-constant 4-term arithmetic progression.
    Returns True if a 4-AP exists, False otherwise.
    Optimized for small-to-medium sets by checking pairs (a, b) and looking for (b+d, b+2d).
    """
    n = len(s)
    if n < 4:
        return False
    
    # Convert to sorted list for potential optimization, but set lookup is O(1)
    # We iterate over all pairs (a, b) with a < b
    # d = b - a
    # Check if b+d and b+2d are in s
    # Note: This is O(n^2) checks. For n~200, that's 40k checks, very fast.
    
    # To speed up, we can just iterate over the set directly.
    # However, iterating over a set in Python is not ordered.
    # We need a < b.
    # Let's just use a double loop over the list for simplicity and predictability,
    # or better, iterate over the set and for each a, iterate over the set again for b?
    # Actually, the most efficient pure Python way for small n is:
    
    s_list = list(s)
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(len(s_list)):
            b = s_list[j]
            if a >= b:
                continue
            d = b - a
            if d == 0:
                continue
            c = b + d
            e = c + d
            if c in s and e in s:
                return True
    return False

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    Uses a hybrid approach:
    1. Start with a greedy construction to get a valid base set.
    2. Use randomized local search to try to add more elements.
    3. Use simulated annealing style swaps if stuck.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n_max = 512
    all_elements = set(range(1, n_max + 1))
    
    # Step 1: Greedy Construction (Fast, provides a valid starting point)
    # We use a slightly smarter greedy: try to add elements in a random order?
    # No, standard greedy (1, 2, 3...) is deterministic and fast.
    # Let's try a few different greedy orders to get a good base.
    
    def greedy_random_order(order):
        selected = set()
        for x in order:
            # Check if adding x creates a 4-AP
            # x can be part of 4-AP in 4 ways:
            # 1. x, x+d, x+2d, x+3d
            # 2. x-d, x, x+d, x+2d
            # 3. x-2d, x-d, x, x+d
            # 4. x-3d, x-2d, x-d, x
            # We check if adding x to 'selected' creates any 4-AP.
            # Instead of checking all d, we can check if x completes a 4-AP with existing elements.
            
            # Efficient check:
            # For each d > 0:
            #   if x-3d, x-2d, x-d in selected -> conflict (x is 4th term)
            #   if x-2d, x-d, x+d in selected -> conflict (x is 3rd term)
            #   if x-d, x+d, x+2d in selected -> conflict (x is 2nd term)
            #   if x+d, x+2d, x+3d in selected -> conflict (x is 1st term)
            
            valid = True
            # Max d such that x-3d >= 1 => d <= (x-1)//3
            max_d1 = (x - 1) // 3
            for d in range(1, max_d1 + 1):
                if (x - 3*d) in selected and (x - 2*d) in selected and (x - d) in selected:
                    valid = False
                    break
            
            if valid:
                # Max d such that x-2d >= 1 => d <= (x-1)//2
                max_d2 = (x - 1) // 2
                for d in range(1, max_d2 + 1):
                    if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                        valid = False
                        break
            
            if valid:
                # Max d such that x-d >= 1 => d <= x-1
                # And x+2d <= 512 => d <= (512-x)//2
                max_d3 = min(x - 1, (512 - x) // 2)
                for d in range(1, max_d3 + 1):
                    if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                        valid = False
                        break
            
            if valid:
                # Max d such that x+3d <= 512 => d <= (512-x)//3
                max_d4 = (512 - x) // 3
                for d in range(1, max_d4 + 1):
                    if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                        valid = False
                        break
            
            if valid:
                selected.add(x)
        return selected

    # Try a few random permutations for greedy to find a good base
    best_set = set()
    best_size = 0
    
    # First, standard greedy (1..512)
    base_set = greedy_random_order(range(1, 513))
    best_set = base_set
    best_size = len(base_set)
    
    # Try 10 random permutations
    for _ in range(10):
        if time.time() > deadline:
            break
        random_order = list(range(1, 513))
        random.shuffle(random_order)
        s = greedy_random_order(random_order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    # Step 2: Local Search / Hill Climbing
    # Try to add random elements not in best_set
    # If adding fails, try swapping? Swapping is complex. Let's stick to adding first.
    # Then, try to remove one element and add two others? That's 3-opt.
    
    current_set = best_set
    current_size = len(current_set)
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            # Try a different random base? Or break.
            # Let's try restarting local search from a new random greedy base
            random_order = list(range(1, 513))
            random.shuffle(random_order)
            new_base = greedy_random_order(random_order)
            if len(new_base) > current_size:
                current_set = new_base
                current_size = len(new_base)
                no_improvement_count = 0
            else:
                break # If we can't find a better base, we might be stuck
                
        # Pick a random element to try to add
        missing = list(all_elements - current_set)
        if not missing:
            break
            
        x = random.choice(missing)
        
        # Check if x can be added
        valid = True
        max_d1 = (x - 1) // 3
        for d in range(1, max_d1 + 1):
            if (x - 3*d) in current_set and (x - 2*d) in current_set and (x - d) in current_set:
                valid = False
                break
        
        if valid:
            max_d2 = (x - 1) // 2
            for d in range(1, max_d2 + 1):
                if (x - 2*d) in current_set and (x - d) in current_set and (x + d) in current_set:
                    valid = False
                    break
        
        if valid:
            max_d3 = min(x - 1, (512 - x) // 2)
            for d in range(1, max_d3 + 1):
                if (x - d) in current_set and (x + d) in current_set and (x + 2*d) in current_set:
                    valid = False
                    break
        
        if valid:
            max_d4 = (512 - x) // 3
            for d in range(1, max_d4 + 1):
                if (x + d) in current_set and (x + 2*d) in current_set and (x + 3*d) in current_set:
                    valid = False
                    break
        
        if valid:
            current_set.add(x)
            current_size += 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
    # Final verification (paranoia check, though the add logic is sound)
    # Just to be absolutely sure, we can run has_4ap_set on the final set
    # But it's O(N^2) which is fine for N~200.
    if has_4ap_set(current_set):
        # This should never happen if the add logic is correct, but if it does,
        # fall back to the best greedy set which was verified by construction.
        return sorted(best_set)
        
    return sorted(current_set)
# EVOLVE-BLOCK-END
