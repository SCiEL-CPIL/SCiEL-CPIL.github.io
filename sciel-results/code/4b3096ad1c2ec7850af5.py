"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _score_set_bitset(s):
    """
    Fast scoring using bitset operations.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in s.
    """
    if not s:
        return 0
    s_set = set(s)
    max_val = max(s_set)
    # Create a bitset where bit i is 1 if i in s
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    count = 0
    # d can range from 1 to max_val // 2
    # a can range from 1 to max_val - 2d
    # We check if there is ANY a such that a, a+d, a+2d are in s
    # (bitset & (bitset >> d) & (bitset >> (2*d))) != 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
            
    return count

def _construct_difference_first():
    """
    Construct the set by prioritizing the completion of missing differences.
    """
    current_set = set()
    # We want to maximize the number of distinct d.
    # Max possible d is roughly 256.
    
    # To make the heuristic work, we need to know which d's are already satisfied.
    # However, recalculating satisfied d's from scratch every step is O(N^2 * D).
    # Instead, we can estimate the "potential" of adding an element x.
    
    # Heuristic:
    # For each candidate x, count how many new 3-term progressions it forms with existing pairs.
    # Specifically, a progression (a, a+d, a+2d) is completed by x if:
    # 1. x = a, a+d, a+2d are in S (x is start) -> requires x+d, x+2d in S
    # 2. x = a+d -> requires x-d, x+d in S
    # 3. x = a+2d -> requires x-2d, x-d in S
    
    # We want to maximize the number of NEW d's added.
    # A d is "new" if it wasn't satisfied before.
    
    # Efficient approximation:
    # Maintain a set of satisfied differences `sat_diffs`.
    # When adding x, check for each d if x completes a progression for d, and if d is not in `sat_diffs`.
    
    sat_diffs = set()
    
    for step in range(32):
        best_elem = None
        best_new_diffs_count = -1
        
        # Candidate elements
        # We can restrict candidates to those that are "close" to existing elements to ensure progressions form,
        # but for 512 range, iterating all 512 is fast enough if the inner loop is efficient.
        
        for elem in range(1, 513):
            if elem in current_set:
                continue
            
            new_diffs = 0
            
            # Check which differences this element would satisfy
            # We iterate over possible d.
            # d must be such that the progression fits in [1, 512].
            # Also, at least two of the elements in the progression must be in current_set.
            
            # Case 1: elem is the start (a)
            # Need elem+d in S and elem+2d in S
            # d can be up to (512 - elem) // 2
            max_d_start = (512 - elem) // 2
            for d in range(1, max_d_start + 1):
                if d in sat_diffs:
                    continue
                if (elem + d) in current_set and (elem + 2 * d) in current_set:
                    new_diffs += 1
            
            # Case 2: elem is the middle (a+d)
            # Need elem-d in S and elem+d in S
            # d can be up to min(elem-1, 512-elem)
            max_d_mid = min(elem - 1, 512 - elem)
            for d in range(1, max_d_mid + 1):
                if d in sat_diffs:
                    continue
                if (elem - d) in current_set and (elem + d) in current_set:
                    new_diffs += 1
            
            # Case 3: elem is the end (a+2d)
            # Need elem-2d in S and elem-d in S
            # d can be up to elem // 2
            max_d_end = elem // 2
            for d in range(1, max_d_end + 1):
                if d in sat_diffs:
                    continue
                if (elem - 2 * d) in current_set and (elem - d) in current_set:
                    new_diffs += 1
            
            if new_diffs > best_new_diffs_count:
                best_new_diffs_count = new_diffs
                best_elem = elem
        
        if best_elem is not None:
            current_set.add(best_elem)
            # Update sat_diffs
            # We need to re-evaluate which d's are now satisfied.
            # It's faster to just recompute the full set of sat_diffs from scratch every step?
            # N=32, D=256. 32*256 checks is cheap.
            bitset = 0
            for x in current_set:
                bitset |= (1 << x)
            new_sat = set()
            max_val = max(current_set)
            for d in range(1, max_val // 2 + 1):
                if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                    new_sat.add(d)
            sat_diffs = new_sat
        else:
            # If no element adds new diffs, pick a random one to break tie or expand range
            # This might happen if we are stuck.
            available = [x for x in range(1, 513) if x not in current_set]
            if available:
                # Pick one that maximizes connectivity (number of pairs formed)
                # Or just random
                best_elem = random.choice(available)
                current_set.add(best_elem)
                # Update sat_diffs
                bitset = 0
                for x in current_set:
                    bitset |= (1 << x)
                new_sat = set()
                max_val = max(current_set)
                for d in range(1, max_val // 2 + 1):
                    if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                        new_sat.add(d)
                sat_diffs = new_sat

    return list(current_set)

def _local_search_optimize(initial_set, deadline):
    """
    Optimize the set using local search (single swaps).
    """
    current_set = set(initial_set)
    current_score = _score_set_bitset(current_set)
    best_set = list(current_set)
    best_score = current_score
    
    # Pre-compute bitsets for faster swap evaluation? 
    # Swapping requires removing one and adding one.
    # Score calculation is O(D) = O(256).
    # Number of swaps: 32 * 512 = ~16k swaps.
    # 16k * 256 ops = 4M ops. Very fast.
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        current_list = sorted(current_set)
        
        for i in range(len(current_list)):
            elem_to_remove = current_list[i]
            
            # Find best replacement
            best_new_elem = None
            best_new_score = current_score
            
            for new_elem in range(1, 513):
                if new_elem in current_set:
                    continue
                
                # Calculate score of new set
                # To speed up, we can try to update score incrementally, but given low D, full recalc is safer and simple.
                # Let's just recalc.
                new_set = (current_set - {elem_to_remove}) | {new_elem}
                new_score = _score_set_bitset(new_set)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_elem = new_elem
            
            if best_new_elem is not None:
                current_set.remove(elem_to_remove)
                current_set.add(best_new_elem)
                current_score = best_new_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                
                break # Restart loop with new set
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin
    
    # 1. Construct initial set using Difference-First heuristic
    initial_set = _construct_difference_first()
    
    # 2. Optimize using Local Search
    optimized_set, score = _local_search_optimize(initial_set, deadline)
    
    # 3. If time remains, try a few random restarts or perturbations on the best found
    # Given the speed of local search, we can do multiple restarts from the best set with perturbations
    best_set = optimized_set
    best_score = score
    
    # Try 5 random perturbations of the best set
    for _ in range(5):
        if time.time() > deadline:
            break
        
        perturbed = set(best_set)
        # Remove 2 random elements
        for _ in range(2):
            if perturbed:
                to_remove = random.choice(list(perturbed))
                perturbed.remove(to_remove)
        
        # Add 2 random elements
        while len(perturbed) < 32:
            new_elem = random.randint(1, 512)
            if new_elem not in perturbed:
                perturbed.add(new_elem)
        
        # Optimize perturbed set
        opt_set, opt_score = _local_search_optimize(list(perturbed), deadline)
        
        if opt_score > best_score:
            best_score = opt_score
            best_set = opt_set
    
    return best_set

# EVOLVE-BLOCK-END
