# EVOLVE-BLOCK-START
import random
import time
import sys

def construct():
    """
    Find a large Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses randomized backtracking with dynamic variable ordering (MRV heuristic)
    and bitset-based conflict checking.
    """
    required = [1, 2, 5, 11]
    max_val = 284
    target_size = 20 # We aim for the theoretical max
    
    # Precompute candidates
    all_candidates = [x for x in range(1, max_val + 1) if x not in required]
    
    best_result = list(required)
    best_size = len(required)
    
    # Time limit awareness
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    def get_initial_sums(elements):
        sums = set()
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                sums.add(elements[i] + elements[j])
        return sums

    def can_add_bitset(used_sums, elements, x):
        """
        Check if x can be added.
        Returns (True, new_used_sums) if valid, (False, None) otherwise.
        """
        # Check x+e for all e in elements
        # And 2*x
        new_sums = set()
        for e in elements:
            s = x + e
            if s in used_sums:
                return False, None
            new_sums.add(s)
        
        s_2x = 2 * x
        if s_2x in used_sums:
            return False, None
        new_sums.add(s_2x)
        
        # Check for duplicates within new_sums
        # Since elements are distinct, x+e are distinct for distinct e.
        # 2*x might equal x+e if x==e, but e is in elements and x is not (candidates are disjoint).
        # So no internal duplicates in new_sums.
        
        return True, used_sums | new_sums

    def solve(elements, used_sums, candidates, depth, max_depth, seed):
        """
        Randomized backtracking solver.
        """
        nonlocal best_result, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if len(elements) > best_size:
            best_result = list(elements)
            best_size = len(elements)
            
        if best_size >= target_size:
            return
            
        if depth >= max_depth:
            return
            
        if not candidates:
            return
            
        # Dynamic Variable Ordering (MRV-like):
        # We want to pick a candidate that is "hard" to place, i.e., it conflicts with many other candidates?
        # Or simpler: pick a candidate that has the fewest valid extensions?
        # Given the density, a simpler heuristic is often effective: 
        # Sort candidates by value (smallest first tends to constrain larger ones) 
        # OR Randomize to explore space.
        # Let's use a hybrid: shuffle, then pick the first one that works? 
        # No, MRV suggests picking the variable with the smallest domain.
        # In this context, every candidate is a "variable" being added.
        # A better heuristic for Sidon sets is often to prioritize smaller numbers 
        # because they create sums that are more likely to collide with other small sums.
        
        # However, pure smallest-first can get stuck.
        # Let's try a few different orderings in a single call or across restarts.
        
        # Heuristic: Sort candidates by "constraint potential"
        # A candidate x is more constrained if it forms sums with existing elements that are "busy".
        # For simplicity and speed, we will shuffle the candidates to randomize the path,
        # but we will also try a deterministic "smallest first" pass initially.
        
        # To implement MRV properly:
        # For each candidate x, count how many other candidates y are "blocked" by x?
        # This is expensive. 
        # Alternative: Just shuffle and pick. Randomized backtracking with restarts is robust.
        
        # We will iterate through shuffled candidates.
        shuffled_candidates = list(candidates)
        random.Random(seed).shuffle(shuffled_candidates)
        
        for x in shuffled_candidates:
            if time.time() - start_time > time_limit:
                return
                
            valid, new_sums = can_add_bitset(used_sums, elements, x)
            if valid:
                new_elements = elements + [x]
                # Update candidates: remove x. 
                # We could also remove candidates that become invalid, but checking validity 
                # at the next level is cheaper than maintaining a global validity map 
                # unless we do strict forward checking.
                # Let's just pass the remaining candidates (minus x).
                new_candidates = [c for c in candidates if c != x]
                
                solve(new_elements, new_sums, new_candidates, depth + 1, max_depth, seed)
                
                if best_size >= target_size:
                    return

    # Strategy:
    # 1. Try a greedy deterministic "smallest first" to get a baseline quickly.
    # 2. Run multiple randomized backtracking searches with different depths/seeds.
    
    # 1. Greedy Baseline
    elements = list(required)
    elements.sort()
    used_sums = get_initial_sums(elements)
    candidates = list(all_candidates)
    candidates.sort()
    
    for x in candidates:
        if time.time() - start_time > time_limit:
            break
        valid, new_sums = can_add_bitset(used_sums, elements, x)
        if valid:
            elements.append(x)
            used_sums = new_sums
            
    if len(elements) > best_size:
        best_result = elements
        best_size = len(elements)
        
    if best_size >= target_size:
        return sorted(best_result)

    # 2. Randomized Backtracking
    # We will perform several restarts.
    # Each restart uses a random seed and a limited depth to avoid getting stuck too deep in a bad branch.
    # However, since we want max size, deeper is better. 
    # We'll use a "beam search" like approach or just deep dives with restarts.
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        seed = 1000 + i
        random.seed(seed)
        
        # Reset state for this restart
        elements = list(required)
        elements.sort()
        used_sums = get_initial_sums(elements)
        candidates = list(all_candidates)
        
        # Limit depth to avoid excessive recursion time if it gets stuck, 
        # but allow enough to find large sets.
        # If we find a set of size 19, we can stop early in the search.
        max_depth = 20 
        
        solve(elements, used_sums, candidates, 0, max_depth, seed)
        
        if best_size >= target_size:
            break

    # Ensure result is valid and contains required elements
    result = sorted(best_result)
    if not all(x in result for x in required):
        # Fallback if something went wrong
        result = [1, 2, 5, 11]
        
    # Final validation
    sums = set()
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
            
    if not valid:
        result = [1, 2, 5, 11]

    return result

# EVOLVE-BLOCK-END
