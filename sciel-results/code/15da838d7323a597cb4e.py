"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    # Optimization: If we have the set, we can just check all
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    # Fast check: 
    # 1. Does x divide sum of any 3 in selected?
    n = len(selected)
    if n >= 3:
        for triple in combinations(selected, 3):
            if sum(triple) % x == 0:
                return False
    
    # 2. Does any a in selected divide sum of (3 others) where one is x?
    # The 3 others must be chosen from selected \ {a}. One of them is x.
    # So we need b, c in selected \ {a} such that (x + b + c) % a == 0.
    # This is equivalent to: exists b, c in selected \ {a} such that b+c == k*a - x.
    # Or simply check all pairs b, c in selected \ {a}.
    for a in selected:
        # We need to pick 2 others from selected \ {a} to join x.
        # Let others be selected \ {a}.
        # We iterate pairs from others.
        # If n is small, this is fast.
        # To be safe and simple, just iterate pairs.
        others = [v for v in selected if v != a]
        if len(others) < 2:
            continue
        for b, c in combinations(others, 2):
            if (x + b + c) % a == 0:
                return False
    return True

def _backtracking_search(candidates, current_set, max_size, time_limit, start_time):
    """
    Backtracking search to maximize set size.
    candidates: list of numbers to consider (sorted)
    current_set: list of currently selected numbers
    """
    best_set = list(current_set)
    best_size = len(current_set)
    
    # If we hit time limit, return best
    if time.time() > time_limit:
        return best_set, best_size
    
    if not candidates:
        return best_set, best_size
    
    # Heuristic: Try to pick larger numbers first? Or smaller?
    # Usually, larger numbers are "safer" as divisors because sums of others are smaller
    # relative to the divisor? No, sums can be large.
    # Actually, if 'a' is large, 'sum % a == 0' is less likely unless sum is a multiple of a.
    # If 'a' is small, 'sum % a == 0' is very likely.
    # So we want to AVOID small numbers? Or include them carefully?
    # The constraint is "No selected integer may divide...".
    # Small numbers are dangerous divisors.
    # So we should probably prefer large numbers.
    
    # Let's try descending order in candidates.
    # But we also want to fill up to max_size.
    
    # Pruning: If len(current_set) + len(candidates) <= best_size, prune.
    if len(current_set) + len(candidates) <= best_size:
        return best_set, best_size
        
    # Try adding the first candidate
    x = candidates[0]
    rest = candidates[1:]
    
    if _can_add(current_set, x):
        new_set = current_set + [x]
        # Update best if larger
        if len(new_set) > best_size:
            best_set = list(new_set)
            best_size = len(new_set)
            
        # Recurse
        sub_best, sub_size = _backtracking_search(rest, new_set, max_size, time_limit, start_time)
        if sub_size > best_size:
            best_set = sub_best
            best_size = sub_size
            
    # Try skipping the candidate
    sub_best, sub_size = _backtracking_search(rest, current_set, max_size, time_limit, start_time)
    if sub_size > best_size:
        best_set = sub_best
        best_size = sub_size
        
    return best_set, best_size

def _greedy_descending():
    """Greedy from 192 down to 1."""
    selected = []
    for x in range(192, 0, -1):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _greedy_ascending():
    """Greedy from 1 up to 192."""
    selected = []
    for x in range(1, 193):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _greedy_random_order():
    """Greedy with random order."""
    candidates = list(range(1, 193))
    random.shuffle(candidates)
    selected = []
    for x in candidates:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # 1. Get a good initial solution via greedy
    # Descending greedy usually works well for "avoid dividing" because large numbers are less likely to be divisors of sums of smaller numbers?
    # Wait, if I pick 192, 191, ..., the sum of 3 others can be up to ~570.
    # 192 does not divide 570. 191 does not.
    # Small numbers like 2, 3, 4 are very likely to divide sums.
    # So descending greedy should be good.
    
    best = _greedy_descending()
    best_size = len(best)
    
    # Try ascending just in case
    asc = _greedy_ascending()
    if len(asc) > best_size:
        best = asc
        best_size = len(asc)
        
    # Try a few random greedy runs
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        rand = _greedy_random_order()
        if len(rand) > best_size:
            best = rand
            best_size = len(rand)
            
    # 2. Backtracking search to improve
    # We want to add more elements.
    # The greedy set is a valid set.
    # We can try to find a larger set by backtracking from the start,
    # but that might be too slow for 192 numbers.
    # Instead, let's use the greedy set as a "seed" and try to add more?
    # No, backtracking from scratch with pruning is better if we order candidates well.
    
    # Let's try a backtracking search that starts with the greedy set?
    # That's complex.
    # Let's try a backtracking search from scratch, but with a time limit.
    # To make it faster, we can use the greedy solution size as a lower bound for pruning.
    
    # Candidates: 1..192.
    # If we do full backtracking on 192 items, it's 2^192. Impossible.
    # We need strong pruning.
    
    # Alternative: Local Search on the greedy set.
    # Try to swap elements to allow more additions?
    
    # Let's try a "Repair" strategy:
    # Take the greedy set.
    # Try to add elements one by one. If it fails, try to remove one existing element and add two new ones?
    
    # Actually, the "inspiration" mentioned backtracking with descending heuristic improved score.
    # Let's implement a limited-depth backtracking or a smart local search.
    
    # Let's try a simple local search:
    # 1. Try to add any missing number.
    # 2. If none can be added, try to swap one number for another to free up space?
    
    # Better: Randomized Local Search
    # Start with best set.
    # Repeat:
    #   Pick a random element in set, remove it.
    #   Try to add random elements not in set.
    #   If size increases, keep. Else revert.
    
    # But the current _local_search in the seed was weak.
    # Let's try a more aggressive local search.
    
    def _local_search_improved(start_set, time_limit):
        current = list(start_set)
        in_set = set(current)
        best = list(current)
        best_size = len(current)
        
        # Pre-compute all numbers not in set
        not_in_set = [x for x in range(1, 193) if x not in in_set]
        
        # Phase 1: Greedy addition
        while True:
            added = False
            # Shuffle to break symmetry
            random.shuffle(not_in_set)
            for x in not_in_set:
                if _can_add(current, x):
                    current.append(x)
                    in_set.add(x)
                    not_in_set.remove(x)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    added = True
                    break
            if not added or time.time() > time_limit:
                break
                
        # Phase 2: Swaps to enable more additions
        # Try to remove one element and add two others?
        # Or just swap one for another if it helps future additions?
        
        # Let's try: Remove one element, try to add as many as possible from not_in_set.
        # If the resulting set is larger, accept.
        
        for _ in range(100): # Limit iterations
            if time.time() > time_limit:
                break
            
            # Pick a random element to remove
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            new_current = [x for i, x in enumerate(current) if i != idx]
            new_in_set = set(new_current)
            
            # Try to add as many as possible from not_in_set (excluding removed, which is now available)
            # We can reuse the greedy addition logic on new_current
            # But we need to be careful with time.
            
            # Simple check: Can we add 'removed' back? No, we want to improve.
            # Let's just try to add any element from not_in_set + [removed]
            # Actually, 'removed' is not in new_current.
            # Let's create a candidate pool: not_in_set + [removed]
            pool = not_in_set + [removed]
            random.shuffle(pool)
            
            improved = False
            for x in pool:
                if x in new_in_set:
                    continue
                if _can_add(new_current, x):
                    new_current.append(x)
                    new_in_set.add(x)
                    if len(new_current) > best_size:
                        best = list(new_current)
                        best_size = len(current) # Wait, len(new_current)
                        best_size = len(new_current)
                        improved = True
                    # Keep adding?
                    # Let's try to add multiple
                    break # Restart addition loop for simplicity? Or continue?
                    # Let's continue adding to new_current
                    
            # If we added anything, update current
            if len(new_current) > len(current):
                current = new_current
                in_set = set(current)
                # Update not_in_set
                not_in_set = [x for x in range(1, 193) if x not in in_set]
                
            # Also try: Remove one, add one (swap)
            # This might help if a small number is blocking many large ones?
            # Unlikely, but possible.
            
        return best

    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        if remaining > 1.0:
            improved = _local_search_improved(best, remaining)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                
    return best
# EVOLVE-BLOCK-END
