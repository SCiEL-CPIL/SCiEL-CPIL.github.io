from fractions import Fraction
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # 1. Precompute minimal conflicts
    # min_conflicts[x] is a list of frozensets T such that sum(1/y for y in T) == 1/x
    # and no proper subset of T sums to 1/x.
    # Elements in T must be from [1, 24] \ {x}.
    
    min_conflicts = {x: [] for x in range(1, 25)}
    
    # We need to find subsets of [1, 24] \ {x} that sum to 1/x.
    # To make this efficient, we can use MITM for each x.
    # Since 24 is small, we can even brute force subsets if we prune, 
    # but MITM is safer.
    
    for x in range(1, 25):
        target = Fraction(1, x)
        available = [y for y in range(1, 25) if y != x]
        
        # Find all subsets of 'available' that sum to 'target'
        # Use MITM
        n = len(available)
        if n == 0:
            continue
            
        mid = n // 2
        left = available[:mid]
        right = available[mid:]
        
        # Generate all subset sums for left and right
        # Store as (sum, mask) to reconstruct set
        # Since we only need minimal sets, we can store the sets directly if count is low,
        # but count might be high.
        
        left_sums = {}
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            if s <= target:
                # Store sum -> list of masks
                if s not in left_sums:
                    left_sums[s] = []
                left_sums[s].append(mask)
                
        right_sums = {}
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if s <= target:
                if s not in right_sums:
                    right_sums[s] = []
                right_sums[s].append(mask)
        
        # Find combinations
        all_conflict_sets = set()
        
        # Check left-only
        if target in left_sums:
            for mask in left_sums[target]:
                subset = set()
                for i in range(len(left)):
                    if mask & (1 << i):
                        subset.add(left[i])
                if subset:
                    all_conflict_sets.add(frozenset(subset))
                    
        # Check right-only
        if target in right_sums:
            for mask in right_sums[target]:
                subset = set()
                for i in range(len(right)):
                    if mask & (1 << i):
                        subset.add(right[i])
                if subset:
                    all_conflict_sets.add(frozenset(subset))
                    
        # Check left + right
        for ls, lmasks in left_sums.items():
            needed = target - ls
            if needed in right_sums:
                for lm in lmasks:
                    for rm in right_sums[needed]:
                        subset = set()
                        for i in range(len(left)):
                            if lm & (1 << i):
                                subset.add(left[i])
                        for i in range(len(right)):
                            if rm & (1 << i):
                                subset.add(right[i])
                        if subset:
                            all_conflict_sets.add(frozenset(subset))
                            
        # Filter to minimal sets
        # A set T is minimal if no proper subset T' in all_conflict_sets is a subset of T
        # Note: The empty set is not in all_conflict_sets because target > 0 and we skip empty masks?
        # Actually, mask 0 gives sum 0. We only added if subset is non-empty.
        
        minimal_sets = []
        conflict_list = list(all_conflict_sets)
        
        # Sort by size to check smaller ones first
        conflict_list.sort(key=len)
        
        for T in conflict_list:
            is_minimal = True
            for T_prime in conflict_list:
                if T_prime == T:
                    continue
                if len(T_prime) >= len(T):
                    break # Since sorted by size, if len >= len(T) and not equal, it can't be a proper subset
                if T_prime.issubset(T):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_sets.append(T)
                
        min_conflicts[x] = minimal_sets

    # 2. Fast validity check
    def is_valid(S):
        S_set = set(S)
        for x in S_set:
            others = S_set - {x}
            for T in min_conflicts[x]:
                if T.issubset(others):
                    return False
        return True

    # 3. Search for max cardinality
    best_set = set()
    
    # Strategy: Start with all elements, remove offending ones, then add back.
    
    def remove_offenders(S):
        S_set = set(S)
        changed = True
        while changed:
            changed = False
            # Find a conflict
            for x in list(S_set):
                others = S_set - {x}
                for T in min_conflicts[x]:
                    if T.issubset(others):
                        # Remove one element from T or x?
                        # Removing x is safe.
                        # Removing an element from T might resolve multiple conflicts.
                        # Let's try removing x first (simplest).
                        # Or remove the element in T that appears in the most other conflicts?
                        # For now, remove x.
                        S_set.remove(x)
                        changed = True
                        break
                if changed:
                    break
        return S_set

    # Initial Greedy: Start with all, remove offenders
    current_set = remove_offenders(set(range(1, 25)))
    best_set = current_set
    
    # Local Search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        excluded = set(range(1, 25)) - current_set
        
        # Try to add any excluded element
        for x in list(excluded):
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                improved = True
                # Restart adding loop to check new possibilities
                excluded = set(range(1, 25)) - current_set
                break
        
        if improved:
            continue
            
        # Try swaps: remove y from current, add x from excluded
        # To increase size, we must add more than we remove.
        # Since we are at a local max for adding, we must swap 1-for-1 to escape local optima?
        # No, swapping 1-for-1 keeps size same. We want to increase size.
        # If we can't add, we are stuck at this size for this configuration.
        # We can try to remove an element to allow adding two others?
        # That's a 1-for-2 swap.
        
        # Try 1-for-2 swaps
        found_swap = False
        for y in list(current_set):
            if time.time() > deadline - 10:
                break
            temp_set = current_set - {y}
            for x1 in excluded:
                if x1 in current_set: continue
                new_set_1 = temp_set | {x1}
                if is_valid(new_set_1):
                    # Now try to add another
                    excluded_2 = set(range(1, 25)) - new_set_1
                    for x2 in excluded_2:
                        if x2 == x1: continue
                        new_set_2 = new_set_1 | {x2}
                        if is_valid(new_set_2):
                            current_set = new_set_2
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            if found_swap:
                break
        
        if not found_swap:
            # Try random restarts if stuck
            if random.random() < 0.1:
                # Randomize
                pass # Maybe implement later if needed

    # Final check
    if len(current_set) > len(best_set):
        best_set = current_set

    # If we have time, try random restarts
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # Random subset of size 15
            rand_set = set(random.sample(range(1, 25), 15))
            # Fix it
            rand_set = remove_offenders(rand_set)
            # Grow it
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                excluded = set(range(1, 25)) - rand_set
                for x in excluded:
                    if is_valid(rand_set | {x}):
                        rand_set.add(x)
                        changed = True
                        break
            if len(rand_set) > len(best_set):
                best_set = rand_set

    if not best_set:
        best_set = {24}
        
    return sorted(best_set)
