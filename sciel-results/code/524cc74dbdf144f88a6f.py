"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from itertools import combinations
import random

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """
    For each element a in [1,24], find all subsets of [1,24]\{a}
    whose reciprocal sum equals 1/a.
    Returns a dict: a -> list of frozensets (the forbidden subsets)
    """
    universe = list(range(1, 25))
    conflicts = {}
    for a in universe:
        others = [x for x in universe if x != a]
        target = Fraction(1, a)
        forbidden = []
        # Use meet-in-the-middle for efficiency
        n = len(others)
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        # Enumerate all subsets of left, store (sum, bitmask)
        left_sums = {}  # sum -> list of bitmasks
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            if s not in left_sums:
                left_sums[s] = []
            left_sums[s].append(mask)
        
        # For each subset of right, check if complement exists in left
        for mask in range(1 << len(right)):
            s_right = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s_right += Fraction(1, right[i])
            needed = target - s_right
            if needed in left_sums:
                for lmask in left_sums[needed]:
                    # Combine masks
                    full_mask = lmask | (mask << len(left))
                    if full_mask == 0:
                        continue  # empty subset, skip
                    subset = set()
                    for i in range(len(left)):
                        if lmask & (1 << i):
                            subset.add(left[i])
                    for i in range(len(right)):
                        if mask & (1 << i):
                            subset.add(right[i])
                    forbidden.append(frozenset(subset))
        
        conflicts[a] = forbidden
    
    return conflicts

def _is_valid(subset, conflicts):
    """
    Check if subset is reciprocal-sum-free.
    For each a in subset, no forbidden subset for a should be entirely contained in subset.
    """
    subset_set = set(subset)
    for a in subset_set:
        for forbidden in conflicts[a]:
            if forbidden.issubset(subset_set):
                return False
    return True

def _count_violations(subset, conflicts):
    """Count number of active conflict constraints."""
    subset_set = set(subset)
    count = 0
    for a in subset_set:
        for forbidden in conflicts[a]:
            if forbidden.issubset(subset_set):
                count += 1
    return count

def construct():
    deadline = time.time() + 80  # 80s deadline with safety margin
    
    # Step 1: Precompute conflicts
    conflicts = _precompute_conflicts()
    
    # Step 2: Greedy construction - start with all elements, remove those causing conflicts
    # Or: start empty, add elements that don't create conflicts
    # Let's try: start with all 24, then iteratively remove elements that are in conflicts
    
    current = set(range(1, 25))
    best = set(current)
    
    # Greedy removal: while there are violations, remove the element involved in the most conflicts
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Count how many active conflicts each element is involved in
        conflict_count = {a: 0 for a in current}
        for a in current:
            for forbidden in conflicts[a]:
                if forbidden.issubset(current):
                    # This conflict is active. Count it for 'a' and for each element in forbidden
                    conflict_count[a] += 1
                    for x in forbidden:
                        if x in current:
                            conflict_count[x] += 1
        
        max_count = max(conflict_count.values()) if conflict_count else 0
        if max_count > 0:
            # Remove the element with the highest conflict count
            candidates = [a for a in current if conflict_count[a] == max_count]
            # Tie-break: remove the one that's in the most forbidden subsets
            random.shuffle(candidates)
            to_remove = candidates[0]
            current.discard(to_remove)
            changed = True
        
        if len(current) > len(best):
            best = set(current)
    
    # Step 3: Local search - try to add back elements
    if time.time() < deadline:
        # Try adding elements one at a time
        for a in range(1, 25):
            if a not in current and time.time() >= deadline:
                break
            trial = current | {a}
            if _is_valid(trial, conflicts):
                current = trial
                best = set(current)
    
    # Step 4: Simulated annealing / local search to improve
    if time.time() < deadline:
        current_set = set(current)
        best_set = set(best)
        
        # Local search: try swapping or adding
        for _ in range(10000):
            if time.time() >= deadline:
                break
            # Try adding a random element
            if len(current_set) < 24:
                candidates_to_add = [a for a in range(1, 25) if a not in current_set]
                if candidates_to_add:
                    a = random.choice(candidates_to_add)
                    trial = current_set | {a}
                    if _is_valid(trial, conflicts):
                        current_set = trial
                        if len(current_set) > len(best_set):
                            best_set = set(current_set)
            
            # Try removing and re-adding
            if current_set:
                a = random.choice(list(current_set))
                trial = current_set - {a}
                if _is_valid(trial, conflicts):
                    current_set = trial
                    # Now try to add something back
                    candidates_to_add = [x for x in range(1, 25) if x not in current_set]
                    random.shuffle(candidates_to_add)
                    for x in candidates_to_add:
                        trial2 = current_set | {x}
                        if _is_valid(trial2, conflicts):
                            current_set = trial2
                            if len(current_set) > len(best_set):
                                best_set = set(current_set)
                            break
    
    # Return the best valid set found
    result = list(best_set) if best_set else list(current)
    if not result:
        result = [13]
    
    # Final validation
    if not _is_valid(result, conflicts):
        # Fallback to a known valid set
        result = list(range(13, 25))
    
    return result

# EVOLVE-BLOCK-END
