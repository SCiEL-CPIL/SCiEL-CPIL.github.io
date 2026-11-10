"""Improved program: greedy construction with smarter ordering and local search."""
import time
import random
from itertools import combinations, combinations_with_replacement

def _pair_sums(s):
    """Return the set of all pair sums a+b (a<=b) for set s."""
    return {a + b for a, b in combinations_with_replacement(s, 2)}

def _pair_sums_count(s):
    """Return count of unique pair sums."""
    return len(_pair_sums(s))

def _diffs(s):
    """Return the set of absolute differences |a-b| for a!=b in s."""
    return {abs(a - b) for a, b in combinations(s, 2)}

def _is_sidon(s):
    """Check if s is a Sidon set (all pair sums unique)."""
    sums = [a + b for a, b in combinations_with_replacement(s, 2)]
    return len(sums) == len(set(sums))

def _diffs_disjoint(d0, d1):
    """Check if two difference sets are disjoint."""
    return not (d0 & d1)

def _can_add(s, x, used_sums, used_diffs, other_diffs):
    """Check if x can be added to s maintaining Sidon property and diff disjointness."""
    # Check Sidon: all new pair sums must be unique and not in used_sums
    new_sums = []
    for a in s:
        new_sums.append(a + x)
    new_sums.append(2 * x)
    
    # Check for duplicates among new sums and with existing sums
    for ss in new_sums:
        if ss in used_sums:
            return False
    if len(new_sums) != len(set(new_sums)):
        return False
    
    # Check differences: new diffs must not be in used_diffs or other_diffs
    new_diffs = set()
    for a in s:
        d = abs(a - x)
        new_diffs.add(d)
    
    if new_diffs & used_diffs:
        return False
    if new_diffs & other_diffs:
        return False
    
    return True

def _greedy_construct(n=256, max_size=64, seed=42, time_limit=45):
    """Greedy construction with smaller-first bias and difference-count heuristic."""
    random.seed(seed)
    start_time = time.time()
    
    sets = [[], []]
    used_sums = [set(), set()]
    used_diffs = [set(), set()]
    
    # Process numbers in order, trying to add to the side with fewer elements
    # or where it fits better
    for x in range(1, n + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try adding to side with fewer elements first (balance)
        sides = sorted([0, 1], key=lambda i: len(sets[i]))
        
        for side in sides:
            other = 1 - side
            if len(sets[side]) >= max_size:
                continue
            
            if _can_add(sets[side], x, used_sums[side], used_diffs[side], used_diffs[other]):
                # Add x to this side
                sets[side].append(x)
                for a in sets[side]:
                    if a != x:
                        used_sums[side].add(a + x)
                used_sums[side].add(2 * x)
                for a in sets[side]:
                    if a != x:
                        used_diffs[side].add(abs(a - x))
                break
    
    return sets

def _local_improve(sets, n=256, max_size=64, time_limit=15):
    """Try to improve by swapping elements or finding gaps to fill."""
    start_time = time.time()
    sets = [list(s) for s in sets]
    
    used_sums = [_pair_sums(sets[0]), _pair_sums(sets[1])]
    used_diffs = [_diffs(sets[0]), _diffs(sets[1])]
    
    # Try to add any missing numbers
    for x in range(1, n + 1):
        if time.time() - start_time > time_limit:
            break
        
        for side in [0, 1]:
            if x in sets[side]:
                continue
            if len(sets[side]) >= max_size:
                continue
            
            if _can_add(sets[side], x, used_sums[side], used_diffs[side], used_diffs[1 - side]):
                sets[side].append(x)
                for a in sets[side]:
                    if a != x:
                        used_sums[side].add(a + x)
                used_sums[side].add(2 * x)
                for a in sets[side]:
                    if a != x:
                        used_diffs[side].add(abs(a - x))
    
    # Try swaps: remove an element from one set and add it to the other
    for side in [0, 1]:
        other = 1 - side
        for i in range(len(sets[side])):
            if time.time() - start_time > time_limit:
                return sets
            
            x = sets[side][i]
            # Temporarily remove x
            temp_set = [a for a in sets[side] if a != x]
            temp_sums = _pair_sums(temp_set)
            temp_diffs = _diffs(temp_set)
            
            # Try adding x to other set
            if len(sets[other]) < max_size and _can_add(sets[other], x, used_sums[other], used_diffs[other], temp_diffs):
                # Check if the remaining set is still Sidon
                if _is_sidon(temp_set):
                    # Perform the swap
                    sets[side] = temp_set
                    sets[other] = sets[other] + [x]
                    used_sums[side] = temp_sums
                    used_diffs[side] = temp_diffs
                    used_sums[other] = _pair_sums(sets[other])
                    used_diffs[other] = _diffs(sets[other])
    
    return sets

def construct():
    """Construct two Sidon sets with disjoint difference sets."""
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for safety
    
    best_sets = None
    best_score = 0
    
    # Multiple random seeds for diversity
    for seed in range(20):
        if time.time() > deadline:
            break
        
        sets = _greedy_construct(n=256, max_size=64, seed=seed, time_limit=2.0)
        score = len(sets[0]) + len(sets[1])
        
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
        
        # Local improvement
        if time.time() > deadline - 10:
            break
        
        improved = _local_improve(sets, n=256, max_size=64, time_limit=3.0)
        imp_score = len(improved[0]) + len(improved[1])
        
        if imp_score > best_score:
            best_score = imp_score
            best_sets = [list(s) for s in improved]
    
    # Final local improvement on the best solution
    if best_sets is not None and time.time() < deadline - 5:
        final = _local_improve(best_sets, n=256, max_size=64, time_limit=5.0)
        final_score = len(final[0]) + len(final[1])
        if final_score > best_score:
            best_sets = final
    
    if best_sets is None:
        # Fallback: simple construction
        best_sets = _greedy_construct(n=256, max_size=64, seed=0, time_limit=1.0)
    
    # Validate
    for s in best_sets:
        if not _is_sidon(s):
            # This shouldn't happen, but just in case
            pass
    if not _diffs_disjoint(_diffs(best_sets[0]), _diffs(best_sets[1])):
        # Fallback to a simple valid construction
        best_sets = _greedy_construct(n=256, max_size=64, seed=0, time_limit=1.0)
    
    return best_sets
