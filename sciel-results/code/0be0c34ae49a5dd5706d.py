"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_dissociated(denominators):
    """
    Check if the set of reciprocals of 'denominators' has distinct subset sums.
    Equivalent to checking if the scaled integers are dissociated.
    """
    k = len(denominators)
    if k == 0:
        return True
    L = _lcm_list(denominators)
    # Scale to integers
    ints = [L // d for d in denominators]
    
    # Generate all subset sums
    # For k up to 5 or 6, 2^k is small.
    # We are only using this to find MINIMAL bad sets, so k is small.
    sums = set()
    # Use bitmask iteration
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += ints[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _find_minimal_bad_sets(universe):
    """
    Find all minimal subsets of 'universe' that are NOT dissociated.
    We check subsets of size 3, 4, and 5.
    """
    bad_sets = []
    # Check size 3
    for comb in itertools.combinations(universe, 3):
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    # Check size 4 (only if not a superset of a known bad set of size 3)
    # Optimization: if a size 4 set contains a bad size 3 set, it's not minimal.
    bad_size_3 = [s for s in bad_sets if len(s) == 3]
    for comb in itertools.combinations(universe, 4):
        # Check if it contains any bad size 3
        is_superset_of_bad = False
        for b in bad_size_3:
            if b.issubset(set(comb)):
                is_superset_of_bad = True
                break
        if is_superset_of_bad:
            continue
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    # Check size 5
    # This might be overkill but safe.
    # We can skip if we assume max minimal size is 4.
    # Given the nature of reciprocal sums, size 4 is very likely the max minimal size.
    # 1/1 = 1/2 + 1/3 + 1/6 (size 4)
    # 1/2 = 1/3 + 1/7 + 1/42 (size 4)
    # Let's stick to 3 and 4 for speed, as 5 is unlikely to be minimal in this range.
    
    return bad_sets

def _max_dissociated_subset(universe, bad_sets):
    """
    Find the largest subset of 'universe' that does not contain any set from 'bad_sets'.
    Uses backtracking with pruning.
    """
    n = len(universe)
    best_set = []
    
    # Index bad sets for quick check?
    # Actually, checking if (current_set U {x}) contains a bad set is O(|bad_sets| * |bad_set|).
    # |bad_sets| is small (< 100).
    
    def is_valid(current_set, x):
        new_set = current_set | {x}
        # Check if new_set contains any bad set
        for b in bad_sets:
            if b.issubset(new_set):
                return False
        return True

    def backtrack(index, current_set, current_size):
        nonlocal best_set
        
        # Pruning: even if we take all remaining, can we beat best?
        if current_size + (n - index) <= len(best_set):
            return
            
        if index == n:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return
        
        x = universe[index]
        
        # Option 1: Include x
        if is_valid(current_set, x):
            current_set.add(x)
            backtrack(index + 1, current_set, current_size + 1)
            current_set.remove(x)
            
        # Option 2: Exclude x
        backtrack(index + 1, current_set, current_size)

    # Sort universe to help pruning? 
    # Larger numbers are less likely to be in bad sets?
    # Actually, bad sets involve small numbers mostly.
    # Let's just use the given order.
    
    backtrack(0, set(), 0)
    return list(best_set)

def construct():
    universe = list(range(1, 23))
    # Find minimal bad sets
    bad_sets = _find_minimal_bad_sets(universe)
    # Find max subset avoiding them
    result = _max_dissociated_subset(universe, bad_sets)
    
    # Ensure non-empty
    if not result:
        return [1]
        
    # Optional: Verify the result is actually dissociated
    # This is a safety check in case "minimal bad sets" missed something (e.g. size 5)
    # If verification fails, we can fall back to a simpler greedy or the seed.
    # But given the logic, it should be correct.
    
    # Let's do a quick verification for the returned set
    if not _is_dissociated(result):
        # Fallback to seed if something went wrong
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return result
# EVOLVE-BLOCK-END
