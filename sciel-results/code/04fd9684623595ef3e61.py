"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_reciprocal_subsum(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some subset of others."""
    if not others:
        return False
    all_vals = [target_x] + list(others)
    lcm_val = reduce(_lcm, all_vals)
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    n = len(candidate_vals)
    if n == 0:
        return False
    half = n // 2
    left = candidate_vals[:half]
    right = candidate_vals[half:]
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
    for s in left_sums:
        if target_int - s in right_sums:
            return True
    return False

def _is_valid_set(selected):
    """Check if selected set is reciprocal-subsum-free."""
    selected = list(selected)
    for i, x in enumerate(selected):
        others = selected[:i] + selected[i+1:]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def _precompute_conflicts():
    """
    Precompute all conflict triples.
    A conflict triple (x, y, z) means: if all three are selected, there's a violation.
    Specifically: 1/x = 1/y + 1/z, or 1/y = 1/x + 1/z, or 1/z = 1/x + 1/y.
    Also higher-order: 1/x = sum of subset of others.
    
    For efficiency, precompute for each pair (y,z) whether 1/y + 1/z equals 1/x
    for some x. Then during B&B, when we select y and z, we know x is blocked.
    """
    # conflict_map[x] = set of pairs (y,z) such that 1/x = 1/y + 1/z
    # More generally: for each x, find all subsets S of [1..24]\{x} where sum(1/s for s in S) = 1/x
    # Then any two elements in S together with x form a conflict.
    
    # For practical purposes, focus on small subsets (size 2, 3)
    # 1/x = 1/y + 1/z => x = yz/(y+z) => y+z must divide yz
    conflict_map = {x: set() for x in range(1, 25)}
    
    for x in range(1, 25):
        # Check pairs: 1/x = 1/y + 1/z
        for y in range(1, 25):
            if y == x:
                continue
            # 1/z = 1/x - 1/y = (y-x)/(xy)
            # z = xy/(y-x) if y > x
            if y > x:
                num = x * y
                den = y - x
                if num % den == 0:
                    z = num // den
                    if 1 <= z <= 24 and z != x and z != y:
                        conflict_map[x].add(frozenset([y, z]))
        
        # Check triples: 1/x = 1/y + 1/z + 1/w
        for y in range(1, 25):
            if y == x:
                continue
            for z in range(y + 1, 25):
                if z == x:
                    continue
                # 1/w = 1/x - 1/y - 1/z
                # Compute as fraction
                num = y * z
                den1 = x * z
                den2 = x * y
                # 1/x - 1/y - 1/z = (yz - xz - xy) / (xyz)
                num_val = y * z - x * z - x * y
                den_val = x * y * z
                if num_val == 0:
                    continue
                if num_val < 0:
                    continue  # 1/w would be negative
                if den_val % num_val == 0:
                    w = den_val // num_val
                    if 1 <= w <= 24 and w != x and w != y and w != z:
                        conflict_map[x].add(frozenset([y, z, w]))
    
    return conflict_map

def _branch_and_bound(ordered_elements, selected_set, blocked_set, best_set, deadline, conflict_map):
    """
    Branch and bound with incremental conflict tracking.
    
    ordered_elements: list of elements to consider (in order)
    selected_set: set of currently selected elements
    blocked_set: set of elements that cannot be selected (would create conflict)
    best_set: best valid set found so far
    deadline: time limit
    conflict_map: precomputed conflicts
    """
    if time.time() > deadline:
        return best_set
    
    # Upper bound: current selected + remaining that aren't blocked
    remaining_count = 0
    for e in ordered_elements:
        if e not in blocked_set and e not in selected_set:
            remaining_count += 1
    
    if len(selected_set) + remaining_count <= len(best_set):
        return best_set
    
    if not ordered_elements:
        if len(selected_set) > len(best_set):
            best_set = list(selected_set)
        return best_set
    
    # Find the next candidate to branch on
    x = None
    idx = -1
    for i, e in enumerate(ordered_elements):
        if e not in blocked_set and e not in selected_set:
            x = e
            idx = i
            break
    
    if x is None:
        # No more candidates
        if len(selected_set) > len(best_set):
            best_set = list(selected_set)
        return best_set
    
    remaining = ordered_elements[idx + 1:]
    
    # Branch 1: Include x
    new_selected = selected_set | {x}
    
    # Check if adding x creates any conflict with existing selected elements
    # A conflict occurs if for some y in selected, there exists a subset S of (selected \ {y}) 
    # such that 1/y = sum(1/s for s in S) ∪ {x}... 
    # Actually, we need: for some y in new_selected, 1/y = sum of 1/s for s in subset of (new_selected \ {y})
    # This is complex. Instead, use the conflict_map: 
    # If x is selected, and we already have y,z selected where {y,z} in conflict_map[x], then conflict.
    # But we also need to check all subset sizes.
    
    # Simpler: just check validity of new_selected
    if _is_valid_set(list(new_selected)):
        # Determine new blocked elements
        new_blocked = set(blocked_set)
        for other in new_selected:
            if other == x:
                continue
            # For each conflict of x that involves 'other', block the other elements in that conflict
            for conflict_set in conflict_map.get(x, set()):
                if other in conflict_set:
                    for elem in conflict_set:
                        if elem != other and elem != x:
                            new_blocked.add(elem)
        
        result = _branch_and_bound(remaining, new_selected, new_blocked, best_set, deadline, conflict_map)
        if len(result) > len(best_set):
            best_set = result
    
    # Branch 2: Exclude x
    result = _branch_and_bound(remaining, selected_set, blocked_set, best_set, deadline, conflict_map)
    if len(result) > len(best_set):
        best_set = result
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Elements 1-24
    all_elements = list(range(1, 25))
    
    # Greedy initial solution: start with large elements (fewer conflicts)
    greedy_set = []
    for x in sorted(all_elements, reverse=True):
        test_set = greedy_set + [x]
        if _is_valid_set(test_set):
            greedy_set = test_set
    
    best_set = greedy_set
    
    # Precompute conflicts
    conflict_map = _precompute_conflicts()
    
    # B&B with elements in descending order (large elements first, fewer conflicts)
    elements_for_bb = list(range(24, 0, -1))
    
    result = _branch_and_bound(elements_for_bb, set(), set(), best_set, deadline, conflict_map)
    if len(result) > len(best_set):
        best_set = result
    
    # If we have time, try another pass with ascending order
    if time.time() < deadline - 5:
        elements_for_bb2 = list(range(1, 25))
        result2 = _branch_and_bound(elements_for_bb2, set(), set(), best_set, deadline, conflict_map)
        if len(result2) > len(best_set):
            best_set = result2
    
    return best_set
# EVOLVE-BLOCK-END
