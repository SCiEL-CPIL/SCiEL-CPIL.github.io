"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_valid(selected):
    """Check if a set is reciprocal-subsum-free.
    
    No reciprocal of a selected element may equal the sum of reciprocals
    of a subset of the OTHER selected elements.
    """
    if not selected:
        return False
    s = set(selected)
    n = len(s)
    # For each element in the set, check all non-empty proper subsets of the rest
    # We need: for each a in S, for each non-empty T subset of S\{a},
    #   1/a != sum_{t in T} 1/t
    # Use exact rational arithmetic
    recip = {a: Fraction(1, a) for a in s}
    for a in s:
        others = [x for x in s if x != a]
        m = len(others)
        # Check all non-empty subsets of others
        # For small m this is fine; for larger m we need to be careful
        # But max m is 23, so 2^23 = 8M which is too slow for each a
        # Better approach: for each a, compute all subset sums of reciprocals of others
        # and check if 1/a is among them
        # Use meet-in-the-middle or just iterate if m is small enough
        if m <= 20:
            # Direct enumeration
            target = recip[a]
            found = False
            # Enumerate all non-empty subsets via bitmask
            for mask in range(1, 1 << m):
                total = Fraction(0, 1)
                for i in range(m):
                    if mask & (1 << i):
                        total += recip[others[i]]
                if total == target:
                    found = True
                    break
            if found:
                return False
        else:
            # Meet in the middle for larger sets
            target = recip[a]
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Generate all subset sums for left
            left_sums = [Fraction(0, 1)]
            for x in left:
                rx = recip[x]
                left_sums = left_sums + [s + rx for s in left_sums]
            
            # Generate all subset sums for right
            right_sums = [Fraction(0, 1)]
            for x in right:
                rx = recip[x]
                right_sums = right_sums + [s + rx for s in right_sums]
            
            # Check if target can be formed as sum of a subset from left and subset from right
            # at least one non-empty overall
            right_set = set(right_sums)
            for ls in left_sums:
                needed = target - ls
                if needed in right_set:
                    # Check it's not the empty+empty case
                    if ls != Fraction(0,1) or needed != Fraction(0,1):
                        return False
    return True

def _greedy_construction():
    """Greedy construction from largest to smallest with local search."""
    elements = list(range(24, 0, -1))  # largest to smallest
    selected = []
    
    for a in elements:
        candidate = selected + [a]
        if _is_valid(candidate):
            selected.append(a)
    
    # Local search: try to add skipped elements
    skipped = [a for a in range(1, 25) if a not in selected]
    improved = True
    while improved:
        improved = False
        for a in skipped[:]:
            candidate = selected + [a]
            if _is_valid(candidate):
                selected.append(a)
                skipped.remove(a)
                improved = True
    
    return selected

def _local_search_optimize(initial):
    """Try to improve the set via swaps and additions."""
    selected = list(initial)
    selected_set = set(selected)
    all_elements = set(range(1, 25))
    
    best = selected
    best_size = len(selected)
    
    deadline = time.time() + 80  # 80 second deadline
    
    while time.time() < deadline:
        # Try adding any element not in the set
        added = False
        for a in sorted(all_elements - selected_set, reverse=True):
            candidate = selected + [a]
            if _is_valid(candidate):
                selected = candidate
                selected_set = set(selected)
                if len(selected) > best_size:
                    best = list(selected)
                    best_size = len(selected)
                added = True
                break
        if not added:
            # Try swaps: remove one, add another
            swapped = False
            for a in sorted(selected_set, reverse=True):
                for b in sorted(all_elements - selected_set, reverse=True):
                    candidate = [x for x in selected if x != a] + [b]
                    if _is_valid(candidate):
                        selected = candidate
                        selected_set = set(selected)
                        if len(selected) > best_size:
                            best = list(selected)
                            best_size = len(selected)
                        swapped = True
                        break
                if swapped or time.time() > deadline:
                    break
            if not swapped:
                # Try 2-for-2 swaps
                two_swapped = False
                sel_list = sorted(selected_set)
                for i in range(len(sel_list)):
                    for j in range(i+1, len(sel_list)):
                        a, b = sel_list[i], sel_list[j]
                        for c in sorted(all_elements - selected_set, reverse=True):
                            for d in sorted(all_elements - selected_set, reverse=True):
                                if c == d:
                                    continue
                                candidate = [x for x in selected if x != a and x != b] + [c, d]
                                if _is_valid(candidate):
                                    selected = candidate
                                    selected_set = set(selected)
                                    if len(selected) > best_size:
                                        best = list(selected)
                                        best_size = len(selected)
                                    two_swapped = True
                                    break
                            if two_swapped or time.time() > deadline:
                                break
                        if two_swapped or time.time() > deadline:
                            break
                    if two_swapped or time.time() > deadline:
                        break
                if not two_swapped:
                    break
    
    return best

def construct():
    """Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality."""
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Start with greedy construction
    initial = _greedy_construction()
    
    if not initial:
        # Fallback: just return a single element
        return [24]
    
    # Try local search optimization
    result = _local_search_optimize(initial)
    
    if time.time() > deadline:
        return result
    
    # Verify the result
    if not _is_valid(result):
        # Fallback to greedy result if it's valid
        if _is_valid(initial):
            return initial
        return [24]
    
    return result
# EVOLVE-BLOCK-END
