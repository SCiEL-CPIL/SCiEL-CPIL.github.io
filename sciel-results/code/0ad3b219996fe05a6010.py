import time
import sys
import random

sys.setrecursionlimit(5000)

_DEADLINE = 85.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_no_divisibility(sums):
    """
    Check if no two distinct values in sums divide one another.
    sums is a set of ints.
    """
    if len(sums) <= 1:
        return True
    
    max_sum = max(sums)
    # For each s, check if any multiple of s is in sums
    for s in sums:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sums:
                return False
            multiple += s
    return True

def _add_element(sums, x):
    """
    Add element x to the set of subset sums.
    Returns new set of sums.
    """
    new_sums = set(sums)
    new_sums.add(x)
    for s in sums:
        new_sums.add(s + x)
    return new_sums

def _remove_element(sums, x, elements):
    """
    Remove element x from the set. We need to reconstruct sums from remaining elements.
    This is expensive, so we only do it in local search.
    """
    remaining = [e for e in elements if e != x]
    new_sums = set()
    # Reconstruct subset sums from scratch
    # This is O(2^n) which is too slow for large n, but n is small (<=30)
    # Actually, for n=30, 2^30 is 1e9, which is too slow.
    # We need a smarter way or accept the cost only when n is small.
    # In local search, we usually keep n around the best found.
    # If best found is ~15, 2^15 = 32768, which is fine.
    # If best found is ~20, 2^20 = 1e6, which is okay for a few moves.
    
    # Let's do it efficiently using a list of sums
    sums_list = [0]
    for e in remaining:
        sums_list = sums_list + [s + e for s in sums_list]
    # Remove 0 (empty subset)
    sums_list.remove(0)
    return set(sums_list)

def _validate_full(elements):
    """
    Full validation of a set of elements.
    Returns True if valid.
    """
    if not elements:
        return True
    
    sums = set()
    sums_list = [0]
    for e in elements:
        sums_list = sums_list + [s + e for s in sums_list]
    
    sums = set(sums_list)
    sums.discard(0)
    
    # Check for duplicates in subset sums?
    # The problem says "Equal sum values are counted once. Two different subsets with the same sum still make the set invalid".
    # Wait, the problem statement says: "Form every NONEMPTY subset sum; no two distinct resulting values may divide one another. Equal sum values are counted once. Two different subsets with the same sum still make the set invalid: removing their common elements leaves two disjoint subsets with the same sum s, and their union has sum 2s, which s divides."
    # This implies that if there are two different subsets with the same sum, it's invalid.
    # However, the condition "no two distinct resulting values may divide one another" is on the *values*.
    # If two subsets have the same sum s, then s is in the set of values.
    # The issue is that if subset A and subset B have the same sum s, then A \ B and B \ A are disjoint non-empty subsets with sum s (if A != B).
    # Then the union A ∪ B has sum 2s.
    # s divides 2s.
    # So the divisibility constraint on the set of *values* actually catches this?
    # No, the set of values contains s and 2s.
    # If s and 2s are both in the set of subset sums, then s divides 2s, so it's invalid.
    # So checking the divisibility of the set of *values* is sufficient?
    # Let's re-read: "no two distinct resulting values may divide one another".
    # If there is a collision (two subsets same sum s), then s is a value.
    # Is 2s necessarily a value?
    # The union of the two subsets has sum 2s. So yes, 2s is a subset sum.
    # So if there is a collision, s and 2s are both subset sums.
    # Since s divides 2s, the divisibility check on the set of values will fail.
    # So we just need to check the divisibility of the set of unique subset sums.
    
    return _check_no_divisibility(sums)

def _greedy_random(candidates, rng):
    """
    Greedy construction with random ordering.
    """
    current_sums = set()
    current_elements = []
    remaining = list(candidates)
    rng.shuffle(remaining)
    
    for x in remaining:
        # Quick check: x must not divide any s in sums, and no s in sums must divide x
        compatible = True
        for s in current_sums:
            if x % s == 0 or s % x == 0:
                compatible = False
                break
        if not compatible:
            continue
        
        new_sums = _add_element(current_sums, x)
        if _check_no_divisibility(new_sums):
            current_sums = new_sums
            current_elements.append(x)
            
    return current_elements

def _local_search(elements, candidates, rng, time_limit=2.0):
    """
    Local search to improve the set.
    Tries to swap out an element for a better one, or add an element.
    """
    if not elements:
        return elements
    
    best_elements = list(elements)
    best_sums = set()
    sums_list = [0]
    for e in best_elements:
        sums_list = sums_list + [s + e for s in sums_list]
    best_sums = set(sums_list)
    best_sums.discard(0)
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try swapping
        for i in range(len(best_elements)):
            if time.time() - start_time > time_limit:
                break
            old_x = best_elements[i]
            
            # Try replacing old_x with a new candidate
            for new_x in candidates:
                if new_x in best_elements:
                    continue
                
                # Quick check: new_x must not divide any s in (best_sums - old_x contributions)?
                # This is hard to check quickly without removing old_x first.
                # Let's just do the full check for the new set.
                
                new_elements = best_elements[:i] + [new_x] + best_elements[i+1:]
                
                # Check validity
                sums_list = [0]
                for e in new_elements:
                    sums_list = sums_list + [s + e for s in sums_list]
                new_sums = set(sums_list)
                new_sums.discard(0)
                
                if _check_no_divisibility(new_sums):
                    # Accept the move
                    best_elements = new_elements
                    best_sums = new_sums
                    improved = True
                    break # Restart search with new best
        
        # Try adding an element (if size < 30)
        if len(best_elements) < 30:
            for new_x in candidates:
                if new_x in best_elements:
                    continue
                if time.time() - start_time > time_limit:
                    break
                
                new_elements = best_elements + [new_x]
                
                sums_list = [0]
                for e in new_elements:
                    sums_list = sums_list + [s + e for s in sums_list]
                new_sums = set(sums_list)
                new_sums.discard(0)
                
                if _check_no_divisibility(new_sums):
                    best_elements = new_elements
                    best_sums = new_sums
                    improved = True
                    break

    return best_elements

def construct():
    global _START
    _START = time.time()
    
    candidates = list(range(1, 257))
    best_set = []
    best_len = 0
    
    # Multi-phase search
    # Phase 1: Several random greedy runs
    rng = random.Random(42)
    
    # Bias towards larger numbers? Or random?
    # Large numbers are less likely to divide small sums.
    # But small numbers might be useful for packing?
    # Let's try a mix.
    
    for i in range(50):
        if _remaining() < 5.0:
            break
        
        # Randomize candidates
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        
        # Bias: maybe put larger numbers first in some runs?
        if i % 2 == 0:
            shuffled.sort(reverse=True)
            # Shuffle a bit
            for j in range(len(shuffled)):
                k = rng.randint(j, len(shuffled)-1)
                shuffled[j], shuffled[k] = shuffled[k], shuffled[j]
        
        current = _greedy_random(shuffled, rng)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = current
            
        # Local search
        if len(current) >= best_len - 2: # Only search if close to best
            improved = _local_search(current, candidates, rng, time_limit=1.0)
            if len(improved) > best_len:
                best_len = len(improved)
                best_set = improved
        
        if _remaining() < 10.0:
            break
            
    # Phase 2: More local search on the best set found
    if best_set and _remaining() > 5.0:
        improved = _local_search(best_set, candidates, rng, time_limit=min(10.0, _remaining() - 5.0))
        if len(improved) > best_len:
            best_len = len(improved)
            best_set = improved

    # Final validation
    if best_set:
        if not _validate_full(best_set):
            # Fallback to a simple valid set if something went wrong
            best_set = [256]
        # Ensure distinct and in range
        best_set = [int(x) for x in best_set if 1 <= x <= 256]
        # Ensure distinct
        seen = set()
        unique = []
        for x in best_set:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        best_set = unique
        
    return best_set

