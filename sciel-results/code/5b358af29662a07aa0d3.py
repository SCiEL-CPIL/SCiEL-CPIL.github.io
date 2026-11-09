import time
import random

START_TIME = time.time()
DEADLINE = 90.0  # seconds

def elapsed():
    return time.time() - START_TIME

def build_bitsets(s):
    """Build cardinality-indexed bitset of subset sums.
    Returns list of ints where bitset[k] has bit j set iff sum j is achievable with k elements.
    """
    n = len(s)
    # Maximum possible sum
    max_sum = sum(s)
    # Use Python big ints as bitsets
    bitsets = [0] * (n + 1)
    bitsets[0] = 1  # bit 0 set: empty sum
    for val in s:
        # Update in reverse to avoid using same element twice
        for k in range(n, 0, -1):
            if k <= n:
                bitsets[k] |= (bitsets[k - 1] << val)
    return bitsets

def is_valid_bitset(bitsets):
    """Check that no two cardinality levels share a sum."""
    n = len(bitsets) - 1
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if bitsets[i] & bitsets[j]:
                return False
    return True

def greedy_construct(order, deadline_margin=5.0):
    """Greedily build a valid set by adding elements in the given order using bitsets."""
    s = []
    bitsets = [1]  # bitsets[0] = 1 (empty sum)
    n = 0
    for val in order:
        if elapsed() > DEADLINE - deadline_margin:
            break
        # Try adding val
        n_new = n + 1
        # Build new bitsets
        new_bitsets = [0] * (n_new + 1)
        new_bitsets[0] = bitsets[0]
        for k in range(n_new, 0, -1):
            if k - 1 <= n:
                new_bitsets[k] |= (bitsets[k - 1] << val)
            if k <= n:
                new_bitsets[k] |= bitsets[k]
        
        # Check validity
        valid = True
        for i in range(n_new + 1):
            for j in range(i + 1, n_new + 1):
                if new_bitsets[i] & new_bitsets[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s.append(val)
            bitsets = new_bitsets
            n = n_new
    return s

def local_search(s, deadline_margin=3.0):
    """Try to grow the set by swapping or adding elements."""
    s = list(s)
    n = len(s)
    bitsets = build_bitsets(s)
    used = set(s)
    
    # Try adding each unused element
    for val in range(1, 257):
        if elapsed() > DEADLINE - deadline_margin:
            break
        if val in used:
            continue
        # Try adding val
        n_new = n + 1
        new_bitsets = [0] * (n_new + 1)
        new_bitsets[0] = bitsets[0]
        for k in range(n_new, 0, -1):
            if k - 1 <= n:
                new_bitsets[k] |= (bitsets[k - 1] << val)
            if k <= n:
                new_bitsets[k] |= bitsets[k]
        
        valid = True
        for i in range(n_new + 1):
            for j in range(i + 1, n_new + 1):
                if new_bitsets[i] & new_bitsets[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s.append(val)
            used.add(val)
            bitsets = new_bitsets
            n = n_new
            # Restart the scan since set changed
            return local_search(s, deadline_margin)
    
    # Try swapping: remove one element, add another
    for i in range(n):
        if elapsed() > DEADLINE - deadline_margin:
            break
        removed = s[i]
        # Build bitsets without s[i]
        remaining = [x for j, x in enumerate(s) if j != i]
        n_rem = n - 1
        rem_bitsets = build_bitsets(remaining)
        
        for val in range(1, 257):
            if elapsed() > DEADLINE - deadline_margin:
                break
            if val in used or val == removed:
                continue
            # Try adding val to remaining
            n_new = n_rem + 1
            new_bitsets = [0] * (n_new + 1)
            new_bitsets[0] = rem_bitsets[0]
            for k in range(n_new, 0, -1):
                if k - 1 <= n_rem:
                    new_bitsets[k] |= (rem_bitsets[k - 1] << val)
                if k <= n_rem:
                    new_bitsets[k] |= rem_bitsets[k]
            
            valid = True
            for a in range(n_new + 1):
                for b in range(a + 1, n_new + 1):
                    if new_bitsets[a] & new_bitsets[b]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Perform swap
                s[i] = val
                used.discard(removed)
                used.add(val)
                bitsets = new_bitsets
                # Continue searching for more improvements
                return local_search(s, deadline_margin)
    
    return s

def construct():
    if elapsed() > DEADLINE - 5.0:
        # Fallback to known valid set
        return list(range(226, 257))
    
    best = list(range(226, 257))  # Known valid size-31 set
    best_size = 31
    
    candidates = list(range(1, 257))
    
    # Try descending order
    if elapsed() < DEADLINE - 15.0:
        s = greedy_construct(sorted(candidates, reverse=True), deadline_margin=10.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Try ascending order
    if elapsed() < DEADLINE - 15.0:
        s = greedy_construct(sorted(candidates), deadline_margin=10.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Try several random orderings
    random.seed(42)
    for trial in range(50):
        if elapsed() > DEADLINE - 15.0:
            break
        random_order = candidates[:]
        random.shuffle(random_order)
        s = greedy_construct(random_order, deadline_margin=10.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Apply local search to the best set
    if elapsed() < DEADLINE - 8.0:
        improved = local_search(best, deadline_margin=5.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Final check: ensure best is valid
    bitsets = build_bitsets(best)
    if not is_valid_bitset(bitsets):
        return list(range(226, 257))
    
    # Ensure all elements are Python ints in a list
    return [int(x) for x in best]
