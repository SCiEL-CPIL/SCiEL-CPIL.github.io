import time
import random
import sys

START_TIME = time.time()
DEADLINE = 90.0

def elapsed():
    return time.time() - START_TIME

def backtrack_search(s, bitsets, candidates, target_size, deadline_margin=5.0):
    """Backtracking search with forward checking to find a set of target_size.
    
    Args:
        s: current list of selected integers
        bitsets: current cardinality-indexed bitsets
        candidates: list of remaining candidate integers to try
        target_size: desired final size
        deadline_margin: safety margin in seconds
    
    Returns:
        A valid set of size >= target_size, or None if not found within time.
    """
    n = len(s)
    if n >= target_size:
        return s[:]
    
    if elapsed() > DEADLINE - deadline_margin:
        return None
    
    # Try adding each candidate
    for i, val in enumerate(candidates):
        if elapsed() > DEADLINE - deadline_margin:
            return None
        
        # Forward check: can we add val without creating cross-cardinality collisions?
        new_bitsets = try_add(s, bitsets, val)
        if new_bitsets is not None:
            # Recursively try to complete the set
            remaining = candidates[i+1:]
            result = backtrack_search(s + [val], new_bitsets, remaining, target_size, deadline_margin)
            if result is not None:
                return result
    
    return None

def backtrack_from_seed(seed_set, target_size, deadline_margin=5.0):
    """Start backtracking from a seed set and try to reach target_size.
    
    Args:
        seed_set: initial valid set
        target_size: desired final size
        deadline_margin: safety margin in seconds
    
    Returns:
        A valid set of size >= target_size, or the seed_set if no improvement found.
    """
    seed_set = list(seed_set)
    bitsets = build_bitsets(seed_set)
    used = set(seed_set)
    
    # Generate candidates not in seed_set, sorted by "promisingness"
    # Larger values tend to be more separable
    candidates = [v for v in range(1, 257) if v not in used]
    candidates.sort(reverse=True)  # Try larger values first for better separation
    
    result = backtrack_search(seed_set, bitsets, candidates, target_size, deadline_margin)
    if result is not None:
        return result
    return seed_set

def build_bitsets(s):
    """Build cardinality-indexed bitset of subset sums.
    Returns list of ints where bitset[k] has bit j set iff sum j is achievable with k elements.
    """
    n = len(s)
    bitsets = [0] * (n + 1)
    bitsets[0] = 1  # bit 0 set: empty sum
    for val in s:
        # Update in reverse to avoid using same element twice
        for k in range(n, 0, -1):
            bitsets[k] |= (bitsets[k - 1] << val)
    return bitsets

def is_valid_bitset(bitsets):
    """Check that no two cardinality levels share a sum.
    Fast: OR all levels; if any bit is set in >=2 levels, popcount of OR < total popcount.
    """
    combined = 0
    total = 0
    for b in bitsets:
        combined |= b
        total += b.bit_count()
    return combined.bit_count() == total

def try_add(s, bitsets, val):
    """Check if adding val to s keeps it valid. Returns new bitsets if valid, else None."""
    n = len(s)
    n_new = n + 1
    new_bitsets = [0] * (n_new + 1)
    new_bitsets[0] = bitsets[0]
    for k in range(n_new, 0, -1):
        if k - 1 <= n:
            new_bitsets[k] |= (bitsets[k - 1] << val)
        if k <= n:
            new_bitsets[k] |= bitsets[k]
    
    # Check validity (fast)
    combined = 0
    total = 0
    for b in new_bitsets:
        combined |= b
        total += b.bit_count()
    if combined.bit_count() != total:
        return None
    return new_bitsets

def try_swap(s, bitsets, remove_val, add_val):
    """Check if swapping remove_val with add_val in s keeps it valid."""
    # Remove remove_val
    # This is effectively building bitsets for s without remove_val
    # Then adding add_val
    s_rem = [x for x in s if x != remove_val]
    rem_bitsets = build_bitsets(s_rem)
    
    return try_add(s_rem, rem_bitsets, add_val)

def local_search(s, deadline_margin=5.0):
    """Iterative local search to maximize set size.
    
    Each round: try to add ANY unused value (all of them, not just first).
    If no addition, try swaps (remove one, add one) to unblock future additions.
    """
    s = list(s)
    bitsets = build_bitsets(s)
    used = set(s)
    
    max_rounds = 200
    for _ in range(max_rounds):
        if elapsed() > DEADLINE - deadline_margin:
            break
        
        # Phase 1: try to add any unused value (try all, keep all that fit)
        added_any = False
        for val in range(1, 257):
            if elapsed() > DEADLINE - deadline_margin:
                break
            if val in used:
                continue
            new_bitsets = try_add(s, bitsets, val)
            if new_bitsets is not None:
                s.append(val)
                used.add(val)
                bitsets = new_bitsets
                added_any = True
        if added_any:
            continue  # re-scan for more additions
        
        # Phase 2: no addition possible; try swaps to change the set composition
        swap_done = False
        n = len(s)
        # For each element to remove, try adding each unused value
        for i in range(n):
            if elapsed() > DEADLINE - deadline_margin:
                break
            remove_val = s[i]
            s_rem = [x for j, x in enumerate(s) if j != i]
            rem_bitsets = build_bitsets(s_rem)
            
            # Try unused values in descending order (larger = more separable)
            for val in range(256, 0, -1):
                if elapsed() > DEADLINE - deadline_margin:
                    break
                if val in used:
                    continue
                new_bitsets = try_add(s_rem, rem_bitsets, val)
                if new_bitsets is not None:
                    # Perform swap
                    s[i] = val
                    used.discard(remove_val)
                    used.add(val)
                    bitsets = new_bitsets
                    swap_done = True
                    break
            if swap_done:
                break
        if not swap_done:
            break  # stuck
    return s

def greedy_construct(order, deadline_margin=5.0):
    """Greedily build a valid set by adding elements in the given order."""
    s = []
    bitsets = [1]
    for val in order:
        if elapsed() > DEADLINE - deadline_margin:
            break
        new_bitsets = try_add(s, bitsets, val)
        if new_bitsets is not None:
            s.append(val)
            bitsets = new_bitsets
    return s

def construct():
    if elapsed() > DEADLINE - 10.0:
        return list(range(226, 257)) # Fallback size 31
    
    # Initial set from greedy
    candidates = list(range(1, 257))
    
    # Try a few orders
    best = []
    best_size = 0
    
    # Descending
    s = greedy_construct(sorted(candidates, reverse=True), deadline_margin=10.0)
    if len(s) > best_size:
        best = s
        best_size = len(s)
        
    # Ascending
    s = greedy_construct(sorted(candidates), deadline_margin=10.0)
    if len(s) > best_size:
        best = s
        best_size = len(s)
        
    # Random orders
    random.seed(42)
    for _ in range(20):
        if elapsed() > DEADLINE - 20.0:
            break
        random.shuffle(candidates)
        s = greedy_construct(candidates, deadline_margin=10.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
            
    # Local search (give it more budget since it's now more effective)
    if elapsed() < DEADLINE - 12.0:
        improved = local_search(best, deadline_margin=8.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Backtracking search to try to exceed current best
    if elapsed() < DEADLINE - 12.0 and best_size < 60:
        # Try to reach best_size + 1, then +2, etc.
        for target in range(best_size + 1, min(best_size + 4, 61)):
            if elapsed() > DEADLINE - 10.0:
                break
            result = backtrack_from_seed(best, target, deadline_margin=8.0)
            if result is not None and len(result) > best_size:
                best = result
                best_size = len(best)
    
    # Final validation
    if not is_valid_bitset(build_bitsets(best)):
        return list(range(226, 257))
        
    return [int(x) for x in best]
