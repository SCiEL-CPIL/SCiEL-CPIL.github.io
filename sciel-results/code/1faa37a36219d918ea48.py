"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

MAX_SUM = 256 * 30  # 7680
MAX_ELEMENTS = 30

# Precompute: for each possible sum value, which other sums does it divide?
# divisors_mask[s] = bitmask of all t > s such that s divides t
# This lets us quickly check if adding a new sum s would conflict with existing sums

@lru_cache(maxsize=None)
def _divisible_mask(s):
    """Return bitmask of all multiples of s up to MAX_SUM (exclusive of s itself)."""
    mask = 0
    for k in range(2, MAX_SUM // s + 1):
        t = s * k
        if t <= MAX_SUM:
            mask |= (1 << t)
    return mask

def _check_divisibility(sums_mask):
    """Check if sums_mask (bitmask of subset sums) has no divisibility pairs."""
    # For each set bit s, check if any multiple of s is also set
    m = sums_mask
    while m:
        # Get lowest set bit
        low = m & (-m)
        s = low.bit_length() - 1
        # Check if any multiple of s (other than s) is in the mask
        mult_mask = _divisible_mask(s)
        if (sums_mask & mult_mask) != 0:
            return False
        m ^= low
    return True

def _add_element(sums_mask, elem):
    """Add elem to the set represented by sums_mask. Return new mask or None if invalid."""
    # New sums: existing sums, elem, and existing sums + elem
    new_mask = sums_mask | (1 << elem) | ((sums_mask << elem))
    # Check divisibility
    if not _check_divisibility(new_mask):
        return None
    return new_mask

def _backtrack_search(candidates, current_sums, current_set, deadline, best_ref):
    """
    Backtracking search to maximize set size.
    candidates: list of remaining candidate values (sorted)
    current_sums: bitmask of current subset sums
    current_set: list of currently selected elements
    deadline: time limit
    best_ref: [best_size, best_set] mutable reference
    """
    if time.time() > deadline:
        return
    
    # Prune: can't beat best
    if len(current_set) + len(candidates) <= best_ref[0]:
        return
    
    if not candidates:
        if len(current_set) > best_ref[0]:
            best_ref[0] = len(current_set)
            best_ref[1] = list(current_set)
        return
    
    # Try including candidates in order of "least restrictive" first
    # Sort by how many conflicts they cause with current sums
    def conflict_score(val):
        new_mask = _add_element(current_sums, val)
        if new_mask is None:
            return -1
        return bin(new_mask).count('1')
    
    # Filter candidates that can be added
    viable = []
    for val in candidates:
        new_mask = _add_element(current_sums, val)
        if new_mask is not None:
            viable.append((val, new_mask))
    
    # Sort by number of resulting sums (fewer = more restrictive = try later)
    # Actually, we want to try less restrictive ones first to find large sets
    viable.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)
    
    for val, new_mask in viable:
        if time.time() > deadline:
            return
        
        current_set.append(val)
        remaining = [c for c in candidates if c > val]  # Keep sorted order
        _backtrack_search(remaining, new_mask, current_set, deadline, best_ref)
        current_set.pop()
        
        if len(current_set) + len(remaining) <= best_ref[0]:
            continue
    
    # Also try skipping some candidates - handled by the recursion above

def _greedy_with_backtrack(deadline):
    """
    Hybrid approach: greedily build a base set, then use backtracking to extend.
    """
    best = [0, []]
    
    # Start with some promising base sets
    base_sets = []
    
    # Base 1: Powers of 2 (always valid for subset sums - all sums distinct)
    base1 = [2**i for i in range(1, 9) if 2**i <= 256]  # 2,4,8,16,32,64,128
    base_sets.append(base1)
    
    # Base 2: Large primes (minimize divisibility)
    primes = [p for p in range(100, 257) if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    base_sets.append(primes[:5])
    
    # Base 3: Mix of large numbers
    base_sets.append([128, 192, 224, 240, 252])
    
    for base in base_sets:
        if time.time() > deadline:
            break
        
        # Verify base is valid
        sums_mask = 0
        valid = True
        for elem in base:
            new_mask = _add_element(sums_mask, elem)
            if new_mask is None:
                valid = False
                break
            sums_mask = new_mask
        
        if not valid:
            continue
        
        # Get all candidates that can be added
        candidates = [v for v in range(1, 257) if v not in base]
        
        # Try to extend with backtracking
        current_set = list(base)
        _backtrack_search(candidates, sums_mask, current_set, deadline, best)
    
    return best[1] if best[1] else []

def _local_improve(sums_mask, current_set, deadline):
    """Try to improve by swapping elements."""
    best = [len(current_set), list(current_set)]
    
    for i in range(len(current_set)):
        if time.time() > deadline:
            break
        
        # Remove element i
        reduced = current_set[:i] + current_set[i+1:]
        
        # Recompute sums for reduced set
        reduced_mask = 0
        valid = True
        for elem in reduced:
            new_mask = _add_element(reduced_mask, elem)
            if new_mask is None:
                valid = False
                break
            reduced_mask = new_mask
        
        if not valid:
            continue
        
        # Try adding different elements
        for new_val in range(1, 257):
            if new_val in reduced:
                continue
            
            new_mask = _add_element(reduced_mask, new_val)
            if new_mask is None:
                continue
            
            new_set = reduced + [new_val]
            
            # Try to extend this further
            candidates = [v for v in range(1, 257) if v not in new_set and v > max(new_set)]
            current_set2 = list(new_set)
            
            # Quick greedy extension
            temp_best = [len(new_set), list(new_set)]
            _backtrack_search(candidates, new_mask, current_set2, min(deadline, time.time() + 5), temp_best)
            
            if temp_best[0] > best[0]:
                best[0] = temp_best[0]
                best[1] = temp_best[1]
    
    return best[1]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Phase 1: Greedy with backtracking from multiple bases
    best_set = _greedy_with_backtrack(deadline - 10)
    
    if not best_set:
        # Fallback: simple valid set
        best_set = [2, 4, 8, 16, 32, 64, 128]
    
    # Phase 2: Local improvement
    if time.time() < deadline - 5:
        # Compute sums for current best
        sums_mask = 0
        for elem in best_set:
            new_mask = _add_element(sums_mask, elem)
            if new_mask is None:
                break
            sums_mask = new_mask
        
        improved = _local_improve(sums_mask, best_set, deadline - 2)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Phase 3: Random restarts if time remains
    if time.time() < deadline - 3:
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            
            random.seed(attempt * 104729 + 7)
            
            # Random base
            base_size = random.randint(3, 6)
            base = random.sample(range(50, 257), base_size)
            base.sort()
            
            # Verify
            sums_mask = 0
            valid = True
            for elem in base:
                new_mask = _add_element(sums_mask, elem)
                if new_mask is None:
                    valid = False
                    break
                sums_mask = new_mask
            
            if not valid:
                continue
            
            candidates = [v for v in range(1, 257) if v not in base]
            current_set = list(base)
            best_ref = [len(base), list(base)]
            _backtrack_search(candidates, sums_mask, current_set, deadline - 2, best_ref)
            
            if best_ref[0] > len(best_set):
                best_set = best_ref[1]
    
    # Ensure valid Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
