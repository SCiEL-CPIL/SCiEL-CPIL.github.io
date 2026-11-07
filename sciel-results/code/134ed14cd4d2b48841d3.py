"""Backtracking search with bitmask-based subset sums and divisibility pruning."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # Max possible sum: 30 * 256 = 7680
    MAX_SUM = 7680
    
    # Precompute for each possible sum s (1..MAX_SUM), a bitmask of all multiples of s up to MAX_SUM
    # mask[s] has bit k set iff k is a multiple of s (k >= 2s, since s itself is allowed)
    # We'll use this to check: if current_sums_mask & mask[s] != 0, then s divides some other sum
    # Actually, we need: for any sum s in our set, no other sum t in our set should be a multiple of s.
    # So for each s, the forbidden sums are 2s, 3s, ... up to MAX_SUM.
    
    # Precompute forbidden_mask[s]: bitmask where bit k is set iff k is a multiple of s and k > s
    forbidden_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        m = 0
        for k in range(2 * s, MAX_SUM + 1, s):
            m |= (1 << k)
        forbidden_mask[s] = m
    
    # Also precompute: for a given sum value s, if we add it, we need to check that no existing sum divides s
    # and s doesn't divide any existing sum.
    # For "no existing sum divides s": for each d | s (d < s), if d is in current sums, invalid.
    # We can precompute divisors_mask[s]: bitmask of all proper divisors of s
    divisors_mask = [0] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        m = 0
        for d in range(1, int(math.isqrt(s)) + 1):
            if s % d == 0:
                if d < s:
                    m |= (1 << d)
                other = s // d
                if other != d and other < s:
                    m |= (1 << other)
        divisors_mask[s] = m
    
    # For checking if adding a new sum ns is valid given current_sums_mask:
    # 1. No existing sum divides ns: (current_sums_mask & divisors_mask[ns]) == 0
    # 2. ns doesn't divide any existing sum: (current_sums_mask & forbidden_mask[ns]) == 0
    # 3. ns itself is not already in current_sums_mask (but we're building sums, so ns is new)
    
    def check_new_sum_valid(ns, current_mask):
        """Check if adding sum ns to current_mask is valid."""
        if ns > MAX_SUM:
            return False
        if current_mask & (1 << ns):
            return False  # already present
        # Check no existing sum divides ns
        if current_mask & divisors_mask[ns]:
            return False
        # Check ns doesn't divide any existing sum
        if current_mask & forbidden_mask[ns]:
            return False
        return True
    
    # For adding element x to current set:
    # New sums = {x} union {s + x for s in current_sums}
    # We need to check all new sums are valid against the expanded set.
    
    def can_add_element(x, current_mask, current_total):
        """Check if element x can be added. Returns (bool, new_mask) if valid."""
        new_total = current_total + x
        if new_total > MAX_SUM:
            return False, current_mask
        
        # Generate all new sums
        # Current sums are bits set in current_mask
        # New sums: x, and s+x for each s in current_mask
        new_sums = set()
        new_sums.add(x)
        # Iterate over set bits in current_mask
        m = current_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            s = lsb.bit_length() - 1
            ns = s + x
            if ns <= MAX_SUM:
                new_sums.add(ns)
            m ^= lsb
        
        # Check each new sum against the combined set (current_mask | new_sums)
        # But we need to check pairwise among new_sums too.
        # Build the full new mask
        new_mask = current_mask
        for ns in new_sums:
            new_mask |= (1 << ns)
        
        # Now check: for each sum in new_sums, verify it's valid in new_mask
        # Actually, we need to check that for every pair of sums in new_mask, one doesn't divide the other.
        # Efficient check: for each s in new_sums, check (new_mask & forbidden_mask[s]) == 0 and (new_mask & divisors_mask[s]) == 0
        # But this is O(|new_sums|) per element, which could be large.
        
        # Alternative: check each new sum against (current_mask | other_new_sums)
        # Simpler: just check all new sums against the full new_mask
        for ns in new_sums:
            if ns > MAX_SUM:
                return False, current_mask
            # Check no other sum in new_mask is a multiple of ns
            if new_mask & forbidden_mask[ns]:
                return False, current_mask
            # Check no other sum in new_mask divides ns
            if new_mask & divisors_mask[ns]:
                return False, current_mask
        
        return True, new_mask
    
    # Backtracking search
    best_set = []
    best_size = 0
    
    pool = list(range(1, 257))
    
    # Use a recursive backtracking with ordering
    # To make it fast, we'll try elements in various orders and use time limits
    
    def backtrack(current_set, current_mask, current_total, index, depth, deadline):
        nonlocal best_set, best_size, start_time
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if depth >= 30:
            return
        
        # Try adding elements from pool[index:]
        for i in range(index, len(pool)):
            if time.time() > deadline:
                return
            
            x = pool[i]
            ok, new_mask = can_add_element(x, current_mask, current_total)
            if ok:
                current_set.append(x)
                backtrack(current_set, new_mask, current_total + x, i + 1, depth + 1, deadline)
                current_set.pop()
    
    # Run multiple backtracking attempts with different orderings
    strategies = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        [x for x in range(1, 257) if all(x % d != 0 for d in range(2, int(math.isqrt(x)) + 1))] + 
        [x for x in range(1, 257) if any(x % d == 0 for d in range(2, int(math.isqrt(x)) + 1))],
    ]
    
    # Add some randomized orderings
    rng = random.Random(42)
    for _ in range(5):
        s = list(range(1, 257))
        rng.shuffle(s)
        strategies.append(s)
    
    for strat_idx, order in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Use a shorter deadline for each strategy to allow multiple attempts
        strat_deadline = min(deadline, start_time + 15.0)
        
        current_set = []
        backtrack(current_set, 0, 0, 0, 0, strat_deadline)
        
        if best_size >= 30:
            break
    
    # Final validation
    if best_set:
        # Validate the best set
        sums = set()
        for x in best_set:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums.update(new_sums)
            sums.add(x)
        
        # Check divisibility
        valid = True
        sums_list = sorted(sums)
        sums_set = sums
        total = sum(best_set)
        for s in sums_list:
            if s == 0:
                continue
            multiple = 2 * s
            while multiple <= total:
                if multiple in sums_set:
                    valid = False
                    break
                multiple += s
            if not valid:
                break
        
        if not valid:
            # Fallback: return empty or a small valid set
            best_set = [1]  # Single element is always valid
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
