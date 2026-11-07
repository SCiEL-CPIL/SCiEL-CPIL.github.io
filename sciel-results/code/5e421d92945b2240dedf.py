import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 86.0  # safety margin before 90s kill

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if len(sums) < 2:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        sums_set = sums
        for u in sum_list:
            if u * 2 > max_sum:
                break
            for v in range(u * 2, max_sum + 1, u):
                if v in sums_set:
                    return False
        return True

    def try_add_element(current_sums, c):
        """Try adding element c. Returns new_sums if valid, else None."""
        # Quick check: c must not divide or be divided by any existing sum
        # Also c+c = 2c must not be in sums (since c is a new sum, 2c could be existing)
        # Actually we need to check all new sums against all old sums and among themselves
        
        # New sums are: {c} U {s+c for s in current_sums}
        # We need to check:
        # 1. c doesn't divide any existing sum and no existing sum divides c
        # 2. For each new sum s+c, it doesn't divide any existing sum and vice versa
        # 3. Among new sums, no two divide each other
        
        # First, check c against all existing sums
        for s in current_sums:
            if s % c == 0 or c % s == 0:
                return None
        
        # Build new sums
        new_sums = set(current_sums)
        new_sums.add(c)
        for s in current_sums:
            new_sums.add(s + c)
        
        # Check for duplicate sums (shouldn't happen with distinct positive ints, but be safe)
        if len(new_sums) != 2 * len(current_sums) + 1:
            return None
        
        # Check divisibility among new sums
        # We only need to check new sums against all sums
        new_only = {c} | {s + c for s in current_sums}
        
        # Check new sums against old sums
        for ns in new_only:
            for s in current_sums:
                if ns % s == 0 or s % ns == 0:
                    return None
        
        # Check new sums against each other
        new_list = sorted(new_only)
        max_new = new_list[-1]
        for i, u in enumerate(new_list):
            if u * 2 > max_new:
                break
            for v in range(u * 2, max_new + 1, u):
                if v in new_only:
                    return None
        
        return new_sums

    def compute_sums(elements):
        """Compute all non-empty subset sums."""
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    best_set = []
    best_size = 0

    # Strategy 1: Deterministic greedy - large numbers first
    # Large numbers have fewer multiples in range, reducing divisibility conflicts
    candidates = list(range(256, 0, -1))
    current = []
    current_sums = set()
    for c in candidates:
        if len(current) >= 30:
            break
        new_sums = try_add_element(current_sums, c)
        if new_sums is not None:
            current.append(c)
            current_sums = new_sums
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]

    # Strategy 2: Randomized greedy with various orderings
    random.seed(42)
    for attempt in range(3000):
        if time.time() > deadline - 5.0:
            break
        
        # Create candidate list with some randomization
        if attempt % 3 == 0:
            # Large numbers first
            cand = list(range(256, 127, -1))
            random.shuffle(cand)
            cand.extend(random.sample(range(1, 128), 60))
        elif attempt % 3 == 1:
            # Mixed
            cand = list(range(256, 0, -1))
            random.shuffle(cand)
        else:
            # Biased toward larger
            cand = list(range(256, 63, -1))
            random.shuffle(cand)
            cand.extend(random.sample(range(1, 64), 30))
        
        current = []
        current_sums = set()
        for c in cand:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 3: Local search - try to improve best set by swaps
    if best_size > 0 and time.time() < deadline - 8.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            base_set = best_set[:]
            base_sums = compute_sums(base_set)
            
            # Try removing each element and adding a better one
            for rem_idx in range(len(base_set)):
                if time.time() > deadline - 5.0:
                    break
                
                reduced_set = base_set[:rem_idx] + base_set[rem_idx+1:]
                red_sums = compute_sums(reduced_set)
                
                # Try adding each candidate not in reduced_set
                found = False
                for c in range(256, 0, -1):
                    if c in reduced_set:
                        continue
                    new_sums = try_add_element(red_sums, c)
                    if new_sums is not None:
                        new_set = reduced_set + [c]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set[:]
                            improved = True
                        found = True
                        break
                
                if not found:
                    # Try to extend: add an element without removing
                    pass
            
            # Try to extend the current best set
            if time.time() > deadline - 5.0:
                break
            base_set = best_set[:]
            base_sums = compute_sums(base_set)
            for c in range(256, 0, -1):
                if c in base_set:
                    continue
                new_sums = try_add_element(base_sums, c)
                if new_sums is not None:
                    best_set = base_set + [c]
                    best_size = len(best_set)
                    improved = True
                    break

    # Final validation
    if best_set:
        final_sums = compute_sums(best_set)
        if not is_valid(final_sums):
            # Fallback to a known valid small set
            best_set = [128, 192, 224]
            best_size = 3

    return [int(x) for x in best_set]
