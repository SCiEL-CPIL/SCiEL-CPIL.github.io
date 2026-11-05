from fractions import Fraction
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # We want the largest subset of [1,24] that is reciprocal-subsum-free
    # For each element x in the set, 1/x must not equal sum of 1/y for any non-empty subset of other elements
    
    # Key insight: larger elements are "safer" because their reciprocals are smaller
    # and hard to match with sums of other reciprocals
    
    # Strategy: try all subsets in descending order of size, but that's 2^24 which is too large
    # Better: use a greedy approach or smart search
    
    # Let's use a backtracking approach: try to include elements from 24 down to 1
    # At each step, check if adding the element violates the constraint
    
    # First, let's define a helper to check if a set is valid
    def is_valid_subset(subset):
        """Check if subset is reciprocal-subsum-free."""
        if len(subset) <= 1:
            return True
        for x in subset:
            others = [y for y in subset if y != x]
            target = Fraction(1, x)
            # Check if any non-empty subset of others sums to target
            # Use meet-in-the-middle for efficiency
            n = len(others)
            if n == 0:
                continue
            # For small n, brute force
            if n <= 16:
                # Generate all subset sums
                found = False
                for mask in range(1, 1 << n):
                    s = Fraction(0, 1)
                    for i in range(n):
                        if mask & (1 << i):
                            s += Fraction(1, others[i])
                            if s >= target:
                                break
                    if s == target:
                        return False
                # If we get here, x is safe
            else:
                # Meet-in-the-middle
                mid = n // 2
                left = others[:mid]
                right = others[mid:]
                
                # Generate all subset sums for left half
                left_sums = set()
                for mask in range(1 << len(left)):
                    s = Fraction(0, 1)
                    for i in range(len(left)):
                        if mask & (1 << i):
                            s += Fraction(1, left[i])
                    if s <= target:
                        left_sums.add(s)
                
                # Generate all subset sums for right half
                right_sums = set()
                for mask in range(1 << len(right)):
                    s = Fraction(0, 1)
                    for i in range(len(right)):
                        if mask & (1 << i):
                            s += Fraction(1, right[i])
                    if s <= target:
                        right_sums.add(s)
                
                # Check if any left_sum + right_sum == target
                # where at least one is non-zero (non-empty subset)
                for ls in left_sums:
                    needed = target - ls
                    if needed in right_sums:
                        # Need to ensure the combined subset is non-empty
                        # If ls == 0 and needed == 0, that's the empty subset, skip
                        if ls == Fraction(0,1) and needed == Fraction(0,1):
                            continue
                        return False
        
        return True
    
    # Greedy approach: try to include elements from 24 down to 1
    # This should work well because larger elements are safer
    
    # Try different orderings and strategies
    best_set = set()
    
    # Strategy 1: Greedy from 24 down to 1
    candidate = set()
    for x in range(24, 0, -1):
        new_candidate = candidate | {x}
        if is_valid_subset(new_candidate):
            candidate = new_candidate
        if time.time() > deadline:
            break
    
    best_set = candidate
    
    # Strategy 2: Try to add back some excluded elements
    # Elements that were excluded might be addable if we remove some others
    excluded = set(range(1, 25)) - candidate
    for x in sorted(excluded, reverse=True):
        new_candidate = candidate | {x}
        if is_valid_subset(new_candidate):
            candidate = new_candidate
        if time.time() > deadline:
            break
    
    best_set = candidate
    
    # Strategy 3: Try swapping - remove one element, add another
    # This is more expensive, so only do if we have time
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_list = sorted(best_set)
            for i, x in enumerate(current_list):
                # Try removing x and adding each excluded element
                for y in sorted(excluded, reverse=True):
                    if y in best_set:
                        continue
                    new_set = best_set - {x} | {y}
                    if len(new_set) > len(best_set) and is_valid_subset(new_set):
                        best_set = new_set
                        excluded = set(range(1, 25)) - best_set
                        improved = True
                        break
                if improved:
                    break
    
    # Ensure non-empty
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)
