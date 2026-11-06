"""Reciprocal-subsum-free set construction."""
import time
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 80.0  # 10s safety margin before 90s kill

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _subset_sum_exists(values, target):
    """Check if any non-empty subset of values sums to exactly target (Fraction).
    Uses meet-in-the-middle for efficiency."""
    n = len(values)
    if n == 0:
        return False
    if n <= 20:
        # Simple DFS with pruning by checking all subsets
        # For n up to ~20, 2^20 = ~1M which is fast
        from itertools import combinations
        for r in range(1, n + 1):
            for combo in combinations(values, r):
                if sum(combo) == target:
                    return True
            if _deadline_hit():
                return False
        return False
    else:
        # Meet in the middle
        mid = n // 2
        left = values[:mid]
        right = values[mid:]
        
        # Generate all subset sums for left half
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # Generate all subset sums for right half and check
        for mask in range(1 << len(right)):
            s = Fraction(0)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if mask == 0:
                # Only consider non-empty subsets total
                if s in left_sums and s != 0:
                    # This would mean empty right + non-empty left
                    # We need to check if the combined subset is non-empty
                    # Since mask==0, the left part must be non-empty
                    # left_sums contains 0 from empty left, so if s!=0 and s in left_sums,
                    # there's a non-empty left subset summing to s
                    return True
            else:
                # mask != 0, so right part is non-empty
                # Check if target - s is in left_sums (including empty left = 0)
                if (target - s) in left_sums:
                    return True
            if _deadline_hit():
                return False
        return False

def _is_valid(S):
    """Check if S is reciprocal-subsum-free.
    For each element a in S, 1/a should NOT equal sum of 1/b for any non-empty
    subset of S \\ {a}."""
    S_set = set(S)
    for a in S:
        target = Fraction(1, a)
        others = [Fraction(1, b) for b in S if b != a]
        if _subset_sum_exists(others, target):
            return False
    return True

def _greedy_construct():
    """Build a valid set greedily, adding elements that maintain validity.
    Try different orderings to find a good set."""
    best = None
    best_size = 0
    
    # Try adding elements in different orders
    base = list(range(1, 25))
    
    # Order 1: descending (larger elements first, less likely to cause issues)
    for order in [base[::-1], base, list(reversed(base))]:
        if _deadline_hit():
            break
        current = []
        for elem in order:
            if _deadline_hit():
                break
            candidate = current + [elem]
            if _is_valid(candidate):
                current = candidate
        if len(current) > best_size:
            best_size = len(current)
            best = current[:]
    
    return best if best else [1]

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Start with a simple valid set
    best = _greedy_construct()
    best_size = len(best) if best else 1
    
    # Try to improve with local search
    import random
    
    # Convert to bitmask for easier manipulation
    # Bit i represents element i+1
    def mask_to_set(m):
        return [i + 1 for i in range(24) if (m >> i) & 1]
    
    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    current_mask = set_to_mask(best)
    
    # Local search: try adding or removing single elements
    no_improve = 0
    max_no_improve = 5000
    
    while not _deadline_hit() and no_improve < max_no_improve:
        improved = False
        # Try adding random elements
        for _ in range(10):
            if _deadline_hit():
                break
            elem = random.randint(1, 24)
            bit = 1 << (elem - 1)
            if current_mask & bit:
                continue
            new_mask = current_mask | bit
            new_set = mask_to_set(new_mask)
            if _is_valid(new_set):
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best = new_set[:]
                    current_mask = new_mask
                    improved = True
                    no_improve = 0
                elif len(new_set) == len(current_mask).bit_count():
                    # Same size, accept to explore
                    if random.random() < 0.3:
                        current_mask = new_mask
                    no_improve = 0
                else:
                    no_improve += 1
        
        # Try removing random elements (to escape local optima)
        if _deadline_hit():
            break
        elems_in_set = [i + 1 for i in range(24) if (current_mask >> i) & 1]
        if elems_in_set:
            elem = random.choice(elems_in_set)
            bit = 1 << (elem - 1)
            new_mask = current_mask & ~bit
            if new_mask != 0:
                new_set = mask_to_set(new_mask)
                if _is_valid(new_set):
                    if random.random() < 0.5:  # Sometimes accept worse to explore
                        current_mask = new_mask
                    no_improve = 0
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    # Return the best set found
    if best is None or not best:
        return [1]
    return best

# Quick verification of initial behavior
if __name__ == "__main__":
    result = construct()
    print(result)
    print(f"Size: {len(result)}")
