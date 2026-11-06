"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    import time as _time
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin: 120s limit, stop at 110s

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        # Optimization: if all elements are odd, it's sum-free.
        # Check parity first
        all_odd = all(x % 2 != 0 for x in s)
        if all_odd:
            return True
            
        # General check
        # To avoid O(N^2) in Python, we can iterate and check sums
        # But for N=240, N^2 is 57600, which is fast.
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def can_add(s, x):
        """Check if x can be added to set s without violating sum-free property."""
        # 1. x = a + b for some a, b in s
        # 2. a + x = c for some a, c in s
        # 3. x + b = c for some b, c in s (same as 2)
        
        # Check 1: is x the sum of two elements in s?
        # Iterate a in s, check if x-a is in s
        for a in s:
            if (x - a) in s:
                # Need to ensure we don't count a=a if x=2a and a is in s
                # If x == 2a, then a is in s, so x-a = a is in s. This is a valid violation.
                return False
        
        # Check 2: is x part of a sum a+x=c?
        # Iterate a in s, check if a+x is in s
        for a in s:
            if (a + x) in s:
                return False
        
        return True

    def solve_instance(elements, time_budget):
        """
        Solve a single instance.
        Strategies:
        1. All odds (guaranteed sum-free).
        2. Greedy addition of evens to odds.
        3. Local search (swap 1 out for 2 in, or add 1 if possible).
        """
        if not elements:
            return []

        # Strategy 1: Odds
        odds = [x for x in elements if x % 2 != 0]
        evens = [x for x in elements if x % 2 == 0]
        
        best_set = set(odds)
        best_size = len(best_set)
        
        # Strategy 2: Greedy add evens
        # Sort evens by some heuristic? Maybe abs value descending?
        # Let's try adding evens in a random order or sorted order.
        # Sorted by abs value might help if large numbers are less likely to be sums.
        evens_sorted = sorted(evens, key=lambda x: -abs(x))
        
        current_set = set(best_set)
        
        for e in evens_sorted:
            if _time.time() > deadline:
                break
            if can_add(current_set, e):
                current_set.add(e)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)

        # Strategy 3: Local Search (Addition)
        # Try to add any element not in best_set
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # Get list of elements not in best_set
            candidates = [x for x in elements if x not in best_set]
            for x in candidates:
                if _time.time() > deadline:
                    break
                if can_add(best_set, x):
                    best_set.add(x)
                    improved = True
                    break # Restart loop as set changed

        # Strategy 4: Local Search (Swap)
        # Try to remove one element to allow adding two others
        # This is expensive, so limit it.
        if _time.time() < deadline - 5:
            improved = True
            iterations = 0
            while improved and _time.time() < deadline - 5 and iterations < 20:
                improved = False
                iterations += 1
                # Pick a random element from best_set to remove
                if not best_set:
                    break
                # To avoid randomness overhead, iterate deterministically but limited
                # Let's try removing each element once per iteration
                for y in list(best_set):
                    if _time.time() > deadline - 5:
                        break
                    test_set = best_set - {y}
                    # Try to add elements from candidates
                    added_count = 0
                    for x in elements:
                        if x in test_set:
                            continue
                        if can_add(test_set, x):
                            test_set.add(x)
                            added_count += 1
                    if len(test_set) > len(best_set):
                        best_set = test_set
                        improved = True
                        break

        # Final verification
        if not is_sum_free(best_set):
            # Fallback to odds
            best_set = set(odds)

        return sorted(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        remaining_time = deadline - _time.time()
        if remaining_time <= 0:
            # Emergency fallback: just return odds
            results.append(sorted([x for x in elements if x % 2 != 0]))
            continue
        
        # Allocate time proportional to size? Or fixed?
        # 60: 20s, 120: 40s, 240: 50s? Total 110s.
        # Let's just use the global deadline.
        result = solve_instance(elements, remaining_time)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
