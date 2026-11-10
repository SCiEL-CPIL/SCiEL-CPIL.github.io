import time
import itertools
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Helper: check if a set of integers has all distinct reciprocal subset sums
    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        # Use Fraction for exact arithmetic
        reciprocals = [Fraction(1, e) for e in elements]
        seen = {Fraction(0, 1)}
        # Build subset sums iteratively
        sums = [Fraction(0, 1)]
        for r in reciprocals:
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True
    
    # Faster check: maintain the set of sums incrementally
    def check_addable(current_sums, seen, element):
        """Check if adding 1/element to the current set maintains dissociation."""
        r = Fraction(1, element)
        new_sums = []
        for s in current_sums:
            ns = s + r
            if ns in seen:
                return None
            new_sums.append(ns)
        return new_sums
    
    all_elements = list(range(1, 23))
    best_set = []
    best_sums = None
    best_seen = None
    
    # Multiple greedy strategies with different orderings
    # Strategy 1: Largest first (smallest reciprocals, "cheapest")
    # Strategy 2: Smallest first (largest reciprocals, "most restrictive")
    # Strategy 3: Alternating
    # Strategy 4: Random-ish orderings
    # Strategy 5: Prime-heavy ordering
    # Strategy 6: Co-prime pairs first
    
    rng = random.Random(42)
    
    # Generate diverse orderings
    base_orders = [
        all_elements[::-1],  # largest first
        all_elements,        # smallest first
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        [11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1, 12],
        sorted(all_elements, key=lambda x: -x),
        sorted(all_elements, key=lambda x: x),
        # Prime-heavy: primes first
        [2, 3, 5, 7, 11, 13, 17, 19, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        # Co-prime pairs: elements with large gcd with fewer others first
        sorted(all_elements, key=lambda x: -sum(1 for y in all_elements if x != y and __import__('math').gcd(x, y) == 1)),
    ]
    
    # Add some randomized orderings
    for _ in range(8):
        shuffled = all_elements[:]
        rng.shuffle(shuffled)
        base_orders.append(shuffled)
    
    orders = base_orders
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_sums = [Fraction(0, 1)]
        current_seen = {Fraction(0, 1)}
        
        for e in order:
            if time.time() > deadline:
                break
            new_sums = check_addable(current_sums, current_seen, e)
            if new_sums is not None:
                current.append(e)
                current_sums.extend(new_sums)
                current_seen.update(new_sums)
        
        if len(current) > len(best_set):
            best_set = current[:]
            best_sums = current_sums[:]
            best_seen = set(current_seen)
    
    # Local search: try to add remaining elements
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable(best_sums, best_seen, e)
                if new_sums is not None:
                    best_set.append(e)
                    best_sums.extend(new_sums)
                    best_seen.update(new_sums)
                    remaining.remove(e)
                    improved = True
    
    # Local search: try swaps to potentially enable adding more elements
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        # Try removing one element and adding two others
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            # Rebuild without removed
            test_set = best_set[:i] + best_set[i+1:]
            test_sums = [Fraction(0, 1)]
            test_seen = {Fraction(0, 1)}
            for e in test_set:
                r = Fraction(1, e)
                new_s = []
                for s in test_sums:
                    ns = s + r
                    if ns in test_seen:
                        break
                    new_s.append(ns)
                else:
                    test_sums.extend(new_s)
                    test_seen.update(new_s)
                    continue
                break
            
            if len(test_sums) != 2 ** len(test_set):
                continue  # not dissociated (shouldn't happen)
            
            # Try to add two elements from remaining
            added = 0
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable(test_sums, test_seen, e)
                if new_sums is not None:
                    test_set.append(e)
                    test_sums.extend(new_sums)
                    test_seen.update(new_sums)
                    added += 1
                    if added >= 2:
                        break
            
            if added >= 2:
                # This swap improves size
                best_set = test_set
                best_sums = test_sums
                best_seen = test_seen
                remaining = [e for e in all_elements if e not in best_set]
    
    # Backtracking search: try to find a larger set by exploring different choices
    # This helps escape greedy local optima
    if time.time() < deadline and len(best_set) < 12:
        def backtrack_search(available, current_set, current_sums, current_seen, depth=0):
            """Try to build a larger dissociated set via backtracking."""
            if time.time() > deadline:
                return current_set, current_sums, current_seen
            if depth > 15:  # limit recursion depth
                return current_set, current_sums, current_seen
            
            best_local = (current_set[:], current_sums[:], set(current_seen))
            
            for idx, e in enumerate(available):
                if time.time() > deadline:
                    break
                new_sums = check_addable(current_sums, current_seen, e)
                if new_sums is not None:
                    new_set = current_set + [e]
                    new_sums_all = current_sums + new_sums
                    new_seen = set(current_seen)
                    new_seen.update(new_sums)
                    
                    # Recurse
                    remaining_avail = available[:idx] + available[idx+1:]
                    result_set, result_sums, result_seen = backtrack_search(
                        remaining_avail, new_set, new_sums_all, new_seen, depth + 1
                    )
                    
                    if len(result_set) > len(best_local[0]):
                        best_local = (result_set, result_sums, result_seen)
            
            return best_local
        
        # Try backtracking from a few different starting points
        start_points = [
            [],  # start from empty
            [22],
            [22, 21],
            [2, 3],
        ]
        
        for start in start_points:
            if time.time() > deadline:
                break
            start_sums = [Fraction(0, 1)]
            start_seen = {Fraction(0, 1)}
            for e in start:
                r = Fraction(1, e)
                new_s = []
                for s in start_sums:
                    ns = s + r
                    if ns in start_seen:
                        break
                    new_s.append(ns)
                else:
                    start_sums.extend(new_s)
                    start_seen.update(new_s)
                    continue
                break
            
            available = [e for e in all_elements if e not in start]
            result_set, result_sums, result_seen = backtrack_search(
                available, start[:], start_sums, start_seen
            )
            
            if len(result_set) > len(best_set):
                best_set = result_set
                best_sums = result_sums
                best_seen = result_seen
    
    # Fallback: use the seed answer if we didn't find anything better
    if not best_set or len(best_set) <= 9:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Final verification
    if not is_dissociated(best_set):
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
