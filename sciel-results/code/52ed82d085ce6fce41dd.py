import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        
        # Fast float check
        reciprocals = [1.0 / e for e in elements]
        sums = [0.0]
        for r in reciprocals:
            new_sums = [s + r for s in sums]
            sums.extend(new_sums)
        
        # Sort to check for collisions
        sums.sort()
        EPS = 1e-12
        for i in range(len(sums) - 1):
            if abs(sums[i+1] - sums[i]) < EPS:
                # Potential collision, verify with Fraction
                return _is_dissociated_exact(elements)
        return True

    def _is_dissociated_exact(elements):
        """Exact check using Fraction."""
        n = len(elements)
        if n == 0:
            return True
        reciprocals = [Fraction(1, e) for e in elements]
        sums = [Fraction(0, 1)]
        seen = {Fraction(0, 1)}
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

    all_elements = list(range(1, 23))
    best_set = []

    # Define multiple orders to try
    # 1. Descending
    orders = [
        list(range(22, 0, -1)),
        list(range(1, 23)),
        list(range(22, 0, -2)) + list(range(1, 23, 2)),
        list(range(1, 23, 2)) + list(range(22, 0, -2)),
        # Interleaved
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
    ]

    # Random restarts
    random.seed(42)
    best_size = len(best_set)
    
    # Run many random greedy attempts
    for _ in range(500):
        if time.time() > deadline:
            break
        
        rand_order = all_elements[:]
        random.shuffle(rand_order)
        
        current = []
        for e in rand_order:
            if time.time() > deadline:
                break
            candidate = current + [e]
            if is_dissociated(candidate):
                current = candidate
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]

    # Local Search: Try to add remaining elements
    if time.time() < deadline:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                candidate = best_set + [e]
                if is_dissociated(candidate):
                    best_set = candidate
                    remaining.remove(e)
                    improved = True

    # Local Search: Swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_len = len(best_set)
            # Try swapping one element out for one in
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for e in all_elements:
                    if e in best_set:
                        continue
                    candidate = best_set[:]
                    candidate[i] = e
                    if is_dissociated(candidate):
                        # Try to fill with remaining
                        rem = [x for x in all_elements if x not in candidate]
                        improved_candidate = candidate[:]
                        for x in rem:
                            if time.time() > deadline:
                                break
                            test = improved_candidate + [x]
                            if is_dissociated(test):
                                improved_candidate = test
                        if len(improved_candidate) > current_len:
                            best_set = improved_candidate
                            best_size = len(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        best_size = len(best_set)
    
    # Final verification with exact check to be safe
    if not _is_dissociated_exact(best_set):
        # Try to repair
        for e in best_set[:]:
            test = [x for x in best_set if x != e]
            if _is_dissociated_exact(test):
                best_set = test
                break
        if not _is_dissociated_exact(best_set):
            best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
