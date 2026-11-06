"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    def is_sidon_subset(subset):
        """Check if a subset is Sidon (all unordered pair sums including diagonals are unique)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(candidate):
        """Find the maximum size of a Sidon subset of candidate."""
        n = len(candidate)
        # Try from largest possible downward
        for size in range(n, 0, -1):
            for subset in combinations(candidate, size):
                if is_sidon_subset(subset):
                    return size
        return 0
    
    def count_sidon_subsets(candidate, max_size=6):
        """Count how many Sidon subsets of size max_size exist (for scoring)."""
        count = 0
        for subset in combinations(candidate, max_size):
            if is_sidon_subset(subset):
                count += 1
        return count
    
    best_set = None
    best_ms = 12  # worst case
    
    # Strategy: search in small ranges to force collisions
    # Focus on very small ranges where sum collisions are most likely
    random.seed(42)
    
    # Prioritize ranges that are just large enough for 12 elements
    # Small ranges force many elements to be close, creating sum collisions
    ranges_to_try = [
        (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17),
        (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23),
        (0, 24), (0, 25), (1, 13), (2, 14), (3, 15), (4, 16),
        (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22)
    ]
    
    for lo, hi in ranges_to_try:
        if time.time() > deadline:
            break
        range_size = hi - lo + 1
        if range_size < 12:
            continue
        
        # Try random sets in this range
        # For smaller ranges, do more trials
        num_trials = 300 if range_size <= 15 else 150
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Generate random 12-element set
            candidate = sorted(random.sample(range(lo, hi + 1), 12))
            
            # Quick check: if max_sidon_size is already good, keep it
            ms = max_sidon_size(candidate)
            if ms < best_ms:
                best_ms = ms
                best_set = candidate
                if best_ms <= 4:
                    return best_set
        
        # Local search from best found in this range
        if best_set is not None and best_ms > 4:
            # Try to improve by swapping elements
            current = list(best_set)
            for iteration in range(30):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(12):
                    for new_val in range(lo, hi + 1):
                        if new_val in current:
                            continue
                        old_val = current[i]
                        current[i] = new_val
                        current.sort()
                        ms = max_sidon_size(current)
                        if ms < best_ms:
                            best_ms = ms
                            best_set = current
                            improved = True
                            if best_ms <= 4:
                                return best_set
                        else:
                            current[i] = old_val
                        current.sort()
                    if improved:
                        current = list(best_set)
                if not improved:
                    break
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
