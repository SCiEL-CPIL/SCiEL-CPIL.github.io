"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _valid_solution(arr):
    """Check if all 2^12 subset sums are distinct."""
    n = len(arr)
    if n != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if len(set(arr)) != 12:
        return False
    # Generate all subset sums
    sums = set()
    sums.add(0)
    for x in arr:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums |= new_sums
    if len(sums) != 1 << n:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # Start with powers of 2 as a valid baseline
    best = [1 << i for i in range(12)]
    best_max = max(best)
    
    # Greedy construction: try to build a superincreasing-like sequence
    # but with smaller values. A sequence where each element is less than
    # sum of all previous + 1 won't work, but we can try to find a set
    # where all subset sums are distinct (a "sum-free" or "distinct subset sum" set).
    
    # Strategy: Start with a known good small set and try to reduce the max.
    # Known: For n=12, the minimal max is around 2^11 = 2048 for powers of 2.
    # But we can do better. Let's try a greedy approach with backtracking.
    
    # Try to find a set with smaller max using a greedy + local search approach
    # Key insight: we need all 4096 subset sums to be distinct.
    
    # Approach 1: Greedy with constraint
    # Start small and add elements one by one, ensuring distinct subset sums
    
    def try_greedy(max_val):
        """Try to build a valid set with max element <= max_val."""
        arr = []
        subset_sums = {0}
        
        for i in range(12):
            # Try to find the smallest value that maintains distinct subset sums
            # and keeps max <= max_val
            found = False
            for val in range(1, max_val + 1):
                if val in arr:
                    continue
                # Check if adding val keeps all subset sums distinct
                new_sums = set(subset_sums)
                for s in subset_sums:
                    ns = s + val
                    if ns in new_sums and ns != s:  # ns == s only if val==0
                        break
                    new_sums.add(ns)
                else:
                    # All good
                    if len(new_sums) == len(subset_sums) * 2:
                        arr.append(val)
                        subset_sums = new_sums
                        found = True
                        break
            if not found:
                return None
        return arr
    
    # Try different max values from low to high
    # Theoretical lower bound: sum of all elements >= 2^12 - 1 = 4095
    # So max must be at least ceil(4095/12) ~ 342, but in practice much higher
    
    # Start searching from a reasonable lower bound
    # Powers of 2 give max = 2048. Let's try to do better.
    
    search_start = 100
    search_end = 2048
    
    # First try greedy with various max values
    for max_val in range(search_start, search_end + 1, 1):
        if time.time() > deadline:
            break
        result = try_greedy(max_val)
        if result is not None:
            if max(result) < best_max:
                best = result
                best_max = max(result)
            # Found a valid solution with this max, we can stop if it's good enough
            # But let's continue to see if we can find an even smaller max
            # Actually, greedy finds the lexicographically smallest, so if it works
            # at max_val, any smaller max_val would also have been found earlier
            break
    
    # Local search / hill climbing to improve
    # Try to reduce the largest element by swapping
    
    if time.time() < deadline:
        # Try local search on the best solution found
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            current = best[:]
            current_max = max(current)
            
            # Try to reduce the largest element
            # Find the index of the max element
            max_idx = current.index(current_max)
            
            # Try replacing the max element with a smaller value
            improved = False
            for new_val in range(1, current_max):
                if new_val in current:
                    continue
                trial = current[:]
                trial[max_idx] = new_val
                if _valid_solution(trial):
                    if max(trial) < current_max:
                        best = trial
                        best_max = max(trial)
                        improved = True
                        break
            
            if not improved:
                # Try other modifications
                # Randomly pick an element and try to reduce it
                idx = random.randint(0, 11)
                orig_val = current[idx]
                for new_val in range(1, orig_val):
                    if new_val in current:
                        continue
                    trial = current[:]
                    trial[idx] = new_val
                    if _valid_solution(trial):
                        if max(trial) < current_max:
                            best = trial
                            best_max = max(trial)
                            improved = True
                            break
            
            if not improved and iteration > 50:
                # Try random restarts
                # Generate a random valid-looking set and try to fix it
                random.seed(iteration)
                trial = sorted(random.sample(range(1, current_max + 1), 12))
                if _valid_solution(trial):
                    if max(trial) < current_max:
                        best = trial
                        best_max = max(trial)
    
    return best

# EVOLVE-BLOCK-END
