import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin

    def get_subset_sums(elements):
        """Compute all nonempty subset sums as a set."""
        sums = set()
        for e in elements:
            new_sums = {e + s for s in sums}
            new_sums.add(e)
            sums = sums | new_sums
        return sums

    def is_divisibility_free(sums):
        """Check if no two distinct sums divide each other. Optimized with early exit."""
        sl = sorted(sums)
        n = len(sl)
        for i in range(n):
            a = sl[i]
            if a <= 1:
                continue
            for j in range(i + 1, n):
                b = sl[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
                # Optimization: if b is much larger than a, skip ahead
                # Actually we can't skip, but we can break if b > max_possible_divisor
                # Better: just continue the loop
        return True

    def try_add(current_elements, current_sums, candidate):
        """Try adding candidate. Returns (new_elements, new_sums) if valid, else None."""
        # Check for collisions first
        new_sums = set(current_sums)
        if candidate in new_sums:
            return None
        new_sums.add(candidate)
        
        # Generate new sums
        for s in list(current_sums):
            ns = s + candidate
            if ns in new_sums:
                return None  # collision
            new_sums.add(ns)
        
        # Check divisibility
        if not is_divisibility_free(new_sums):
            return None
        
        return current_elements + [candidate], new_sums

    best_set = [31, 32, 34]  # fallback
    best_size = len(best_set)

    # Backtracking search with time limit
    def backtrack(elements, sums, candidates, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if len(elements) > best_size:
            best_set = list(elements)
            best_size = len(elements)
        
        if not candidates:
            return
        
        for i, cand in enumerate(candidates):
            if time.time() > deadline:
                return
            result = try_add(elements, sums, cand)
            if result is not None:
                new_elements, new_sums = result
                backtrack(new_elements, new_sums, candidates[i+1:], depth + 1)

    # Strategy 1: Backtracking with various starting points
    # Use numbers from ranges that are likely to work
    for start_val in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 30, 32, 36, 40, 48, 56, 64, 80, 100, 128]:
        if time.time() > deadline - 2.0:
            break
        # Generate candidates
        candidates = list(range(start_val, 257))
        
        # Try sorted order first (often works better for backtracking)
        backtrack([], set(), sorted(candidates), 0)
        
        # Try a few random orderings
        if time.time() > deadline - 2.0:
            break
        random.seed(start_val)
        for _ in range(3):
            shuffled = list(candidates)
            random.shuffle(shuffled)
            backtrack([], set(), shuffled, 0)
            if time.time() > deadline - 2.0:
                break

    # Strategy 2: Greedy extension with large numbers (from inspiration)
    # Focus on [128, 256] to minimize divisibility conflicts
    large_candidates = list(range(128, 257))
    
    # Try several starting seeds with large numbers
    seed_sets = [
        [128, 129, 131],
        [130, 131, 133],
        [140, 141, 143],
        [150, 151, 153],
        [160, 161, 163],
        [170, 171, 173],
        [180, 181, 183],
        [190, 191, 193],
        [200, 201, 203],
        [210, 211, 213],
        [220, 221, 223],
        [230, 231, 233],
        [240, 241, 243],
        [245, 247, 249],
        [250, 251, 253],
    ]

    for seed in seed_sets:
        if time.time() > deadline - 3.0:
            break
        if not is_divisibility_free(get_subset_sums(seed)):
            continue
        
        # Try descending order
        candidates = [x for x in large_candidates if x not in seed]
        candidates.sort(reverse=True)
        
        current_elements = list(seed)
        current_sums = get_subset_sums(seed)
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            result = try_add(current_elements, current_sums, x)
            if result is not None:
                current_elements, current_sums = result
        
        if len(current_elements) > best_size:
            best_set = current_elements
            best_size = len(best_set)

        # Try ascending
        if time.time() > deadline - 3.0:
            continue
        candidates.sort()
        current_elements = list(seed)
        current_sums = get_subset_sums(seed)
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            result = try_add(current_elements, current_sums, x)
            if result is not None:
                current_elements, current_sums = result
        
        if len(current_elements) > best_size:
            best_set = current_elements
            best_size = len(best_set)

    # Strategy 3: Randomized restarts with large numbers
    random.seed(42)
    for _ in range(300):
        if time.time() > deadline - 3.0:
            break
        seed_size = random.randint(2, 5)
        seed = random.sample(large_candidates, seed_size)
        seed.sort()
        if not is_divisibility_free(get_subset_sums(seed)):
            continue
        
        candidates = [x for x in large_candidates if x not in seed]
        random.shuffle(candidates)
        
        current_elements = list(seed)
        current_sums = get_subset_sums(seed)
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            result = try_add(current_elements, current_sums, x)
            if result is not None:
                current_elements, current_sums = result
        
        if len(current_elements) > best_size:
            best_set = current_elements
            best_size = len(best_set)

    # Strategy 4: Local search with swaps
    if time.time() < deadline - 5.0:
        current_sums = get_subset_sums(best_set)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 5.0:
                    break
                for x in range(128, 257):
                    if x in best_set:
                        continue
                    new_set = best_set[:i] + [x] + best_set[i+1:]
                    if is_divisibility_free(get_subset_sums(new_set)):
                        # Try to extend from this swapped set
                        current_sums_new = get_subset_sums(new_set)
                        candidates = [y for y in range(128, 257) if y not in new_set]
                        candidates.sort(reverse=True)
                        
                        current_elements = list(new_set)
                        current_sums = current_sums_new
                        
                        for y in candidates:
                            if time.time() > deadline - 5.0:
                                break
                            result = try_add(current_elements, current_sums, y)
                            if result is not None:
                                current_elements, current_sums = result
                        
                        if len(current_elements) > best_size:
                            best_set = current_elements
                            best_size = len(best_set)
                            current_sums = get_subset_sums(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Final verification
    if not is_divisibility_free(get_subset_sums(best_set)):
        best_set = [31, 32, 34]

    result = list(best_set)
    result = [int(x) for x in result]
    return result
