import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices

        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break

        return selected_list

    def local_search_walk(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)
        
        # Use a set for O(1) membership check
        selected_set = set(selected_list)
        
        while time.time() < deadline:
            # 1. Try to remove a random codeword
            if len(selected_list) > 0:
                i = rng.randrange(len(selected_list))
                c = selected_list[i]
                
                # Check if c can be removed
                # Condition: all points in ball(c) have coverage_count > 1
                if np.all(coverage_count[balls[c]] > 1):
                    # Remove c
                    coverage_count[balls[c]] -= 1
                    selected_list.pop(i)
                    selected_set.remove(c)
                    
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        # Reset the "walk" to continue from this new best
                        # We don't reset the walk, we just continue
                else:
                    # Try a swap
                    # Pick a random c' not in selected_set
                    # To avoid O(N) check for c' in set, we can just pick random and check
                    # If it's in set, try again (with a limit)
                    for _ in range(10):
                        c_prime = rng.randrange(total_points)
                        if c_prime in selected_set:
                            continue
                        
                        # Tentatively swap
                        # Remove c, Add c_prime
                        coverage_count[balls[c]] -= 1
                        coverage_count[balls[c_prime]] += 1
                        
                        # Check validity of the new set
                        # We need to ensure all points are covered
                        # Since we only changed the coverage of points in ball(c) and ball(c_prime),
                        # we only need to check those.
                        # Points in ball(c) lost 1 coverage. If any became 0, invalid.
                        # Points in ball(c_prime) gained 1 coverage. No risk of 0.
                        # Points outside both balls are unchanged.
                        
                        # So check: min(coverage_count[ball(c)]) >= 1
                        # And also, we must ensure that points in ball(c_prime) that were previously
                        # covered only by c (if c was in ball(c_prime)) are still covered.
                        # Wait, c is removed. So if a point x is in ball(c) AND ball(c_prime),
                        # it lost 1 from c and gained 1 from c_prime. Net change 0.
                        # If x is in ball(c) but NOT ball(c_prime), it lost 1. Must be >= 1.
                        # If x is in ball(c_prime) but NOT ball(c), it gained 1. No risk.
                        
                        # So the only risk is points in ball(c) \ ball(c_prime).
                        # But checking min(coverage_count[ball(c)]) >= 1 is sufficient and simpler.
                        
                        if np.all(coverage_count[balls[c]] >= 1):
                            # Valid swap
                            selected_list[i] = c_prime
                            selected_set.remove(c)
                            selected_set.add(c_prime)
                            # Do not update best_len, size is same
                            break
                        else:
                            # Revert
                            coverage_count[balls[c_prime]] -= 1
                            coverage_count[balls[c]] += 1
                            # Try another c_prime
                            continue
            
            # 2. Occasionally try to prune more aggressively or restart
            # If we are stuck for a while, we might want to try a different strategy
            # But the walk should eventually find a removal if one exists in the neighborhood

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    # Run greedy set cover with different seeds
    best_code = None
    best_len = float('inf')

    # Do a few greedy runs to get a good starting point
    for seed in range(5):
        if time.time() > deadline - 60:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local search with Random Walk
    best_code = local_search_walk(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
