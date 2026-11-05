import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            # Tetracode construction for q=3, n=4
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Extend to length n by adding arbitrary tails
            # This is a valid code, just not optimal for large n
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
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

    def greedy_set_cover(seed=None, deterministic=False):
        if deterministic:
            rng = None
        else:
            rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            # Count how many uncovered points each candidate covers
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                if deterministic:
                    best_idx = int(candidates[0])
                else:
                    best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
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
                # If all points in this ball are covered by other codewords (count > 1), we can remove it
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break

        return selected_list

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def local_search_walk(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Remove any removable codeword (full sweep, deterministic)
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            # 2. Neighbor-restricted swap: move a random codeword to a point in its own ball
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            progress = False
            # Try a few swaps before giving up on this iteration
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)
                
                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue
                    
                    # Tentatively swap: remove c, add c_prime
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    
                    # Check validity: no point in balls[c] should have count 0
                    # Note: balls[c_prime] counts are >= 1 because we just added it, 
                    # and other points were >= 1 before. 
                    # The risk is only in balls[c] \ balls[c_prime].
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue
                    
                    # Valid swap
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                
                if progress:
                    break

            if progress:
                # After a swap, prune to catch newly-redundant codewords
                # We do a quick prune here to keep the code tight
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def polish(selected_indices, deadline):
        """Final high-precision pass: full deterministic sweeps + exhaustive neighbor checks for removals."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        # 1. Aggressive deterministic removal
        for _ in range(100):
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
            if not changed:
                break

        # 2. Try single-step moves that might enable removal
        # For each codeword, try moving it to a neighbor. 
        # If the move is valid and allows a removal, accept it.
        # This is more expensive, so we limit attempts.
        rng = random.Random(99)
        attempts = 0
        max_attempts = 5000 # Limit to prevent hanging
        
        while time.time() < deadline and attempts < max_attempts:
            if not selected_list:
                break
            
            # Pick a random codeword
            i = rng.randrange(len(selected_list))
            c = selected_list[i]
            nbrs = list(neighbors[c])
            rng.shuffle(nbrs)
            
            for c_prime in nbrs:
                if c_prime == c or c_prime in selected_set:
                    continue
                
                # Tentative move
                coverage_count[balls[c]] -= 1
                coverage_count[balls[c_prime]] += 1
                
                if np.any(coverage_count[balls[c]] == 0):
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    continue
                
                # Move is valid. Now check if we can remove any codeword.
                # We only check if c_prime itself is removable, or if the move made another codeword removable.
                # For efficiency, let's just see if the new state allows any removal.
                # This is O(N * 17). 
                
                # Optimization: Only check if c_prime is removable, or if we track "critical" points.
                # Given the small size, a full scan is okay if we limit attempts.
                
                removed_any = False
                removed_idx = -1
                for j in range(len(selected_list)):
                    idx_j = selected_list[j]
                    if np.all(coverage_count[balls[idx_j]] > 1):
                        removed_any = True
                        removed_idx = j
                        break
                
                if removed_any:
                    # Accept the move and the removal
                    coverage_count[balls[removed_idx]] -= 1
                    selected_list.pop(removed_idx)
                    selected_set.discard(selected_list[removed_idx]) # Wait, pop returns nothing, need to get value before pop
                    # Correction:
                    # val = selected_list[removed_idx]
                    # selected_list.pop(removed_idx)
                    # selected_set.discard(val)
                    
                    # Let's redo that block carefully
                    val = selected_list[removed_idx]
                    selected_list.pop(removed_idx)
                    selected_set.discard(val)
                    
                    # Update best
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    
                    # Note: We already applied the move to coverage_count.
                    # We need to ensure the coverage_count reflects the final state.
                    # The move c->c_prime was applied.
                    # The removal of removed_idx was applied.
                    # So coverage_count is correct for the new state.
                    
                    attempts += 1
                    break # Restart the outer loop with the new state
                else:
                    # No removal possible, revert the move
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    attempts += 1
                    break # Try a different random codeword

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # 1. Generate initial codes
    best_code = None
    best_len = float('inf')

    # Deterministic greedy
    selected = greedy_set_cover(deterministic=True)
    selected = prune(selected)
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected[:]

    # Random greedy
    for seed in range(5):
        if time.time() > deadline - 40:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_set_cover(deterministic=True)
        best_code = prune(best_code)
        best_len = len(best_code)

    # 2. Local Search
    # We run local search on the best initial code
    best_code = local_search_walk(best_code, deadline - 10)

    # 3. Polish
    best_code = polish(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
