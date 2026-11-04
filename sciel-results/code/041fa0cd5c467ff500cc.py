import numpy as np
import time
import itertools

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

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def greedy_cover():
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        # Precompute for each point, which balls cover it
        # This is the inverse of neighbors
        # point_to_balls[p] = list of ball indices b such that p in neighbors[b]
        # We can build this by iterating over all balls
        point_to_balls = [[] for _ in range(total_points)]
        for b in range(total_points):
            for p in neighbors[b]:
                point_to_balls[p].append(b)
        
        while not np.all(is_covered):
            # Find ball with max uncovered points
            # We need to compute sum of ~is_covered[neighbors[b]] for each b
            # This is expensive if done naively.
            # However, we can maintain a list of active balls?
            # Or just vectorize:
            # mask = ~is_covered
            # counts = mask[neighbors].sum(axis=1)
            mask = ~is_covered
            counts = mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Pick random ball among those with max count
            candidates = np.where(counts == max_count)[0]
            b = candidates[np.random.randint(len(candidates))]
            
            selected.append(b)
            new_points = neighbors[b][~is_covered[neighbors[b]]]
            # For each newly covered point, decrement counts of all balls covering it
            # But we don't have counts as a state variable, we recompute.
            # This is O(N * BallSize) per step. N=6561, BallSize=17. ~111k ops.
            # 400 steps -> 44M ops. OK in numpy.
            
            is_covered[neighbors[b]] = True
            
        return selected

    def prune(selected):
        if not selected:
            return []
        
        selected_list = list(selected)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to avoid index shifting issues?
            # Or just restart.
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    selected_list.pop(i)
                    coverage_count[neighbors[idx]] -= 1
                    changed = True
                    break # Restart scan
        return selected_list

    def local_search(selected, deadline):
        selected_list = prune(selected)
        best_list = selected_list[:]
        best_len = len(best_list)
        
        # Precompute point_to_balls for the current set? No, it's static.
        # But we need to know which selected balls cover a point.
        # Let's maintain coverage_count.
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        rng = np.random.default_rng(42)
        
        # Simulated Annealing / Random Walk
        # Objective: minimize len(selected_list)
        # Moves:
        # 1. Remove a redundant codeword (if any). Always accept.
        # 2. Swap a codeword with a neighbor. Accept if valid (no uncovered points) and (len decreases OR random prob).
        #    Actually, swap keeps len same. We need to try removing after swap.
        # 3. Add a codeword? No, we want to minimize.
        
        # Strategy:
        # Loop:
        #   - Try to remove any redundant codeword.
        #   - If none, try a random swap (c -> c').
        #   - Check if swap is valid (all points in c's ball still covered by others or c').
        #     Actually, if we swap c to c', points in ball(c) \ ball(c') lose one cover.
        #     If any of those had count 1, they become 0 -> Invalid.
        #     Points in ball(c') \ ball(c) gain one cover.
        #     Points in intersection keep same count.
        #   - If valid, perform swap. Then try to prune.
        #     If pruning succeeds, keep.
        #     If pruning fails, revert swap? Or keep and continue?
        #     Keeping neutral swaps allows exploration.
        
        while time.time() < deadline:
            # 1. Prune
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                # Recount
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_list = selected_list[:]
            
            if not selected_list:
                break

            # 2. Random Swap
            # Pick random c
            c_idx = rng.integers(len(selected_list))
            c = selected_list[c_idx]
            
            # Pick random neighbor c'
            # Exclude c itself? neighbors[c][0] is c.
            # Pick from 1..16
            c_prime_idx = rng.integers(1, ball_size)
            c_prime = neighbors[c, c_prime_idx]
            
            if c_prime in selected_list:
                continue
            
            # Check validity of swap
            # Points in ball(c) that are NOT in ball(c') will have their count decreased by 1.
            # If any of those had count 1, they become 0.
            # We need to check: min(coverage_count[ball(c) \ ball(c')]) > 1?
            # Or simply:
            # Tentatively update counts?
            # Or check: for p in ball(c):
            #   if p not in ball(c_prime) and coverage_count[p] == 1: invalid
            
            # Efficient check:
            # Create a set of points in ball(c_prime) for O(1) lookup?
            # Or use boolean mask.
            # mask_cp = np.zeros(total_points, dtype=bool)
            # mask_cp[neighbors[c_prime]] = True
            # points_to_check = neighbors[c][~mask_cp[neighbors[c]]]
            # if np.any(coverage_count[points_to_check] == 1): invalid
            
            # This creates a boolean array of size 6561 every time.
            # Optimization: Precompute intersection sizes? No, depends on c, c'.
            
            # Let's just do the boolean mask approach. It's fast in numpy.
            mask_cp = np.zeros(total_points, dtype=bool)
            mask_cp[neighbors[c_prime]] = True
            
            ball_c = neighbors[c]
            points_lost = ball_c[~mask_cp[ball_c]]
            
            if len(points_lost) > 0 and np.any(coverage_count[points_lost] == 1):
                continue # Invalid swap
            
            # Swap is valid. Perform it.
            coverage_count[ball_c] -= 1
            coverage_count[neighbors[c_prime]] += 1
            selected_list[c_idx] = c_prime
            
            # Now try to prune again immediately
            # If we can remove something, great.
            # If not, we have a neutral swap.
            # We keep the swap to explore.
            
            # Occasionally, if we haven't improved in a while, we might want to revert neutral swaps?
            # Or just let the random walk wander.
            
            # Check for improvement
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_list = selected_list[:]

        return best_list

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    # Run greedy
    best_code = greedy_cover()
    best_code = prune(best_code)
    
    # Run local search
    best_code = local_search(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
