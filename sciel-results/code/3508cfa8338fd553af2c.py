import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
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
    # neighbors[idx] is a list of 17 indices
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

    # Pre-compute neighbor sets for fast membership testing
    neighbor_sets = [set(neighbors[i]) for i in range(total_points)]

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Pre-compute reverse mapping: for each point, which balls contain it?
        rev_neighbors = [[] for _ in range(total_points)]
        for b in range(total_points):
            for p in neighbors[b]:
                rev_neighbors[p].append(b)
                
        # uncovered_counts[b] = number of uncovered points in ball b.
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        while np.any(uncovered_counts > 0):
            # Find max uncovered count
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Random tie-breaking among max
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            
            # Update: for each point p in neighbors[best_idx] that was uncovered,
            # decrement uncovered_counts for all balls containing p.
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for b_prime in rev_neighbors[p]:
                        uncovered_counts[b_prime] -= 1
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    selected_list.pop(i)
                    coverage_count[neighbors[idx]] -= 1
                    changed = True
                    break  # Restart scan since indices shifted
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Pre-compute a set for fast membership check
        selected_set = set(selected_list)
        
        index_to_pos = {idx: i for i, idx in enumerate(selected_list)}
        
        stall_counter = 0
        max_stall = 5000  # Try more iterations before giving up
        
        while time.time() < deadline and stall_counter < max_stall:
            if not selected_list:
                break
                
            c = selected_list[rng.integers(len(selected_list))]
            
            # Try all neighbors of c as potential replacements
            found_swap = False
            for pos in range(1, ball_size):
                c_prime = neighbors[c][pos]
                
                if c_prime in selected_set:
                    continue
                
                # Tentative swap c -> c_prime
                # Check validity: no point in neighbors[c] \ neighbors[c_prime] has count == 1
                
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                # Check if any point in neighbors[c] is now 0
                if np.any(coverage_count[neighbors[c]] == 0):
                    # Invalid, revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    continue
                else:
                    # Valid swap - update structures
                    pos_in_list = index_to_pos[c]
                    selected_list[pos_in_list] = c_prime
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    index_to_pos.pop(c)
                    index_to_pos[c_prime] = pos_in_list
                    
                    found_swap = True
                    stall_counter = 0
                    break # Restart outer loop after a swap
            
            if not found_swap:
                stall_counter += 1
                # If we've stalled a lot, try a more aggressive approach:
                # randomly pick a codeword and try to remove it, then repair
                if stall_counter % 100 == 0:
                    # Try removing a random codeword and see if we can repair
                    rem_idx = selected_list[rng.integers(len(selected_list))]
                    coverage_count[neighbors[rem_idx]] -= 1
                    
                    if not np.any(coverage_count == 0):
                        # Removal was valid! We reduced the code size.
                        pos_in_list = index_to_pos[rem_idx]
                        selected_list.pop(pos_in_list)
                        selected_set.remove(rem_idx)
                        index_to_pos.pop(rem_idx)
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        stall_counter = 0
                    else:
                        # Revert
                        coverage_count[neighbors[rem_idx]] += 1
        
        return best_code

    def local_search_phase2(selected_indices, deadline):
        """Second phase: more aggressive exploration with larger moves"""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(123)
        selected_set = set(selected_list)
        index_to_pos = {idx: i for i, idx in enumerate(selected_list)}
        
        stall_counter = 0
        max_stall = 3000
        
        while time.time() < deadline and stall_counter < max_stall:
            if not selected_list:
                break
                
            # Randomly select a codeword to try moving
            c = selected_list[rng.integers(len(selected_list))]
            
            # Try neighbors of c, but also try some random points
            candidates = list(neighbors[c][1:])
            # Add a few random candidates for diversity
            for _ in range(3):
                random_point = rng.integers(total_points)
                if random_point not in selected_set:
                    candidates.append(random_point)
            
            found_swap = False
            for c_prime in candidates:
                if c_prime in selected_set:
                    continue
                
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                if np.any(coverage_count[neighbors[c]] == 0):
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    continue
                else:
                    pos_in_list = index_to_pos[c]
                    selected_list[pos_in_list] = c_prime
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    index_to_pos.pop(c)
                    index_to_pos[c_prime] = pos_in_list
                    
                    found_swap = True
                    stall_counter = 0
                    break
            
            if not found_swap:
                stall_counter += 1
                # Try removal every 50 stalls
                if stall_counter % 50 == 0:
                    rem_idx = selected_list[rng.integers(len(selected_list))]
                    coverage_count[neighbors[rem_idx]] -= 1
                    
                    if not np.any(coverage_count == 0):
                        pos_in_list = index_to_pos[rem_idx]
                        selected_list.pop(pos_in_list)
                        selected_set.remove(rem_idx)
                        index_to_pos.pop(rem_idx)
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        stall_counter = 0
                    else:
                        coverage_count[neighbors[rem_idx]] += 1
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0
    
    # Run a few greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')
    
    # Run 5 greedy instances
    for seed in range(5):
        if time.time() > deadline - 60:  # Leave 60s for local search
            break
        selected = luby_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = luby_greedy(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search on the best code - Phase 1
    best_code = local_search_optimize(best_code, deadline)
    
    # Local Search Phase 2 - more aggressive
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best_code = local_search_phase2(best_code, deadline)
    
    # Final prune
    best_code = prune(best_code)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
