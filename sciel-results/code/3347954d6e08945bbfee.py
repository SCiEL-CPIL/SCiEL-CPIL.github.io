import numpy as np
import time
import itertools
import random

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

    # Build reverse mapping: for each point, which balls contain it
    rev_neighbors = [[] for _ in range(total_points)]
    for b in range(total_points):
        for p in neighbors[b]:
            rev_neighbors[p].append(b)
    # Convert to numpy arrays for faster access
    rev_neighbors_arr = []
    for p in range(total_points):
        rev_neighbors_arr.append(np.array(rev_neighbors[p], dtype=np.int32))

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    # Build a strong initial solution using structured construction
    # Use the van Laarhoven et al. approach: start with a good base and extend
    def build_structured_initial():
        # Start with a greedy solution but with a smarter ordering
        # Use the fact that we can partition the space into blocks
        # For n=8, q=3, radius=1: use a 4+4 decomposition
        # First 4 coords: use tetracode (9 words covering 81 words with radius 1)
        # Last 4 coords: use tetracode (9 words covering 81 words with radius 1)
        # Total: 9*9 = 81 words, but this gives radius 1 only if we combine properly
        
        # Actually, for radius 1 covering, we need a different approach.
        # Let's use a systematic greedy with good initialization.
        
        # Start with all 3^8 words grouped by first 4 coordinates
        # For each group of 81 words (last 4 coords vary), we need ~5 codewords
        # to cover all 81 with radius 1 (ball size 17, so 81/17 ~ 4.76)
        
        # Better: use a 2-phase approach
        # Phase 1: Cover the "skeleton" using a coarser code
        # Phase 2: Fill in gaps
        
        # Let's just do a well-optimized greedy
        rng = np.random.default_rng(12345)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Use a priority queue approach: always pick the ball covering most uncovered points
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        while np.any(uncovered_counts > 0):
            # Find max uncovered count with random tie-breaking
            max_count = np.max(uncovered_counts)
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            
            # Update counts
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for b_prime in rev_neighbors_arr[p]:
                        uncovered_counts[b_prime] -= 1
            
        return selected

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Luby's weighting: use random priorities
        # Each ball gets a random priority, pick the one with max priority among those covering most uncovered
        ball_priorities = rng.random(total_points)
        
        while not np.all(is_covered):
            # For each ball, compute how many uncovered points it covers
            # and its priority
            best_idx = -1
            best_key = -1
            best_uncovered = -1
            
            for b in range(total_points):
                if is_covered[neighbors[b]].all():
                    continue
                uncovered_in_ball = np.sum(~is_covered[neighbors[b]])
                # Luby's rule: pick randomly among top performers
                # We'll use a soft-max approach
                key = uncovered_in_ball + ball_priorities[b] * 0.1
                if key > best_key:
                    best_key = key
                    best_idx = b
                    best_uncovered = uncovered_in_ball
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def fast_greedy(seed):
        """Faster greedy using vectorized operations"""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Use a simpler but faster approach: random ordering with local improvement
        order = rng.permutation(total_points)
        
        # First pass: greedily add balls that cover many uncovered points
        for b in order:
            if np.all(is_covered):
                break
            uncovered_in_ball = np.sum(~is_covered[neighbors[b]])
            if uncovered_in_ball >= 3:  # Lower threshold for better initial coverage
                selected.append(b)
                is_covered[neighbors[b]] = True
        
        # Second pass: fill remaining gaps - only add if it covers uncovered points
        if not np.all(is_covered):
            remaining_order = [b for b in order if b not in set(selected)]
            for b in remaining_order:
                if np.all(is_covered):
                    break
                if np.any(~is_covered[neighbors[b]]):
                    is_covered[neighbors[b]] = True
                    selected.append(b)
        
        return selected

    def prune(selected_indices, max_iter=100000):
        """Efficient pruning: remove redundant codewords"""
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        if len(selected_list) == 0:
            return []
        
        # Build coverage count array
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        # Iteratively remove redundant codewords
        # Use a queue-based approach for efficiency
        changed = True
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            # Find a redundant codeword
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break
        
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        """Improved local search with multiple strategies"""
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
        
        # Strategy 1: Single move (replace one codeword with a neighbor)
        # Strategy 2: Try to remove a codeword and see if coverage is still complete
        # Strategy 3: 2-opt: remove two codewords, add one that covers the gap
        
        no_improve_count = 0
        max_no_improve = 8000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improvement_made = False
            
            # Strategy 2: Try to remove codewords (most direct improvement)
            # Pick a random codeword and check if removing it breaks coverage
            for _ in range(150):  # Try up to 150 removals per iteration
                if not selected_list:
                    break
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                
                # Check if removing c leaves any point uncovered
                risky_points = neighbors[c][coverage_count[neighbors[c]] == 1]
                
                if len(risky_points) == 0:
                    # Safe to remove
                    coverage_count[neighbors[c]] -= 1
                    idx_in_list = selected_list.index(c)
                    selected_list.pop(idx_in_list)
                    selected_set.remove(c)
                    improvement_made = True
                    no_improve_count = 0
                    break
                else:
                    # Check if we can find a replacement that covers all risky points
                    # and is not already selected
                    # Try neighbors of c
                    for pos in range(1, ball_size):
                        c_prime = neighbors[c][pos]
                        if c_prime in selected_set:
                            continue
                        
                        # Check if c_prime covers all risky points
                        if np.all(np.isin(risky_points, neighbors[c_prime])):
                            # Perform swap
                            coverage_count[neighbors[c]] -= 1
                            coverage_count[neighbors[c_prime]] += 1
                            
                            idx_in_list = selected_list.index(c)
                            selected_list[idx_in_list] = c_prime
                            selected_set.remove(c)
                            selected_set.add(c_prime)
                            
                            improvement_made = True
                            no_improve_count = 0
                            break
                    if improvement_made:
                        break
            
            if not improvement_made:
                no_improve_count += 1
            
            # Track best
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            # Occasionally do a full prune
            if no_improve_count % 30 == 0 and no_improve_count > 0:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    selected_set = set(selected_list)
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
        
        # Final high-precision polish: iterate prune + local swaps until no improvement
        while time.time() < deadline:
            pruned = prune(selected_list)
            if len(pruned) == len(selected_list):
                break
            selected_list = pruned
            selected_set = set(selected_list)
            coverage_count = np.zeros(total_points, dtype=np.int32)
            for idx in selected_list:
                coverage_count[neighbors[idx]] += 1
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
        
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build initial solution
    # Try multiple greedy approaches and pick the best
    best_code = None
    best_len = float('inf')
    
    # Method 1: Structured greedy
    if time.time() < deadline - 70:
        selected = build_structured_initial()
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
    
    # Method 2: Fast greedy with different seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 60:
            break
        selected = fast_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
    
    if best_code is None:
        best_code = build_structured_initial()
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search
    best_code = local_search_optimize(best_code, deadline)
    
    # Final prune
    best_code = prune(best_code)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
