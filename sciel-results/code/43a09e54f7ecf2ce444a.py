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
    powers = np.array([1] * n, dtype=np.int64)
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

    def greedy_cover(seed, deadline):
        """Fast greedy set cover with vectorized operations."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Pre-compute for speed
        while not np.all(is_covered) and time.time() < deadline:
            # Compute uncovered count for each candidate
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Luby weight^2 for selection probability
            weights = uncovered_counts.astype(np.float64)
            weights[weights == 0] = 0
            weights_sq = weights * weights
            total_w = weights_sq.sum()
            if total_w == 0:
                break
            probs = weights_sq / total_w
            best_idx = rng.choice(total_points, p=probs)
            
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        """Remove redundant codewords."""
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
                if np.all(coverage_count[ball] > 1):
                    coverage_count[ball] -= 1
                    selected_list.pop(i)
                    changed = True
                    break
            
        return selected_list

    def kick_and_prune(selected_indices, rng, deadline):
        """
        Kick and prune: randomly perturb the code, then aggressively prune.
        This helps escape local minima.
        """
        selected_list = list(selected_indices)
        best_code = selected_list[:]
        best_len = len(selected_list)
        
        # Try multiple kicks
        num_kicks = 50
        for kick_iter in range(num_kicks):
            if time.time() > deadline:
                break
            
            # Kick: randomly replace 1-3 codewords with nearby candidates
            kicked = selected_list[:]
            num_replacements = rng.integers(1, 4)
            for _ in range(num_replacements):
                if not kicked:
                    break
                pos = rng.integers(len(kicked))
                c = kicked[pos]
                # Pick a random neighbor of c (excluding c itself)
                c_prime_candidates = neighbors[c][1:]
                c_prime = c_prime_candidates[rng.integers(len(c_prime_candidates))]
                # Check if c_prime is already in the set
                if c_prime not in kicked:
                    kicked[pos] = c_prime
            
            # Prune the kicked code
            pruned = prune(kicked)
            
            if len(pruned) < best_len:
                best_len = len(pruned)
                best_code = pruned[:]
                selected_list = pruned[:]  # Continue from improved solution
        
        return best_code

    def local_search(selected_indices, deadline):
        """
        Local search with kick-and-prune strategy.
        """
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        best_code = selected_list[:]
        best_len = len(selected_list)
        
        rng = np.random.default_rng(12345)
        
        # Phase 1: Aggressive pruning
        while time.time() < deadline - 30:
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
            else:
                break
        
        # Phase 2: Kick and prune iterations
        kick_deadline = deadline - 5
        while time.time() < kick_deadline:
            # Perform a kick
            kicked = selected_list[:]
            num_replacements = rng.integers(1, 4)
            for _ in range(num_replacements):
                if not kicked:
                    break
                pos = rng.integers(len(kicked))
                c = kicked[pos]
                c_prime_candidates = neighbors[c][1:]
                c_prime = c_prime_candidates[rng.integers(len(c_prime_candidates))]
                if c_prime not in kicked:
                    kicked[pos] = c_prime
            
            # Prune
            pruned = prune(kicked)
            
            if len(pruned) < len(selected_list):
                # Improvement: accept
                selected_list = pruned
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
            elif rng.random() < 0.3:
                # Neutral or slight regression: accept with some probability for exploration
                selected_list = pruned
            # else: revert (keep original)
        
        return best_code

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Run multiple greedy initializations with different seeds
    best_code = None
    best_len = float('inf')
    
    greedy_deadline = start_time + 30.0
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        if time.time() > greedy_deadline:
            break
        selected = greedy_cover(seed, deadline)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]
    
    if best_code is None:
        best_code = greedy_cover(0, deadline)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Run local search on the best greedy solution
    remaining_deadline = deadline - 5
    best_code = local_search(best_code, remaining_deadline)
    
    # Final prune
    best_code = prune(best_code)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
