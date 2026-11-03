import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_space = q ** n

    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within Hamming distance <= radius of i
    # For q=3, n=7, radius=1: 1 + 7*(3-1) = 15 neighbors

    neighbors = np.zeros((total_space, 1 + n * (q - 1)), dtype=np.int32)
    
    # Weights for fast index calculation
    weights = [q ** (n - 1 - i) for i in range(n)]

    def get_neighbors(idx):
        w = []
        tmp = idx
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        
        nbs = [idx]
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Use weight for faster index calculation
                    n_idx = idx + (val - orig_val) * weights[pos]
                    nbs.append(n_idx)
        
        return nbs

    for i in range(total_space):
        nbs = get_neighbors(i)
        neighbors[i] = nbs

    # Precompute a neighbor mask for fast intersection checks in local search
    # neighbor_mask[i, j] is True if j is a neighbor of i
    # This is a 2187x2187 boolean matrix, which is ~4.5 MB, acceptable.
    neighbor_mask = np.zeros((total_space, total_space), dtype=bool)
    for i in range(total_space):
        neighbor_mask[i, neighbors[i]] = True

    def greedy_cover(randomize=False):
        covered = np.zeros(total_space, dtype=bool)
        code = []
        
        while not np.all(covered):
            gains = np.sum(~covered[neighbors], axis=1)
            
            if randomize:
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 0:
                    best_idx = candidates[random.randint(0, len(candidates) - 1)]
                else:
                    break
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            code.append(best_idx)
            covered[neighbors[best_idx]] = True
            
        return code

    def reduce_code(code):
        """Remove redundant codewords from code. Returns reduced list."""
        if not code:
            return code
        
        current_code = list(code)
        
        if len(current_code) == 0:
            return current_code
            
        flat_neighbors = neighbors[current_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                min_count = np.min(cover_count[neighbors[idx]])
                
                if min_count > 1:
                    cover_count[neighbors[idx]] -= 1
                    current_code.pop(i)
                    improved = True
                    break
        
        return current_code

    def swap_local_search(code, max_iterations=500):
        """
        Perform swap-based local search to improve the covering code.
        Try removing a codeword and replacing it with a better one, then reduce.
        """
        if not code:
            return code
        
        current_code = list(code)
        best_code = current_code[:]
        best_len = len(current_code)
        
        if best_len == 0:
            return best_code
            
        # Build cover_count for current_code
        flat_neighbors = neighbors[best_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random codeword to swap out
            i = random.randint(0, len(current_code) - 1)
            idx_out = current_code[i]
            
            # Tentatively remove idx_out
            cover_count[neighbors[idx_out]] -= 1
            current_code.pop(i)
            
            # Find uncovered words
            uncovered_mask = (cover_count == 0)
            uncovered_count = np.sum(uncovered_mask)
            
            if uncovered_count == 0:
                # It was redundant - restore and continue
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Find best replacement: maximize coverage of uncovered words
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            # Exclude words already in current_code
            for cw in current_code:
                gains[cw] = -1
            
            best_j = np.argmax(gains)
            
            if gains[best_j] <= 0:
                # No good replacement found, revert
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Add best_j
            current_code.append(best_j)
            cover_count[neighbors[best_j]] += 1
            
            # Now try to reduce the new set
            new_code = list(current_code)
            # Rebuild cover_count for new_code to be safe
            flat_new = neighbors[new_code].flatten()
            new_cover_count = np.bincount(flat_new, minlength=total_space)
            
            # Try to remove redundant words
            improved = True
            while improved:
                improved = False
                random.shuffle(new_code)
                for k in range(len(new_code)):
                    idx = new_code[k]
                    if np.min(new_cover_count[neighbors[idx]]) > 1:
                        new_cover_count[neighbors[idx]] -= 1
                        new_code.pop(k)
                        improved = True
                        break
            
            if len(new_code) < best_len:
                best_len = len(new_code)
                best_code = new_code
                # Update current state to best
                current_code = new_code
                flat_best = neighbors[best_code].flatten()
                cover_count = np.bincount(flat_best, minlength=total_space)
            else:
                # Revert
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                # Remove best_j from current_code
                if best_j in current_code:
                    current_code.remove(best_j)
                cover_count[neighbors[best_j]] -= 1
        
        return best_code

    def simulated_annealing(code, max_iters=2000):
        """
        Simulated annealing on the binary selection vector.
        State: set of selected codewords.
        Moves: Add, Remove, Swap.
        Energy: number of uncovered words * big_penalty + number of selected codewords.
        We want to minimize energy.
        """
        current_code = list(code)
        if not current_code:
            return current_code
            
        current_cover_count = np.bincount(neighbors[current_code].flatten(), minlength=total_space)
        current_uncovered = np.sum(current_cover_count == 0)
        current_size = len(current_code)
        
        # Energy function: heavy penalty for uncovered words, light penalty for size
        # We treat "uncovered" as the primary objective.
        # If uncovered > 0, energy is dominated by uncovered.
        # If uncovered == 0, energy is size.
        # To allow exploration, we can use a smooth surrogate or just the raw counts.
        # Let's use: E = 1000 * uncovered + size
        
        best_code = current_code[:]
        best_uncovered = current_uncovered
        best_size = current_size
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        alpha = 0.999 # Cooling rate
        
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.33:
                # Add a random codeword
                candidate = random.randint(0, total_space - 1)
                if candidate not in current_code:
                    new_code = current_code + [candidate]
                    # Calculate delta
                    # This is expensive to do exactly every time without precomputation of coverage
                    # But for SA, we can approximate or just compute new energy.
                    # Given the small space (2187), computing new energy is O(N * K) where K is code size.
                    # N=2187, K~190. 2187*190 ~ 400k ops. 2000 iters ~ 800M ops. Too slow in Python.
                    
                    # Optimization: Maintain cover_count.
                    # When adding candidate:
                    # new_cover_count = current_cover_count.copy()
                    # new_cover_count[neighbors[candidate]] += 1
                    # delta_uncovered = (current_uncovered - np.sum((current_cover_count == 0) & (neighbors[candidate] == 0))) 
                    # Wait, simpler:
                    # Newly covered words are those in neighbors[candidate] that were previously 0.
                    # gain = np.sum(current_cover_count[neighbors[candidate]] == 0)
                    # new_uncovered = current_uncovered - gain
                    # new_size = current_size + 1
                    # delta_E = 1000 * (new_uncovered - current_uncovered) + (new_size - current_size)
                    #       = 1000 * (-gain) + 1
                    #       = 1 - 1000 * gain
                    
                    gain = np.sum(current_cover_count[neighbors[candidate]] == 0)
                    delta_E = 1 - 1000 * gain
                    
                    if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                        # Accept
                        current_code.append(candidate)
                        current_cover_count[neighbors[candidate]] += 1
                        current_uncovered -= gain
                        current_size += 1
                        
                        if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                            best_uncovered = current_uncovered
                            best_size = current_size
                            best_code = current_code[:]
                            
            elif move_type < 0.66:
                # Remove a random codeword
                if current_size == 0:
                    T *= alpha
                    continue
                    
                idx_to_remove = random.randint(0, current_size - 1)
                candidate = current_code[idx_to_remove]
                
                # Loss of coverage: words that were covered only by candidate
                # loss = np.sum(current_cover_count[neighbors[candidate]] == 1)
                loss = np.sum(current_cover_count[neighbors[candidate]] == 1)
                
                # new_uncovered = current_uncovered + loss
                # new_size = current_size - 1
                # delta_E = 1000 * loss - 1
                
                delta_E = 1000 * loss - 1
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    current_code.pop(idx_to_remove)
                    current_cover_count[neighbors[candidate]] -= 1
                    current_uncovered += loss
                    current_size -= 1
                    
                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                        best_uncovered = current_uncovered
                        best_size = current_size
                        best_code = current_code[:]
                        
            else:
                # Swap: Remove one, Add one
                if current_size == 0:
                    T *= alpha
                    continue
                    
                idx_to_remove = random.randint(0, current_size - 1)
                candidate_out = current_code[idx_to_remove]
                candidate_in = random.randint(0, total_space - 1)
                
                if candidate_in == candidate_out:
                    T *= alpha
                    continue
                    
                # Calculate delta
                # 1. Remove candidate_out
                # loss_out = np.sum(current_cover_count[neighbors[candidate_out]] == 1)
                # But this is tricky because candidate_in might cover some of those.
                
                # Better approach:
                # Temporarily remove candidate_out
                # cover_temp = current_cover_count.copy()
                # cover_temp[neighbors[candidate_out]] -= 1
                # uncovered_temp = np.sum(cover_temp == 0)
                
                # Now add candidate_in
                # cover_temp[neighbors[candidate_in]] += 1
                # uncovered_new = np.sum(cover_temp == 0)
                
                # This is O(N) per step. 2000 steps * 2187 ~ 4M ops. This is fine.
                
                cover_temp = current_cover_count.copy()
                cover_temp[neighbors[candidate_out]] -= 1
                # Count uncovered after removal
                uncovered_after_removal = np.sum(cover_temp == 0)
                
                # Count how many of those are covered by candidate_in
                # gain_in = np.sum((cover_temp[neighbors[candidate_in]] == 0))
                # Note: cover_temp[neighbors[candidate_in]] gives counts for neighbors of candidate_in
                # We want to know how many words w in neighbors[candidate_in] have cover_temp[w] == 0
                gain_in = np.sum(cover_temp[neighbors[candidate_in]] == 0)
                
                uncovered_new = uncovered_after_removal - gain_in
                
                delta_E = 1000 * (uncovered_new - current_uncovered) # Size change is 0 (swap)
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    current_code[idx_to_remove] = candidate_in
                    # Update cover_count
                    current_cover_count[neighbors[candidate_out]] -= 1
                    current_cover_count[neighbors[candidate_in]] += 1
                    current_uncovered = uncovered_new
                    # Size remains same
                    
                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                        best_uncovered = current_uncovered
                        best_size = current_size
                        best_code = current_code[:]
                        
            T *= alpha
            
            # If we found a perfect cover (uncovered == 0), we can try to reduce size greedily or continue SA to minimize size
            # But SA with the energy function will naturally try to reduce size if uncovered is 0.
            
        return best_code

    # Strategy:
    # 1. Run deterministic greedy + reduce.
    # 2. Run randomized greedy + reduce + swap local search multiple times.
    # 3. Use Simulated Annealing on the best solutions found so far.
    # 4. Keep the best.

    best_code = greedy_cover(randomize=False)
    best_code = reduce_code(best_code)
    best_len = len(best_code)
    
    # Try swap local search on the initial solution
    improved_code = swap_local_search(best_code, max_iterations=200)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code

    # Try Simulated Annealing on the initial solution
    sa_code = simulated_annealing(best_code, max_iters=500)
    # Reduce the SA result
    sa_code = reduce_code(sa_code)
    if len(sa_code) < best_len:
        best_len = len(sa_code)
        best_code = sa_code

    # Randomized restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        cand_code = greedy_cover(randomize=True)
        cand_code = reduce_code(cand_code)
        
        # Alternate between swap local search and simulated annealing
        if num_restarts % 2 == 0:
            cand_code = swap_local_search(cand_code, max_iterations=100)
        else:
            cand_code = simulated_annealing(cand_code, max_iters=300)
            cand_code = reduce_code(cand_code)
            
        num_restarts += 1
        
        if len(cand_code) < best_len:
            best_len = len(cand_code)
            best_code = cand_code
            
            # If we find a significantly better solution, we might want to run SA on it immediately
            # But the loop continues, so it will be used in future comparisons or if we add a specific check.

    # Final polish: Run reduce and swap on the absolute best
    best_code = reduce_code(best_code)
    best_code = swap_local_search(best_code, max_iterations=100)
    best_code = reduce_code(best_code)
    
    # One more SA pass on the final best if time permits
    if time.time() < deadline - 2.0:
        sa_final = simulated_annealing(best_code, max_iters=200)
        sa_final = reduce_code(sa_final)
        if len(sa_final) < len(best_code):
            best_code = sa_final

    # Convert indices back to words
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    return [idx_to_word(i) for i in best_code]
