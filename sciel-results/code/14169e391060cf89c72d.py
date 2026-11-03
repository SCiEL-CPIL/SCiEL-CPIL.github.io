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
                    nw = w[:]
                    nw[pos] = val
                    n_idx = 0
                    for v in nw:
                        n_idx = n_idx * q + v
                    nbs.append(n_idx)
        
        return nbs

    for i in range(total_space):
        nbs = get_neighbors(i)
        neighbors[i] = nbs

    def greedy_cover(randomize=False):
        covered = np.zeros(total_space, dtype=bool)
        code = []
        
        while not np.all(covered):
            gains = np.sum(covered[neighbors] == 0, axis=1)
            
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
            
            # Exclude words already in current_code (except we want to allow re-adding idx_out? No)
            # Actually, we should exclude all current codewords to force a real swap
            # But if the best gain is 0, no good swap exists
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

    # Strategy:
    # 1. Run deterministic greedy + reduce.
    # 2. Run randomized greedy + reduce + swap local search multiple times.
    # 3. Keep the best.

    best_code = greedy_cover(randomize=False)
    best_code = reduce_code(best_code)
    best_len = len(best_code)
    
    # Try swap local search on the initial solution
    improved_code = swap_local_search(best_code, max_iterations=200)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code
    
    # Randomized restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        cand_code = greedy_cover(randomize=True)
        cand_code = reduce_code(cand_code)
        cand_code = swap_local_search(cand_code, max_iterations=100)
        num_restarts += 1
        
        if len(cand_code) < best_len:
            best_len = len(cand_code)
            best_code = cand_code
            
    # Convert indices back to words
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    return [idx_to_word(i) for i in best_code]
