import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Specific parameters for the task: q=3, n=7, radius=1
    # Total space size: 3^7 = 2187
    # We represent words as integers 0..2186
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        # For this specific task, we assume q=3, n=7, radius=1
        # A simple greedy fallback
        return _greedy_fallback(q, n, radius)
    
    space_size = q ** n  # 2187
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices covered by word i
    neighbors = []
    for i in range(space_size):
        t = [0] * n
        x = i
        for j in range(n):
            t[j] = x % q
            x //= q
        
        covered = [i]
        for pos in range(n):
            orig_val = t[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t[pos] = new_val
                val = 0
                mult = 1
                for j in range(n):
                    val += t[j] * mult
                    mult *= q
                covered.append(val)
        neighbors.append(covered)
        
    # Convert neighbors to numpy arrays for faster coverage checking?
    # Actually, we can maintain a coverage count array.
    # coverage_count[j] = number of codewords that cover word j
    
    def run_greedy():
        """
        Standard greedy set cover.
        Returns a list of codeword indices.
        """
        covered_count = np.zeros(space_size, dtype=np.int32)
        codewords = []
        
        # To allow for restarts with different tie-breaking, we can shuffle indices
        # but greedy is deterministic if we always pick max.
        # We can add a small random perturbation to break ties.
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            best_idx = -1
            best_new = 0
            
            # Iterate through all possible words
            # Optimization: We can skip words that are already in codewords? No, they might still cover new things if neighbors overlap differently? 
            # Actually, if a word is already a codeword, adding it again is useless (distinct codewords required).
            # So we only consider words NOT in codewords.
            
            # For speed, we just iterate all and check if it's in the set.
            # A set of codewords for O(1) lookup.
            cw_set = set(codewords)
            
            for i in range(space_size):
                if i in cw_set:
                    continue
                
                # Count new coverage
                # new_cov = sum(1 for nb in neighbors[i] if covered_count[nb] == 0)
                
                # Vectorized check is faster
                # new_cov = np.sum(covered_count[neighbors[i]] == 0)
                
                # To avoid numpy overhead for small lists, manual loop might be competitive, 
                # but 2187 * 186 * 15 is ~6M ops. Numpy is better.
                
                new_cov = np.sum(covered_count[neighbors[i]] == 0)
                
                if new_cov > best_new:
                    best_new = new_cov
                    best_idx = i
                    # Early exit? No, we need the max.
            
            if best_idx == -1:
                break # All covered
            
            codewords.append(best_idx)
            for nb in neighbors[best_idx]:
                covered_count[nb] += 1
            
            if np.all(covered_count > 0):
                break
                
        return codewords, covered_count

    def local_search(codewords, covered_count):
        """
        Local search: Try to remove redundant codewords.
        Then try swaps if removal fails? 
        For now, aggressive removal.
        """
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Try to remove each codeword
            # Iterate in reverse order or random order to avoid bias
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                    
                cw = codewords[idx]
                
                # Check if removing cw leaves any word uncovered
                # A word is uncovered if covered_count[word] == 1 AND that word is covered by cw
                # So, check if any neighbor of cw has covered_count == 1
                
                is_redundant = True
                for nb in neighbors[cw]:
                    if covered_count[nb] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for nb in neighbors[cw]:
                        covered_count[nb] -= 1
                    codewords.pop(idx)
                    improved = True
                    # Restart the loop to check again, as removals can make others redundant
                    break
        
        return codewords, covered_count

    def local_search_swap(codewords, covered_count):
        """
        Try to swap a codeword with another to enable removals.
        This is more complex. 
        If removal fails, try to swap codeword c_i with a non-codeword c_j 
        such that the new set is still valid.
        This is O(N^2) per step, which might be too slow.
        Instead, we can try a simpler heuristic: 
        If a codeword is "critical" (covers some unique words), can we find another codeword 
        that covers those unique words better, allowing the critical one to be removed?
        """
        # For time efficiency, we might skip complex swaps and rely on multiple greedy restarts
        # or just the removal phase which is very fast.
        pass

    # Strategy:
    # 1. Run Greedy
    # 2. Run Local Search (Removal)
    # 3. If time permits, try restarting Greedy with different tie-breaking to find a better initial solution
    #    or run multiple Greedy+LocalSearch cycles.
    
    best_codewords = []
    best_size = float('inf')
    
    # Run a few times to ensure we find a good local optimum
    # Since Greedy is deterministic, we need to perturb it.
    # We can perturb by adding a small random weight to the "new coverage" score.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturbed Greedy
        covered_count = np.zeros(space_size, dtype=np.int32)
        codewords = []
        cw_set = set()
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            best_idx = -1
            best_score = -1
            
            # Add random noise to break ties and explore different paths
            # Score = new_coverage + random_noise
            
            for i in range(space_size):
                if i in cw_set:
                    continue
                
                new_cov = np.sum(covered_count[neighbors[i]] == 0)
                
                # If new_cov is 0, it's useless
                if new_cov == 0:
                    continue
                
                # Add small noise
                noise = random.random() * 0.1
                score = new_cov + noise
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                break # All covered or no useful moves
            
            codewords.append(best_idx)
            cw_set.add(best_idx)
            for nb in neighbors[best_idx]:
                covered_count[nb] += 1
            
            if np.all(covered_count > 0):
                break
        
        # Local Search: Remove redundant
        codewords, covered_count = local_search(codewords, covered_count)
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
        # Optional: Try to improve further with swaps if time allows
        # But let's stick to Remove for now as it's very effective for Greedy outputs.
        
    # Convert best indices back to lists
    def int_to_tuple(x, n):
        return [x // (q**i) % q for i in range(n)]
        
    result = [int_to_tuple(i, n) for i in best_codewords]
    
    return result

def _greedy_fallback(q, n, radius):
    # Simple fallback for other parameters
    space_size = q ** n
    neighbors = []
    for i in range(space_size):
        t = [0] * n
        x = i
        for j in range(n):
            t[j] = x % q
            x //= q
        covered = [i]
        for pos in range(n):
            orig_val = t[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t[pos] = new_val
                val = 0
                mult = 1
                for j in range(n):
                    val += t[j] * mult
                    mult *= q
                covered.append(val)
        neighbors.append(covered)
        
    covered_count = np.zeros(space_size, dtype=np.int32)
    codewords = []
    cw_set = set()
    
    while True:
        best_idx = -1
        best_new = 0
        for i in range(space_size):
            if i in cw_set:
                continue
            new_cov = np.sum(covered_count[neighbors[i]] == 0)
            if new_cov > best_new:
                best_new = new_cov
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(best_idx)
        cw_set.add(best_idx)
        for nb in neighbors[best_idx]:
            covered_count[nb] += 1
        if np.all(covered_count > 0):
            break
            
    def int_to_tuple(x, n):
        return [x // (q**i) % q for i in range(n)]
    return [int_to_tuple(i, n) for i in codewords]
