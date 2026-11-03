import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute neighbors for radius 1 as a list of NumPy arrays
    # neighbors[idx] = np.array of indices within distance <= 1
    neighbors = []
    
    # Pre-compute digit weights
    weights = [q ** (n - 1 - i) for i in range(n)]
    
    for idx in range(total_words):
        w = [0] * n
        temp = idx
        for d in range(n-1, -1, -1):
            w[d] = temp % q
            temp //= q
            
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                new_idx = idx + (new_val - w[i]) * weights[i]
                neigh.append(new_idx)
        neighbors.append(np.array(neigh, dtype=np.int32))

    # Pre-compute a boolean mask matrix for fast "is neighbor" checks?
    # 2187 x 2187 boolean matrix is ~4.8 MB, which is fine.
    # neighbor_mask[idx, j] is True if j is in neighbors[idx]
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for idx in range(total_words):
        neighbor_mask[idx, neighbors[idx]] = True

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    def greedy_construct(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Use NumPy for vectorized gain calculation
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        remaining_count = total_words
        
        # Pre-compute neighbor matrix as int8 for matrix multiplication?
        # gain = sum(uncovered[neighbors[idx]])
        # We can use a matrix M where M[idx, j] = 1 if j in neighbors[idx]
        # Then gains = M @ uncovered
        # M is 2187x2187.
        # Let's construct M once.
        
        while remaining_count > 0:
            # Vectorized gain calculation
            # M is boolean/int matrix
            gains = (M @ uncovered).astype(np.int32)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one
            chosen_idx = candidates[np.random.randint(len(candidates))]
            
            code.append(chosen_idx)
            
            # Update uncovered
            # uncovered[neighbors[chosen_idx]] = 0
            # But we need to count how many become 0 to update remaining_count
            # prev_uncovered = uncovered[neighbors[chosen_idx]]
            # new_uncovered = 0
            # remaining_count -= np.sum(prev_uncovered)
            
            nb_idx = neighbors[chosen_idx]
            # Count how many are currently uncovered
            num_covered = np.sum(uncovered[nb_idx])
            uncovered[nb_idx] = 0
            remaining_count -= num_covered
            
        return code

    # Build M matrix (2187x2187)
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for idx in range(total_words):
        M[idx, neighbors[idx]] = 1

    def local_search(code):
        # Convert to list of np.int32 for consistency? Or just list of ints.
        # Use NumPy for coverage updates
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1
        
        improved = True
        # Limit iterations to prevent infinite loops or excessive time
        max_iters = 1000
        iter_count = 0
        
        while improved and iter_count < max_iters:
            iter_count += 1
            improved = False
            
            # 1. Greedy Removal
            # Find indices where all neighbors have coverage > 1
            # For each codeword idx, check if min(coverage[neighbors[idx]]) > 1
            removable_indices = []
            for i, idx in enumerate(code):
                # Vectorized check
                if np.min(coverage[neighbors[idx]]) > 1:
                    removable_indices.append(i)
            
            if removable_indices:
                # Remove a random one
                rem_idx = random.choice(removable_indices)
                idx_to_remove = code[rem_idx]
                
                coverage[neighbors[idx_to_remove]] -= 1
                del code[rem_idx]
                improved = True
                continue
                
            # 2. Swaps
            # If no removal possible, try swaps.
            # For each codeword c, find critical neighbors (coverage == 1).
            # We want to find w such that all critical neighbors of c are in neighbors[w].
            # i.e., for all cnb in critical_neighbors, neighbor_mask[w, cnb] == True.
            
            swaps_made = 0
            # Shuffle code to avoid bias
            random.shuffle(code)
            
            for i in range(len(code)):
                c = code[i]
                critical_neighbors = neighbors[c][coverage[neighbors[c]] == 1]
                
                if len(critical_neighbors) == 0:
                    # Should have been removed
                    continue
                
                # We want to find w such that neighbor_mask[w, critical_neighbors] are all True.
                # This is equivalent to: np.all(neighbor_mask[:, critical_neighbors], axis=1)
                # But this creates a 2187 x len(critical) boolean array.
                # Since len(critical) is small (usually 1-3), this is fast.
                
                # Optimization: If len(critical) is 1, just find all w that cover that one neighbor.
                # If len(critical) > 1, intersect the sets.
                
                # Get candidates w that cover the first critical neighbor
                first_cnb = critical_neighbors[0]
                # w must be such that neighbor_mask[w, first_cnb] is True
                # This is equivalent to neighbors of first_cnb? No.
                # neighbor_mask[w, first_cnb] is True if first_cnb is in neighbors[w].
                # By symmetry of Hamming distance, this is equivalent to w in neighbors[first_cnb].
                # So candidates_w = neighbors[first_cnb]
                
                candidates_w = neighbors[first_cnb]
                
                if len(critical_neighbors) > 1:
                    # Filter candidates_w to those that cover all other critical neighbors
                    # For each w in candidates_w, check if all other critical_neighbors are in neighbors[w]
                    # i.e., neighbor_mask[w, other_crits] are all True.
                    
                    # Vectorized filter
                    # Create a boolean mask for valid w
                    # We can check: for each w in candidates_w, 
                    # does it cover all critical_neighbors?
                    
                    # Let's just iterate if the list is small, or use matrix ops.
                    # candidates_w is size 1+2n = 15.
                    # critical_neighbors is small.
                    
                    valid_w = []
                    for w in candidates_w:
                        if w == c:
                            continue
                        # Check if w covers all critical neighbors
                        # neighbor_mask[w, critical_neighbors] should be all True
                        if np.all(neighbor_mask[w, critical_neighbors]):
                            valid_w.append(w)
                    
                    if valid_w:
                        # Pick one
                        w = random.choice(valid_w)
                        
                        # Perform swap
                        coverage[neighbors[c]] -= 1
                        coverage[neighbors[w]] += 1
                        code[i] = w
                        improved = True
                        swaps_made += 1
                        break # Restart loop after swap
                else:
                    # Only one critical neighbor
                    # Any w in neighbors[first_cnb] (except c) works?
                    # Yes, because the only constraint is covering that one neighbor.
                    # But we must ensure w != c.
                    candidates_w_list = [w for w in candidates_w if w != c]
                    if candidates_w_list:
                        w = random.choice(candidates_w_list)
                        # Perform swap
                        coverage[neighbors[c]] -= 1
                        coverage[neighbors[w]] += 1
                        code[i] = w
                        improved = True
                        swaps_made += 1
                        break # Restart loop after swap
            
            if swaps_made > 0:
                continue # Restart while loop

        return code

    # Main Iterated Local Search
    start_time = time.time()
    best_code = None
    best_size = float('inf')
    seed = 42
    
    # Run multiple iterations
    while time.time() - start_time < 110:
        # Generate random greedy cover
        code = greedy_construct(seed)
        seed += 1
        
        # Local search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code is None:
        # Fallback: Standard greedy
        code = greedy_construct(0)
        best_code = code
        best_size = len(code)

    # Convert to list of lists
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
