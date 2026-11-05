import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Vectorized Greedy Initialization
    # We want to pick words that cover the most uncovered points.
    # To speed this up, we can precompute the neighbor masks as boolean arrays.
    # However, storing 6561 x 6561 boolean matrix is 43MB, which is fine.
    # Let's create a boolean adjacency matrix.
    # adj[i] is a boolean array of length N, True where neighbor.
    
    # This takes a bit of time but allows vectorized operations.
    adj_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        adj_matrix[i, neighbors[i]] = True
        
    start_time = time.time()
    time_limit = 115.0
    
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Vectorized Greedy
    # At each step, calculate gain for all candidates.
    # gain[i] = sum(uncovered & adj_matrix[i])
    # This is expensive if done naively. 
    # Instead, we can maintain the uncovered set and pick the best.
    # To make it faster, we can use a heuristic: 
    # Randomly sample a subset of candidates if the full set is too slow?
    # No, 6561 is small enough that a few iterations of full scan might be okay if optimized.
    # But the previous loop was slow. Let's try a batched approach or just accept the speed if we use numpy sums.
    
    # Actually, the previous loop was:
    # for i in range(N): cnt = np.sum(uncovered[neighbors[i]])
    # This is 6561 numpy sums per iteration.
    # If we use the adjacency matrix:
    # gains = np.sum(uncovered[None, :] & adj_matrix, axis=1)
    # This is a single vectorized operation.
    
    while np.any(uncovered):
        # Calculate gains for all candidates
        # We only care about candidates not already in C? 
        # Actually, adding a candidate already in C does nothing (gain 0 relative to new coverage? No, it covers same things).
        # So we can just pick the max gain. If max gain is 0, we are done.
        
        # Optimization: If the number of uncovered points is small, we can be smarter.
        # For now, let's use the vectorized gain calculation.
        # Note: `uncovered` is a boolean array.
        # `adj_matrix` is boolean.
        # `uncovered[None, :]` broadcasts uncovered to shape (1, N).
        # `& adj_matrix` gives a (N, N) boolean matrix where entry (i,j) is True if j is uncovered and i covers j.
        # `np.sum(..., axis=1)` gives the count of uncovered points covered by each candidate i.
        
        gains = np.sum(uncovered[None, :] & adj_matrix, axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() - start_time > time_limit:
            break
            
    C_set = set(C)
    
    # Compute coverage counts
    def compute_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
        
    def reduce_code(code, counts):
        """Remove redundant codewords."""
        if not code:
            return list(code), counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts
    
    C, counts = reduce_code(C, compute_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local search loop
    # We will use a more aggressive search:
    # 1. Try to remove any redundant word.
    # 2. Try to swap a word with another that allows removal of a third word or reduces size.
    
    while time.time() - start_time < time_limit:
        # Reduction pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        if len(C) == old_size:
            # No reduction. Try swaps.
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been removed
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                    
                if len(unique_points) > 3:
                    continue
                    
                # Find candidates covering all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                        
                if not candidates_set:
                    continue
                    
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                    
                c_prime = random.choice(list(candidates_set))
                
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                    
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    # Keep neutral swap to escape local minima
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
                    
            if not improved:
                # Random kicks
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    if not C:
                        break
                    c_idx = random.randint(0, len(C) - 1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand) - 1)]
                    if c_prime in C:
                        continue
                        
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                        
                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts
                        
    # Final reduction
    best_C, _ = reduce_code(best_C, compute_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
