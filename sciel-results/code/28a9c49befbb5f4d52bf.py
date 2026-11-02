import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute coverage masks: each block covers which t-subsets
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    
    # Build cover_matrix: n_k x c_k_t, storing t-subset indices
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, ks in enumerate(all_k_subsets):
        for j, ts in enumerate(itertools.combinations(ks, t)):
            cover_matrix[i, j] = t_to_idx[ts]
    
    # For fast subset checking: precompute for each block a frozenset of its t-subset indices
    block_t_sets = [frozenset(cover_matrix[i]) for i in range(n_k)]
    
    # For reverse lookup: which blocks cover each t-subset
    # This is expensive to build fully, so we'll compute on demand with caching
    t_to_blocks_cache = {}
    
    def get_blocks_for_t_subset(ts_idx):
        """Get all block indices that cover a given t-subset index."""
        if ts_idx not in t_to_blocks_cache:
            # Find all blocks whose t-set contains ts_idx
            # cover_matrix has shape (n_k, c_k_t)
            # For each block i, check if ts_idx is in cover_matrix[i]
            # This is slow for large n_k, so we precompute a sparse representation
            t_to_blocks_cache[ts_idx] = None  # placeholder
            
            # Build by scanning (expensive but we cache)
            blocks = []
            for i in range(n_k):
                if ts_idx in cover_matrix[i]:
                    blocks.append(i)
            t_to_blocks_cache[ts_idx] = blocks
        return t_to_blocks_cache[ts_idx]
    
    # To speed up, let's precompute the reverse index properly
    # For each t-subset index, list of blocks that contain it
    print("Building reverse index...")
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(c_k_t):
            ts_idx = cover_matrix[i, j]
            t_to_blocks[ts_idx].append(i)
    # Convert to numpy arrays for speed
    t_to_blocks_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                           for lst in t_to_blocks]
    print("Reverse index built.")
    
    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover(seed):
        """
        Vectorized greedy set cover with randomized tie-breaking.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Gains: number of uncovered t-subsets covered by each block
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(chosen_idx)
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks
    
    def remove_redundant(block_indices, time_limit=None):
        """
        Remove redundant blocks. A block is redundant if all its t-subsets
        are covered by at least one other block in the set.
        """
        if time_limit is None:
            time_limit = deadline - 1.0
        
        if not block_indices:
            return block_indices
        
        current = list(block_indices)
        changed = True
        
        while changed and time.time() < time_limit:
            changed = False
            
            # Build coverage count
            coverage = np.zeros(n_t, dtype=np.int32)
            for b_idx in current:
                coverage[cover_matrix[b_idx]] += 1
            
            to_remove = []
            for b_idx in current:
                # Check if any t-subset covered by b_idx is uniquely covered by b_idx
                if np.any(coverage[cover_matrix[b_idx]] == 1):
                    continue  # Not redundant
                else:
                    to_remove.append(b_idx)
            
            if to_remove:
                remove_set = set(to_remove)
                current = [b for b in current if b not in remove_set]
                changed = True
        
        return current
    
    def local_search_swap(block_indices, time_limit=None):
        """
        Try to improve coverage by swapping blocks.
        For each block, try to replace it with another block that covers
        its unique subsets, then try to remove any newly redundant blocks.
        """
        if time_limit is None:
            time_limit = deadline - 1.0
        
        if not block_indices:
            return block_indices
        
        current = list(block_indices)
        n_blocks = len(current)
        
        if n_blocks < 2:
            return current
        
        # Build coverage
        def build_coverage(blks):
            cov = np.zeros(n_t, dtype=np.int32)
            for b in blks:
                cov[cover_matrix[b]] += 1
            return cov
        
        coverage = build_coverage(current)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            coverage = build_coverage(current)
            
            for i, b_idx in enumerate(current):
                if time.time() > time_limit:
                    break
                
                # Find unique subsets of b_idx
                unique_mask = (coverage == 1)
                b_subsets = cover_matrix[b_idx]
                unique_in_b = b_subsets[unique_mask[b_subsets]]
                
                if len(unique_in_b) == 0:
                    # b_idx is redundant, should have been removed
                    continue
                
                # Pick the "hardest" unique subset (one covered by fewest other blocks)
                # Actually, we need a replacement that covers ALL unique subsets
                # Start with the subset that has the fewest covering blocks
                min_count = float('inf')
                hardest = None
                for u in unique_in_b:
                    cnt = len(t_to_blocks_arrays[u])
                    if cnt < min_count:
                        min_count = cnt
                        hardest = u
                
                # Get candidates: blocks that cover the hardest subset
                candidates = t_to_blocks_arrays[hardest]
                
                # Filter to blocks not already in current
                current_set = set(current)
                candidates = [c for c in candidates if c not in current_set]
                
                # Shuffle for diversity
                random.shuffle(candidates)
                
                # Try each candidate
                for c_idx in candidates:
                    if time.time() > time_limit:
                        break
                    
                    # Check if c_idx covers all unique subsets of b_idx
                    c_subsets = block_t_sets[c_idx]
                    if all(u in c_subsets for u in unique_in_b):
                        # Try the swap
                        new_current = current.copy()
                        new_current[i] = c_idx
                        
                        # Check validity: all t-subsets still covered
                        new_cov = build_coverage(new_current)
                        if np.all(new_cov >= 1):
                            # Valid swap! Now try to remove redundancies
                            new_current = remove_redundant(new_current, time_limit)
                            
                            if len(new_current) < len(current):
                                current = new_current
                                improved = True
                                break
                            elif len(new_current) == len(current):
                                # Same size, but might enable further improvements
                                # Accept with some probability for diversity
                                if random.random() < 0.3:
                                    current = new_current
                                improved = True
                                break
                            
        return current
    
    def hill_climb(block_indices, time_limit=None):
        """
        More aggressive local search: try random perturbations and accept
        if they lead to improvement after redundancy removal.
        """
        if time_limit is None:
            time_limit = deadline - 1.0
        
        if not block_indices:
            return block_indices
        
        current = list(block_indices)
        best = current[:]
        best_len = len(best)
        
        while time.time() < time_limit:
            # Try a random perturbation
            if not current:
                break
            
            # Option 1: Randomly replace one block
            if random.random() < 0.5 and len(current) > 0:
                i = random.randint(0, len(current) - 1)
                new_block = random.randint(0, n_k - 1)
                perturbed = current.copy()
                perturbed[i] = new_block
            else:
                # Option 2: Add a random block, then remove redundancies
                new_block = random.randint(0, n_k - 1)
                if new_block not in current:
                    perturbed = current + [new_block]
                else:
                    continue
            
            # Remove redundancies from perturbed
            cleaned = remove_redundant(perturbed, time.time() + 1)
            
            if len(cleaned) < best_len:
                best = cleaned
                best_len = len(best)
                current = cleaned
            elif len(cleaned) == len(best) and random.random() < 0.2:
                current = cleaned
        
        return best
    
    # Main loop
    restart_count = 0
    total_restarts_budget = 20  # We'll do at most this many full restarts
    
    while time.time() < deadline - 15 and restart_count < total_restarts_budget:
        seed = restart_count * 7 + 13
        
        # 1. Greedy construction
        block_indices = fast_greedy_cover(seed)
        
        if not block_indices:
            restart_count += 1
            continue
        
        # 2. Remove redundancies (quick pass)
        pruned = remove_redundant(block_indices, time.time() + 2)
        
        # 3. Local search swap
        swapped = local_search_swap(pruned, time.time() + 5)
        
        # 4. Hill climb
        final = hill_climb(swapped, time.time() + 5)
        
        if len(final) < best_len:
            best_len = len(final)
            best_blocks = final[:]
        
        restart_count += 1
    
    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 2:
        # More aggressive local search
        polished = local_search_swap(best_blocks, deadline - 1)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished
        
        # Hill climb with more time
        if time.time() < deadline - 1:
            final_polished = hill_climb(best_blocks, deadline - 0.5)
            if len(final_polished) < best_len:
                best_len = len(final_polished)
                best_blocks = final_polished
    
    # Convert back to list of lists of integers
    if best_blocks is not None:
        return [list(all_k_subsets[b]) for b in best_blocks]
    else:
        return []
