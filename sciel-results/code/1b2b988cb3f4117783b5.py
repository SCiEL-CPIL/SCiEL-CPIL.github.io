import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # Build incidence matrix: coverage[i, j] = 1 if k_subset i covers t_subset j
    # Use numpy for vectorized operations
    # For v=19, k=8, t=5: num_k_subsets=75582, num_t_subsets=11628
    # Incidence matrix is 75582 x 11628 booleans -> ~878 MB, too large
    # Instead, use bitmask approach with numpy for gain computation

    # For each k-subset, store the list of t-subset indices it covers
    # This is more memory efficient and allows fast gain computation
    cover_lists = []
    for ks in k_subsets:
        covered = [t_subset_index[ts] for ts in itertools.combinations(ks, t)]
        cover_lists.append(covered)

    # Precompute for each t-subset, which k-subsets cover it (reverse index)
    # This helps in local search
    t_to_k = [[] for _ in range(num_t_subsets)]
    for i, covered in enumerate(cover_lists):
        for j in covered:
            t_to_k[j].append(i)

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')

    # Vectorized greedy using numpy
    def vectorized_greedy(uncovered_mask, used_mask, rng):
        """Greedy with numpy vectorization for gain computation."""
        current_blocks = []
        uncovered = uncovered_mask.copy()
        used = used_mask.copy()
        
        while uncovered.any():
            if time.time() > deadline:
                return None, None
            
            # Compute gain for each unused block
            # gain[i] = number of uncovered t-subsets covered by k_subset i
            # We need: for each block i, count how many of its covered t-subsets are still uncovered
            # This is: sum over j in cover_lists[i] of (uncovered[j])
            
            # Vectorized approach: create a matrix where row i has 1s at positions cover_lists[i]
            # Then gain[i] = sum(matrix[i] * uncovered)
            # But this matrix is sparse and large.
            
            # Alternative: use the fact that each block covers C(8,5)=56 t-subsets
            # We can precompute the coverage as a sparse structure
            
            # For speed, let's use a different approach:
            # Maintain an array gain[i] for each block
            # When we add a block, we decrement gain for all blocks that share covered t-subsets
            
            # Actually, let's just do the direct computation but with numpy
            # Create a boolean array for uncovered
            unc = uncovered
            
            # For each block, compute how many of its covered t-subsets are uncovered
            # Use np.array operations
            gains = np.zeros(num_k_subsets, dtype=np.int32)
            for i in range(num_k_subsets):
                if not used[i]:
                    # Count how many of cover_lists[i] are in uncovered
                    gains[i] = np.sum(unc[cover_lists[i]])
            
            # Find max gain among unused blocks
            max_gain = np.max(gains[~used])
            if max_gain <= 0:
                return None, None
            
            # Get all blocks with max gain
            candidates = np.where((gains == max_gain) & (~used))[0]
            chosen = rng.choice(candidates)
            
            current_blocks.append(k_subsets[chosen])
            used[chosen] = True
            # Update uncovered
            uncovered[cover_lists[chosen]] = False
        
        return current_blocks, used

    # Local search: try to swap one block for another to reduce count
    def local_search(blocks, rng, time_limit):
        """Try to improve the solution by removing one block and replacing with better ones."""
        current_blocks = [list(b) for b in blocks]
        current_count = len(current_blocks)
        
        # Build coverage map
        covered = np.zeros(num_t_subsets, dtype=bool)
        for b in current_blocks:
            for ts in itertools.combinations(b, t):
                covered[t_subset_index[ts]] = True
        
        if not covered.all():
            return current_blocks
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each block
            for idx in range(len(current_blocks)):
                if time.time() > time_limit:
                    break
                
                # Remove block idx
                removed = current_blocks[idx]
                temp_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                
                # Recompute coverage
                temp_covered = np.zeros(num_t_subsets, dtype=bool)
                for b in temp_blocks:
                    for ts in itertools.combinations(b, t):
                        temp_covered[t_subset_index[ts]] = True
                
                # Find uncovered t-subsets
                uncovered_idx = np.where(~temp_covered)[0]
                if len(uncovered_idx) == 0:
                    # We can remove this block!
                    current_blocks = temp_blocks
                    covered = temp_covered
                    current_count -= 1
                    improved = True
                    break
                
                # Try to find a single block that covers all uncovered t-subsets
                # This is unlikely, so try greedy replacement
                # Find blocks that cover at least one uncovered t-subset
                candidate_blocks = set()
                for uj in uncovered_idx:
                    candidate_blocks.update(t_to_k[uj])
                
                if not candidate_blocks:
                    continue
                
                # Greedily add blocks to cover all uncovered
                temp_uncovered = set(uncovered_idx.tolist())
                added = []
                while temp_uncovered and time.time() < time_limit:
                    best_block = None
                    best_gain = 0
                    for cb in candidate_blocks:
                        gain = 0
                        for ts_idx in cover_lists[cb]:
                            if ts_idx in temp_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = cb
                    if best_block is None or best_gain == 0:
                        break
                    added.append(k_subsets[best_block])
                    for ts_idx in cover_lists[best_block]:
                        temp_uncovered.discard(ts_idx)
                    candidate_blocks.discard(best_block)
                
                if not temp_uncovered and len(added) < 1:
                    # Replaced one block with zero (shouldn't happen since we removed one)
                    pass
                elif not temp_uncovered and len(added) < 1:
                    current_blocks = temp_blocks + added
                    current_count = len(current_blocks)
                    improved = True
                    break
                elif not temp_uncovered and len(added) < 1:
                    current_blocks = temp_blocks + added
                    current_count = len(current_blocks)
                    improved = True
                    break
        
        return current_blocks

    # Run multiple randomized greedy restarts
    num_restarts = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Reserve 10s for local search
            break
        
        rng = random.Random(restart)
        
        # Randomized greedy: shuffle the order of candidate selection
        uncovered = np.ones(num_t_subsets, dtype=bool)
        used = np.zeros(num_k_subsets, dtype=bool)
        current_blocks = []
        
        while uncovered.any():
            if time.time() > deadline - 5:
                break
            
            # Compute gains vectorized
            gains = np.zeros(num_k_subsets, dtype=np.int32)
            for i in range(num_k_subsets):
                if not used[i]:
                    gains[i] = np.sum(uncovered[cover_lists[i]])
            
            max_gain = np.max(gains[~used])
            if max_gain <= 0:
                break
            
            # Random tie-breaking
            candidates = np.where((gains == max_gain) & (~used))[0]
            chosen = rng.choice(candidates)
            
            current_blocks.append(k_subsets[chosen])
            used[chosen] = True
            uncovered[cover_lists[chosen]] = False
        
        if not uncovered.any() and len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
            # Try local search on the best solution found
            if time.time() < deadline - 5:
                ls_time_limit = min(deadline - 2, time.time() + 2)
                improved = local_search(best_blocks, rng, ls_time_limit)
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_blocks = improved

    # Fallback: deterministic greedy
    if best_blocks is None:
        uncovered = np.ones(num_t_subsets, dtype=bool)
        used = np.zeros(num_k_subsets, dtype=bool)
        current_blocks = []
        
        while uncovered.any():
            gains = np.zeros(num_k_subsets, dtype=np.int32)
            for i in range(num_k_subsets):
                if not used[i]:
                    gains[i] = np.sum(uncovered[cover_lists[i]])
            
            max_gain = np.max(gains[~used])
            if max_gain <= 0:
                break
            
            chosen = np.argmax(gains * (~used))
            current_blocks.append(k_subsets[chosen])
            used[chosen] = True
            uncovered[cover_lists[chosen]] = False
        
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
