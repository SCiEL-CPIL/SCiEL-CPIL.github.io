import itertools
import numpy as np
from scipy.sparse import csr_matrix
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # 3. Build sparse coverage matrix
    # M[i, j] = 1 if block i covers t-subset j
    rows = []
    cols = []
    for b_idx, block in enumerate(k_blocks):
        # Find all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            s_idx = subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    data = np.ones(len(rows), dtype=np.int8)
    M = csr_matrix((data, (rows, cols)), shape=(n_k, n_t))
    
    # Precompute row indices for fast access
    M_coo = M.tocoo()
    # For each block, store the list of t-subset indices it covers
    block_coverage = [[] for _ in range(n_k)]
    for i in range(len(M_coo.row)):
        block_coverage[M_coo.row[i]].append(M_coo.col[i])
    
    # 4. Greedy construction function with GRASP-style randomization
    def greedy_construct(seed, alpha=0.5):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t, dtype=np.int8)
        blocks_used = []
        used_blocks_mask = np.zeros(n_k, dtype=bool)
        
        while np.any(uncovered):
            # Calculate gain for each block
            # M @ uncovered gives count of uncovered subsets covered by each block
            gains = M @ uncovered
            gains = gains.astype(np.float64)
            
            # Add overlap penalty: penalize blocks that overlap heavily with already used blocks
            # This is a simplified version of the "potential" mentioned in the directive
            # We can approximate overlap by looking at how many t-subsets are already covered
            # but since we track 'uncovered', the gain already accounts for this.
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            # Find candidate blocks: those with gain >= alpha * max_gain
            candidates = np.where(gains >= alpha * max_gain)[0]
            
            # Exclude already used blocks
            candidates = candidates[~used_blocks_mask[candidates]]
            
            if len(candidates) == 0:
                # Fallback to best available
                candidates = np.where(gains == max_gain)[0]
                candidates = candidates[~used_blocks_mask[candidates]]
                if len(candidates) == 0:
                    break
            
            # Pick one randomly from candidates
            best_block_idx = candidates[rng.integers(len(candidates))]
            
            blocks_used.append(best_block_idx)
            used_blocks_mask[best_block_idx] = True
            
            # Update uncovered
            uncovered[block_coverage[best_block_idx]] = 0
            
            # Check time
            if time.time() > deadline:
                break
        
        return [k_blocks[i] for i in blocks_used]
    
    # 5. Pruning function (1-swap local search / deletion)
    def prune_covering(blocks):
        if not blocks:
            return blocks
            
        # Count how many times each t-subset is covered
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[t_sub]
                cover_count[s_idx] += 1
                
        # Iterative deletion of redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                for t_sub in itertools.combinations(b, t):
                    s_idx = subset_to_idx[t_sub]
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    # Remove b, decrement counts
                    for t_sub in itertools.combinations(b, t):
                        s_idx = subset_to_idx[t_sub]
                        cover_count[s_idx] -= 1
                    changed = True
            blocks = new_blocks
            
        return blocks

    # 6. 1-swap local search: try to replace a block with another to improve coverage
    def local_search_1swap(blocks, seed):
        if not blocks:
            return blocks
            
        rng = np.random.default_rng(seed)
        
        # Build a set of blocks for fast lookup
        block_set = set(blocks)
        
        # Count coverage
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[t_sub]
                cover_count[s_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            blocks_list = list(blocks)
            
            # Try to find a swap: remove block i, add block j
            # such that coverage is maintained and we gain something (e.g., reduce overlap)
            # Since we minimize the number of blocks, a swap that keeps size constant but
            # allows further pruning is useful. Or a swap that directly reduces size.
            
            # A simpler 1-swap: try to replace a block with a different one that covers
            # the same unique t-subsets but potentially allows more pruning.
            
            # For now, let's just do a few random restarts with different alphas
            # and rely on the greedy + prune.
            
            # Actually, let's implement a simple heuristic:
            # Try to remove a block and see if we can cover its unique parts with one other block.
            for i in range(len(blocks_list)):
                if time.time() > deadline - 5.0:
                    break
                    
                b_i = blocks_list[i]
                
                # Find t-subsets uniquely covered by b_i
                unique_subs = []
                for t_sub in itertools.combinations(b_i, t):
                    s_idx = subset_to_idx[t_sub]
                    if cover_count[s_idx] == 1:
                        unique_subs.append(t_sub)
                        
                if not unique_subs:
                    # b_i is redundant, remove it
                    for t_sub in itertools.combinations(b_i, t):
                        s_idx = subset_to_idx[t_sub]
                        cover_count[s_idx] -= 1
                    blocks_list.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a block j that covers all unique_subs
                    # This is expensive, so we might skip it or do it sparsely
                    pass
                    
            blocks = blocks_list
            
        return blocks

    best_covering = None
    best_size = float('inf')
    
    # Run multiple greedy restarts with different alpha values
    alphas = [0.5, 0.7, 0.9, 1.0]
    
    for alpha in alphas:
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
                
            seed = i * 100 + int(alpha * 10)
            covering = greedy_construct(seed, alpha)
            covering = prune_covering(covering)
            
            if len(covering) < best_size:
                best_size = len(covering)
                best_covering = covering
                
    if best_covering is None:
        # Fallback to single greedy
        best_covering = greedy_construct(0, 1.0)
        best_covering = prune_covering(best_covering)
        
    return best_covering
