import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute block_triples: for each block, which triples it covers
    # c_k_t is the number of t-subsets in a k-set
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute coverage of each triple by blocks (inverse mapping)
    # This is a sparse matrix conceptually. We can use it for fast lookups.
    # triple_to_blocks: list of lists of block indices covering each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx].append(i)

    def greedy_cover(uncovered_init, use_weighted=False, random_tie=False, seed=None):
        """Run greedy set cover, optionally weighted."""
        if seed is not None:
            np.random.seed(seed)
        uncovered = uncovered_init.copy()
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                # Weighted gain: sum of weights of uncovered triples in block
                # Weights are not precomputed here, we use inverse frequency as a proxy
                # But for speed, unweighted is often better if we do many restarts
                gains = np.sum(uncovered[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
                
            # Find the block with maximum gain
            if random_tie:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                    
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage counting."""
        if not selected_indices:
            return selected_indices
            
        selected_set = set(selected_indices)
        remaining = list(selected_indices)
        
        # Compute coverage count for all triples
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in selected_indices:
            coverage_count[block_triples[block_idx]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            for block_idx in remaining:
                if time.time() > deadline:
                    return new_remaining + remaining[remaining.index(block_idx):]
                    
                block_triple_indices = block_triples[block_idx]
                
                # Check if all triples in this block have coverage > 1
                if np.all(coverage_count[block_triple_indices] > 1):
                    # Remove this block
                    coverage_count[block_triple_indices] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining

    def local_search_improve(solution_blocks, max_iterations=200):
        """Try to improve solution by swapping blocks (1-1 swaps and deletions)."""
        if not solution_blocks:
            return solution_blocks
            
        current_blocks = list(solution_blocks)
        
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in current_blocks:
            coverage_count[block_triples[block_idx]] += 1
            
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try removing each block (deletion)
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    return current_blocks
                    
                block_idx = current_blocks[i]
                block_triple_indices = block_triples[block_idx]
                
                # Check if all triples have coverage > 1
                if np.all(coverage_count[block_triple_indices] > 1):
                    # Remove block
                    coverage_count[block_triple_indices] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart search after removal
            
            # If no deletion possible, try 1-1 swap
            if not improved:
                for i in range(len(current_blocks)):
                    if time.time() > deadline:
                        return current_blocks
                        
                    block_idx = current_blocks[i]
                    block_triple_indices = block_triples[block_idx]
                    
                    # Temporarily remove
                    for triple_idx in block_triple_indices:
                        coverage_count[triple_idx] -= 1
                    
                    # Find uncovered triples
                    uncovered_mask = coverage_count == 0
                    if not np.any(uncovered_mask):
                        # No uncovered, we can keep the removal (this is a deletion, handled above, but just in case)
                        current_blocks.pop(i)
                        improved = True
                        break
                    
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    # Find a block that covers all uncovered triples
                    best_new_block = -1
                    best_extra = float('inf')
                    
                    # Optimization: Only check blocks that cover at least one of the uncovered triples
                    # But we need to cover ALL of them.
                    # A necessary condition is that the new block must be in the intersection of triple_to_blocks for all uncovered triples.
                    # Let's just iterate over all blocks for simplicity, but break early if possible.
                    # For v=24, n_blocks is 346k. This loop is heavy.
                    
                    # Better: Filter candidates.
                    # A candidate block must cover ALL uncovered triples.
                    # Let's pick the rarest uncovered triple and only check blocks that cover it.
                    # Then verify it covers the others.
                    
                    # Find the uncovered triple with the fewest covering blocks
                    min_freq = float('inf')
                    min_triple = -1
                    for t_idx in uncovered_indices:
                        freq = len(triple_to_blocks[t_idx])
                        if freq < min_freq:
                            min_freq = freq
                            min_triple = t_idx
                            
                    if min_triple == -1:
                        # Should not happen if uncovered_mask has any True
                        for triple_idx in block_triple_indices:
                            coverage_count[triple_idx] += 1
                        continue

                    candidates = triple_to_blocks[min_triple]
                    
                    for new_block_idx in candidates:
                        if new_block_idx in current_blocks:
                            continue
                        
                        new_block_triples = block_triples[new_block_idx]
                        
                        # Check if all uncovered triples are covered
                        all_covered = True
                        for unc_idx in uncovered_indices:
                            # Check if unc_idx is in new_block_triples
                            # Since new_block_triples is small (35 elements), linear search is ok
                            if unc_idx not in new_block_triples:
                                all_covered = False
                                break
                                
                        if not all_covered:
                            continue
                        
                        # Count how many new triples this would cover (that are currently uncovered)
                        # Actually, we just need to find *a* block. 
                        # To minimize future regret, we might prefer one that covers more "hard" triples, 
                        # but for 1-1 swap, any valid swap reduces size by 0 (swap) or 1 (deletion).
                        # Wait, a swap keeps the size the same. A deletion reduces it by 1.
                        # The goal of local search is to reach a state where a deletion is possible.
                        # So we just need *any* valid swap that might lead to a deletion later.
                        # Or, if we are doing "1-1 swap" to *maintain* coverage while changing the set, 
                        # we are looking for a path to a smaller set.
                        
                        # Actually, if we swap B_i with B_j, the size remains the same.
                        # We only reduce size if we can delete a block.
                        # So the strategy is: 
                        # 1. Try to delete.
                        # 2. If not, try to swap to a "better" configuration that allows deletion.
                        # What is "better"? Hard to define. 
                        # Random swaps are often effective.
                        
                        best_new_block = new_block_idx
                        break # Take the first valid swap found
                    
                    if best_new_block != -1:
                        # Perform the swap
                        for triple_idx in block_triples[best_new_block]:
                            coverage_count[triple_idx] += 1
                        current_blocks.pop(i)
                        current_blocks.append(best_new_block)
                        improved = True
                        break
                    else:
                        # Restore coverage
                        for triple_idx in block_triple_indices:
                            coverage_count[triple_idx] += 1
                        continue
                else:
                    pass
        
        return current_blocks

    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        pruned = local_search_improve(pruned)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 2: Weighted greedy (using inverse frequency as weight)
    # We need to precompute weights for this
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    def greedy_cover_weighted(uncovered_init, random_tie=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        uncovered = uncovered_init.copy()
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            
            if random_tie:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                    
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected

    selected = greedy_cover_weighted(np.ones(n_triples, dtype=bool), random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        pruned = local_search_improve(pruned)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
            
    # Strategy 3-20: Random tie-breaking restarts
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
            
        use_weighted = (restart % 2 == 0)
        seed = restart
        if use_weighted:
            selected = greedy_cover_weighted(np.ones(n_triples, dtype=bool), random_tie=True, seed=seed)
        else:
            selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=True, seed=seed)
            
        if selected:
            pruned = prune_blocks(selected)
            pruned = local_search_improve(pruned)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(all_blocks[i]) for i in pruned]
                
    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in all_blocks[:100]]
        
    return best_solution
