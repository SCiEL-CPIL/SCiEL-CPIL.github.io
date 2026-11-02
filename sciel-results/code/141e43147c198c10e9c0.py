import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}

    # Precompute all k-subsets and their coverage of t-subsets
    all_k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(all_k_subsets)

    # cover_matrix[i] contains the indices of t-subsets covered by all_k_subsets[i]
    # Shape: (n_k_subsets, C(k,t))
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)

    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triple_to_idx[triple]

    # Block to index mapping
    block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}

    def fast_greedy_cover(seed, weights=None):
        """
        Vectorized greedy set cover with optional weights for tie-breaking.
        If weights is None, uses uniform random tie-breaking.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Vectorized gain calculation
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if weights is not None and len(candidates) > 1:
                # Weighted random selection
                cand_weights = weights[candidates]
                total_w = np.sum(cand_weights)
                if total_w > 0:
                    probs = cand_weights / total_w
                    # Random choice with probabilities
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    chosen_pos = np.searchsorted(cumsum, r)
                    if chosen_pos >= len(candidates):
                        chosen_pos = len(candidates) - 1
                    chosen_idx = candidates[chosen_pos]
                else:
                    chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def build_coverage_map(blocks):
        """Build coverage count array for a list of blocks."""
        if not blocks:
            return np.zeros(n_triples, dtype=np.int32)
        coverage = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            idx = block_to_k_idx[b]
            coverage[cover_matrix[idx]] += 1
        return coverage

    def build_coverage_map_vec(block_indices):
        """Build coverage count array from block indices (vectorized)."""
        if len(block_indices) == 0:
            return np.zeros(n_triples, dtype=np.int32)
        coverage = np.zeros(n_triples, dtype=np.int32)
        for idx in block_indices:
            coverage[cover_matrix[idx]] += 1
        return coverage

    def local_search_remove_redundant(blocks):
        """Remove redundant blocks (those with no unique triples)."""
        if not blocks:
            return blocks
        
        coverage = build_coverage_map(blocks)
        changed = True
        
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                # Check if this block has any unique triples
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                coverage = build_coverage_map(new_blocks)
                blocks = new_blocks
                changed = True
        
        return blocks

    def local_search_remove_redundant_vec(block_indices):
        """Remove redundant block indices (those with no unique triples). Returns new list of indices."""
        if not block_indices:
            return block_indices
        
        coverage = build_coverage_map_vec(block_indices)
        changed = True
        
        while changed:
            changed = False
            to_remove = []
            
            for i, idx in enumerate(block_indices):
                # Check if this block has any unique triples
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_indices = [idx for i, idx in enumerate(block_indices) if i not in remove_set]
                coverage = build_coverage_map_vec(new_indices)
                block_indices = new_indices
                changed = True
        
        return block_indices

    def swap_local_search(blocks, max_iter=50):
        """
        Try to swap blocks to enable further removals.
        For each block with unique triples, try to find a replacement
        that covers those unique triples and might allow removing others.
        """
        if not blocks:
            return blocks
        
        # Work with indices for efficiency
        block_indices = [block_to_k_idx[b] for b in blocks]
        coverage = build_coverage_map_vec(block_indices)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if time.time() > deadline - 10:
                break
            
            # Recompute coverage
            coverage = build_coverage_map_vec(block_indices)
            
            for i, idx in enumerate(block_indices):
                unique_triples_in_block = cover_matrix[idx][coverage[cover_matrix[idx]] == 1]
                
                if len(unique_triples_in_block) == 0:
                    continue
                
                # Find candidate blocks that cover all unique_triples_in_block
                # Use structured approach: find blocks containing the first unique triple,
                # then filter for those covering all unique triples
                first_triple = unique_triples_in_block[0]
                
                # Find all block indices that contain this triple
                # cover_matrix has shape (n_k_subsets, c_k_t)
                # A block covers triple `first_triple` if first_triple is in cover_matrix[i]
                # We can check: np.isin(first_triple, cover_matrix[i])
                # For efficiency, precompute which blocks cover each triple?
                # That's too large. Instead, sample candidates.
                
                # Try structured candidates: blocks that share many elements with current block
                # plus random ones
                current_block = all_k_subsets[idx]
                current_set = set(current_block)
                
                # Generate candidates: blocks that overlap heavily with current block
                # or cover the unique triples
                tried = set()
                candidates_to_try = []
                
                # First, try blocks that contain all points of current block except one
                # (these are likely to cover similar triples)
                for pt in current_set:
                    for other_pt in all_points:
                        if other_pt == pt or other_pt in current_set:
                            continue
                        new_block = tuple(sorted((current_set - {pt}) | {other_pt}))
                        if new_block in block_to_k_idx:
                            ci = block_to_k_idx[new_block]
                            if ci not in tried and ci not in block_indices:
                                tried.add(ci)
                                candidates_to_try.append(ci)
                
                # Add random candidates
                for _ in range(50):
                    ci = random.randint(0, n_k_subsets - 1)
                    if ci not in tried and ci not in block_indices:
                        tried.add(ci)
                        candidates_to_try.append(ci)
                
                for cand_idx in candidates_to_try:
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if cand_block covers all unique_triples_in_block
                    cand_covered = cover_matrix[cand_idx]
                    if not np.all(np.isin(unique_triples_in_block, cand_covered)):
                        continue
                    
                    # Try the swap: replace block i with cand_idx
                    new_indices = block_indices[:i] + [cand_idx] + block_indices[i+1:]
                    new_coverage = build_coverage_map_vec(new_indices)
                    
                    # Check if this swap allows removing at least one block
                    removable = 0
                    for j, nb_idx in enumerate(new_indices):
                        if j == i:
                            continue
                        if not np.any(new_coverage[cover_matrix[nb_idx]] == 1):
                            removable += 1
                    
                    if removable >= 1:
                        # Perform the swap and removal
                        new_indices = local_search_remove_redundant_vec(new_indices)
                        block_indices = new_indices
                        improved = True
                        break
                
                if improved:
                    break
        
        return [all_k_subsets[idx] for idx in block_indices]

    best_blocks = None
    best_len = float('inf')
    
    # Strategy 1: Multiple greedy restarts with different seeds
    restart_count = 0
    while time.time() < deadline - 30:
        blocks = fast_greedy_cover(restart_count)
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
        
        restart_count += 1
    
    # Strategy 2: Swap-based local search on best solution
    if best_blocks is not None and time.time() < deadline - 15:
        polished = swap_local_search(best_blocks, max_iter=100)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished[:]
    
    # Strategy 3: More greedy restarts if time allows
    while time.time() < deadline - 5:
        blocks = fast_greedy_cover(restart_count)
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
        
        restart_count += 1
    
    # Final polish: try swap local search on best
    if best_blocks is not None and time.time() < deadline - 3:
        polished = swap_local_search(best_blocks, max_iter=50)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished[:]
    
    return [list(b) for b in best_blocks] if best_blocks is not None else []
