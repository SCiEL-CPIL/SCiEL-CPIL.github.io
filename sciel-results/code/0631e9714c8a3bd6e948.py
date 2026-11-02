"""Optimized covering design search for C(24,7,3).

Strategy:
1. Fast greedy construction (multiple seeds).
2. Fast reduction (remove redundant blocks).
3. Local search: try to replace pairs of blocks with a better block.
"""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin (45s buffer)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute: for each point x, list of (t-1)-subsets containing x mapped to t-subset indices
    # More efficient: for each (t-1)-subset, list of points that can extend it
    t_minus_1_to_tsubsets = {}
    for idx, subset in enumerate(all_t_subsets):
        for i in range(t):
            rest = tuple(sorted(subset[:i] + subset[i+1:]))
            if rest not in t_minus_1_to_tsubsets:
                t_minus_1_to_tsubsets[rest] = []
            t_minus_1_to_tsubsets[rest].append(idx)
    
    def compute_gain_for_block(block, uncovered_set):
        """Compute which t-subsets this block covers."""
        covered = []
        for subset in itertools.combinations(block, t):
            idx = t_subset_to_idx[subset]
            covered.append(idx)
        return covered
    
    def greedy_cover_v2(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        uncovered_list = list(range(n_t_subsets))
        idx_to_pos = {i: i for i in range(n_t_subsets)}
        
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 25:
                return None
            
            # Pick a random uncovered t-subset
            pos = rng.randint(0, uncovered_count - 1)
            start_idx = uncovered_list[pos]
            
            # Remove start_idx from uncovered_list and idx_to_pos
            last_idx = uncovered_list[-1]
            uncovered_list[pos] = last_idx
            idx_to_pos[last_idx] = pos
            uncovered_list.pop()
            del idx_to_pos[start_idx]
            uncovered[start_idx] = False
            uncovered_count -= 1
            
            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            
            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 25:
                    return None
                
                block_set = set(block)
                best_x = -1
                best_gain = -1
                best_ties = []
                
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_tuple = tuple(sorted(rest))
                        if rest_tuple in t_minus_1_to_tsubsets:
                            for idx in t_minus_1_to_tsubsets[rest_tuple]:
                                if uncovered[idx]:
                                    gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_ties = [x]
                    elif gain == best_gain and gain > 0:
                        best_ties.append(x)
                
                if best_ties:
                    best_x = rng.choice(best_ties)
                elif best_x >= 0:
                    pass
                else:
                    break
            
            if len(block) < k:
                return None
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered: remove all t-subsets of the new block
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    # Remove idx from uncovered_list
                    if idx in idx_to_pos:
                        p = idx_to_pos[idx]
                        last_idx = uncovered_list[-1]
                        uncovered_list[p] = last_idx
                        idx_to_pos[last_idx] = p
                        uncovered_list.pop()
                        del idx_to_pos[idx]
        
        return blocks

    def reduce_blocks(blocks):
        """Remove redundant blocks with iterative passes."""
        if not blocks:
            return []
        
        blocks = list(blocks)
        n_blocks = len(blocks)
        
        # Compute coverage counts and for each t-subset, which blocks cover it
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        # For each block, store the list of t-subset indices it covers
        block_covers = []
        for block in blocks:
            covers = []
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                covers.append(idx)
                coverage_count[idx] += 1
            block_covers.append(covers)
        
        improved = True
        while improved:
            if time.time() > deadline - 15:
                break
            improved = False
            
            # Try removing each block
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                
                block = blocks[bi]
                can_remove = True
                affected_indices = []
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] -= 1
                    affected_indices.append(idx)
                    if coverage_count[idx] == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(bi)
                    block_covers.pop(bi)
                    improved = True
                    break
                else:
                    # Restore coverage counts
                    for a_idx in affected_indices:
                        coverage_count[a_idx] += 1
        
        return blocks

    def local_search_pair_swap(blocks, max_iterations=50):
        """Try to replace two blocks with one better block."""
        if len(blocks) < 2:
            return blocks
        
        blocks = list(blocks)
        n_blocks = len(blocks)
        
        # Precompute coverage
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        for block in blocks:
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                coverage_count[idx] += 1
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            if time.time() > deadline - 10:
                break
            improved = False
            iterations += 1
            
            # Try all pairs of blocks
            found_swap = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(blocks)):
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if removing blocks i and j leaves any t-subset uncovered
                    # First, count coverage from blocks i and j
                    temp_coverage = coverage_count.copy()
                    for subset in itertools.combinations(blocks[i], t):
                        idx = t_subset_to_idx[subset]
                        temp_coverage[idx] -= 1
                    for subset in itertools.combinations(blocks[j], t):
                        idx = t_subset_to_idx[subset]
                        temp_coverage[idx] -= 1
                    
                    # Find t-subsets that would become uncovered
                    uncovered_after_removal = []
                    for idx in range(n_t_subsets):
                        if temp_coverage[idx] == 0:
                            uncovered_after_removal.append(idx)
                    
                    if len(uncovered_after_removal) == 0:
                        # We can remove both blocks
                        blocks.pop(j)
                        blocks.pop(i)
                        coverage_count = temp_coverage
                        improved = True
                        found_swap = True
                        break
                    
                    # Try to find a single block that covers all uncovered_after_removal
                    # The new block must contain all points from these t-subsets
                    # Collect all points from uncovered t-subsets
                    all_points = set()
                    for idx in uncovered_after_removal:
                        all_points.update(all_t_subsets[idx])
                    
                    if len(all_points) > k:
                        continue
                    
                    # Try to find a k-block that covers all these t-subsets
                    # The block must contain all points in all_points
                    remaining_points = list(set(range(v)) - all_points)
                    needed = k - len(all_points)
                    
                    if needed < 0:
                        continue
                    
                    # Greedily pick remaining points to maximize coverage of uncovered_after_removal
                    new_block = list(all_points)
                    remaining_needed = needed
                    
                    for _ in range(remaining_needed):
                        if time.time() > deadline - 10:
                            break
                        if not remaining_points:
                            break
                        
                        best_x = -1
                        best_gain = -1
                        for x in remaining_points:
                            gain = 0
                            for idx in uncovered_after_removal:
                                if x in all_t_subsets[idx]:
                                    # Check if adding x helps cover this t-subset
                                    # We need all points of the t-subset to be in new_block
                                    new_block_set = set(new_block) | {x}
                                    if set(all_t_subsets[idx]).issubset(new_block_set):
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        
                        if best_x >= 0:
                            new_block.append(best_x)
                            remaining_points.remove(best_x)
                        else:
                            break
                    
                    if len(new_block) == k:
                        new_block.sort()
                        # Verify this block covers all uncovered_after_removal
                        new_block_set = set(new_block)
                        valid = True
                        for idx in uncovered_after_removal:
                            if not set(all_t_subsets[idx]).issubset(new_block_set):
                                valid = False
                                break
                        
                        if valid:
                            # Replace blocks i and j with new_block
                            blocks[i] = new_block
                            blocks.pop(j)
                            # Update coverage
                            coverage_count = temp_coverage
                            for subset in itertools.combinations(new_block, t):
                                idx = t_subset_to_idx[subset]
                                coverage_count[idx] += 1
                            improved = True
                            found_swap = True
                            break
                
                if found_swap:
                    break
            
            if not improved:
                break
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds for greedy construction
    seed = 0
    while time.time() < deadline - 30:
        result = greedy_cover_v2(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            # Apply local search
            searched = local_search_pair_swap(reduced, max_iterations=20)
            if len(searched) < best_count:
                best_count = len(searched)
                best_blocks = searched
        seed += 1
    
    if best_blocks is None:
        # Fallback: simple greedy without randomization to ensure completion
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        uncovered_list = list(range(n_t_subsets))
        idx_to_pos = {i: i for i in range(n_t_subsets)}
        blocks = []
        
        while uncovered_count > 0:
            start_idx = uncovered_list[0]
            
            last_idx = uncovered_list[-1]
            uncovered_list[0] = last_idx
            idx_to_pos[last_idx] = 0
            uncovered_list.pop()
            del idx_to_pos[start_idx]
            uncovered[start_idx] = False
            uncovered_count -= 1
            
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_tuple = tuple(sorted(rest))
                        if rest_tuple in t_minus_1_to_tsubsets:
                            for idx in t_minus_1_to_tsubsets[rest_tuple]:
                                if uncovered[idx]:
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x >= 0:
                    block.append(best_x)
                else:
                    break
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    if idx in idx_to_pos:
                        p = idx_to_pos[idx]
                        last_idx = uncovered_list[-1]
                        uncovered_list[p] = last_idx
                        idx_to_pos[last_idx] = p
                        uncovered_list.pop()
                        del idx_to_pos[idx]
        
        best_blocks = reduce_blocks(blocks)
    
    return best_blocks

# EVOLVE-BLOCK-END
