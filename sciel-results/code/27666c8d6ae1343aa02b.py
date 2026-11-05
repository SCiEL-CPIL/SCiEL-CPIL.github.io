import itertools
import random
import sys
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    M = len(all_t_subsets)  # 11628 for v=19, t=5
    
    # Precompute all k-blocks and their coverage bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)  # 75582 for v=19, k=8
    
    # For each block, compute which t-subsets it covers as a bitmask (Python int)
    block_mask = [0] * B
    # Also: for each t-subset, which blocks cover it
    subset_blocks = [[] for _ in range(M)]
    
    for bi, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            subset_blocks[idx].append(bi)
        block_mask[bi] = mask
    
    # Greedy to get initial solution
    def greedy_cover():
        uncovered = (1 << M) - 1
        blocks = []
        while uncovered:
            best_block = -1
            best_gain = -1
            # Pick an uncovered t-subset to guide selection
            # Find lowest set bit
            lsb = uncovered & (-uncovered)
            idx = lsb.bit_length() - 1
            # Try all blocks covering this t-subset
            for bi in subset_blocks[idx]:
                gain = (block_mask[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            blocks.append(all_blocks[best_block])
            uncovered &= ~block_mask[best_block]
        return blocks
    
    # Get initial greedy solution
    initial_blocks = greedy_cover()
    best_blocks = initial_blocks
    best_len = len(initial_blocks)
    
    # Reduction: remove redundant blocks
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        # Count coverage
        coverage_counts = [0] * M
        block_indices = []
        for b in blocks:
            # Find which precomputed block this is
            b_tuple = tuple(b)
            # We need to map back. Let's use a dict
            bi = block_to_idx.get(b_tuple)
            if bi is None:
                # Should not happen
                continue
            block_indices.append(bi)
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                coverage_counts[idx] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(block_indices) - 1, -1, -1):
                bi = block_indices[i]
                can_remove = True
                for ts in itertools.combinations(all_blocks[bi], t):
                    idx = t_subset_to_idx[ts]
                    coverage_counts[idx] -= 1
                    if coverage_counts[idx] == 0:
                        can_remove = False
                        break
                if can_remove:
                    block_indices.pop(i)
                    improved = True
                else:
                    for ts in itertools.combinations(all_blocks[bi], t):
                        idx = t_subset_to_idx[ts]
                        coverage_counts[idx] += 1
        
        return [all_blocks[bi] for bi in block_indices]
    
    # Create mapping from block tuple to index
    block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    
    best_blocks = reduce_blocks(best_blocks)
    best_len = len(best_blocks)
    
    # LP-based lower bound using dual simplex approach
    # The dual of set cover LP: max sum(y_i) s.t. sum_{j: i in S_j} y_j <= 1 for all blocks j, y_i >= 0
    # We use a fast greedy approximation to the dual for pruning
    
    def lp_lower_bound(uncovered_mask, depth):
        """Compute LP-based lower bound for remaining uncovered set."""
        if uncovered_mask == 0:
            return 0
        
        # Get list of uncovered t-subsets
        uncovered_list = []
        temp = uncovered_mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            uncovered_list.append(idx)
            temp ^= lsb
        
        if not uncovered_list:
            return 0
        
        # Simple LP dual bound: for each uncovered subset, count how many 
        # blocks can cover it, use min over subsets of (1 / max_blocks_covering)
        # This is a valid lower bound since each block covers at most 1 of these
        
        # Better: use fractional packing - assign weights to uncovered subsets
        # such that for each block, sum of weights of subsets it covers <= 1
        # Then total weight is a lower bound
        
        # Fast heuristic: greedy dual
        # Start with y_i = 0 for all uncovered i
        # Repeatedly: find block that maximizes sum of y_i over uncovered i it covers
        # Normalize and update
        
        # Even faster: use the fact that each block covers C(8,5)=56 subsets
        # So a crude bound is ceil(len(uncovered_list) / 56)
        crude_bound = (len(uncovered_list) + 55) // 56
        
        # Try to get a better bound using greedy dual
        # This is expensive, so only do it for small uncovered sets
        if len(uncovered_list) > 2000:
            return crude_bound
        
        # Greedy dual: maintain y[i] for uncovered indices
        y = [0.0] * M
        # Map from uncovered idx to position
        unc_set = set(uncovered_list)
        
        # For each block, compute which uncovered subsets it covers
        # And the current sum of y over those subsets
        
        # This is still expensive. Use a simpler approach:
        # For each uncovered subset, find the minimum number of blocks needed
        # considering overlap
        
        # Alternative: use the primal-dual gap
        # For now, return a conservative bound
        return crude_bound
    
    # Improved B&B with better branching and pruning
    best_len_bnb = best_len
    best_blocks_bnb = best_blocks
    nodes_explored = 0
    
    def bb_search(uncovered_mask, current_blocks, depth):
        nonlocal best_len_bnb, best_blocks_bnb, nodes_explored
        
        if time.time() - start_time > time_limit:
            return
        
        nodes_explored += 1
        
        if uncovered_mask == 0:
            if len(current_blocks) < best_len_bnb:
                best_len_bnb = len(current_blocks)
                best_blocks_bnb = list(current_blocks)
            return
        
        # Lower bound using greedy
        lb = 0
        remaining = uncovered_mask
        while remaining and lb < best_len_bnb - len(current_blocks):
            lsb = remaining & (-remaining)
            idx = lsb.bit_length() - 1
            best_gain = 0
            for bi in subset_blocks[idx]:
                gain = (block_mask[bi] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
            if best_gain == 0:
                return
            lb += 1
            best_bi = -1
            for bi in subset_blocks[idx]:
                gain = (block_mask[bi] & remaining).bit_count()
                if gain == best_gain:
                    best_bi = bi
                    break
            remaining &= ~block_mask[best_bi]
        
        if len(current_blocks) + lb >= best_len_bnb:
            return
        
        # Pick an uncovered t-subset with minimum number of covering blocks (MRV heuristic)
        lsb = uncovered_mask & (-uncovered_mask)
        idx = lsb.bit_length() - 1
        
        # Get candidate blocks covering this t-subset, sorted by coverage
        candidates = []
        for bi in subset_blocks[idx]:
            gain = (block_mask[bi] & uncovered_mask).bit_count()
            if gain > 0:
                candidates.append((gain, bi))
        
        if not candidates:
            return
        
        # Sort by gain descending
        candidates.sort(reverse=True)
        
        # Limit branching factor for speed
        max_branch = 8
        for gain, bi in candidates[:max_branch]:
            new_mask = uncovered_mask & ~block_mask[bi]
            current_blocks.append(all_blocks[bi])
            bb_search(new_mask, current_blocks, depth + 1)
            current_blocks.pop()
            
            if time.time() - start_time > time_limit:
                return
    
    # Start B&B from greedy solution's uncovered (should be 0, so start from scratch)
    full_mask = (1 << M) - 1
    bb_search(full_mask, [], 0)
    
    if best_len_bnb < best_len:
        best_blocks = best_blocks_bnb
        best_len = best_len_bnb
    
    return best_blocks
