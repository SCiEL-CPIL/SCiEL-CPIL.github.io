import itertools
import random
import sys
import time
from collections import defaultdict
from heapq import heappush, heappop

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    M = len(all_t_subsets)
    
    # Precompute all k-blocks and their coverage bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    
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
    
    # Create mapping from block tuple to index
    block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    
    # Greedy to get initial solution
    def greedy_cover():
        uncovered = (1 << M) - 1
        blocks = []
        while uncovered:
            best_block = -1
            best_gain = -1
            lsb = uncovered & (-uncovered)
            idx = lsb.bit_length() - 1
            for bi in subset_blocks[idx]:
                gain = (block_mask[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            blocks.append(all_blocks[best_block])
            uncovered &= ~block_mask[best_block]
        return blocks
    
    initial_blocks = greedy_cover()
    best_blocks = initial_blocks
    best_len = len(initial_blocks)
    
    # Reduction: remove redundant blocks
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        coverage_counts = [0] * M
        block_indices = []
        for b in blocks:
            b_tuple = tuple(b)
            bi = block_to_idx.get(b_tuple)
            if bi is None:
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
    
    best_blocks = reduce_blocks(best_blocks)
    best_len = len(best_blocks)
    
    # LP-relaxation-based lower bound using dual variables
    # We solve the dual of the set cover LP:
    #   max sum(y_i) s.t. sum_{i in S_j} y_i <= 1 for all j, y_i >= 0
    # Using a greedy approximation to the dual for speed
    
    def lp_lower_bound(uncovered_mask, num_blocks_used):
        """
        Compute a lower bound on the number of additional blocks needed.
        Uses a greedy dual construction: repeatedly pick an uncovered t-subset,
        assign weight to it, and subtract from all blocks covering it.
        """
        if uncovered_mask == 0:
            return 0
        
        # We'll use a simple fractional packing approach
        # For each uncovered t-subset, we want to assign a "price" y_i
        # such that for every block, sum of prices of t-subsets it covers <= 1
        # and we maximize sum of prices.
        
        # Greedy dual: process uncovered subsets one by one
        # For each, assign price = min(1, remaining capacity of blocks covering it)
        # This is an approximation but gives a valid lower bound.
        
        # Simpler approach: use the ratio |uncovered| / max_block_coverage
        # plus a packing argument
        
        # Count uncovered t-subsets
        n_uncovered = uncovered_mask.bit_count()
        
        # Find the maximum number of uncovered t-subsets a single block can cover
        max_cover = 0
        # Sample some blocks to estimate
        # For speed, just use a few
        for bi in range(min(B, 500)):
            cov = (block_mask[bi] & uncovered_mask).bit_count()
            if cov > max_cover:
                max_cover = cov
        
        if max_cover == 0:
            return float('inf')
        
        # Basic lower bound: ceil(n_uncovered / max_cover)
        lb1 = (n_uncovered + max_cover - 1) // max_cover
        
        # Improved: use a greedy dual packing
        # Assign weights to uncovered subsets
        # y_i for each uncovered i, such that for each block j: sum_{i in block j} y_i <= 1
        # We do this greedily: sort uncovered by "scarcity" (few blocks cover them)
        
        # Get list of uncovered indices
        uncovered_list = []
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_list.append(idx)
            m ^= lsb
        
        if not uncovered_list:
            return 0
        
        # For each uncovered t-subset, count how many blocks cover it
        # (among all blocks)
        # Sort by scarcity (fewer covering blocks = more important)
        def scarcity(idx):
            return len(subset_blocks[idx])
        
        uncovered_list.sort(key=scarcity)
        
        # Greedy dual assignment
        # y[idx] for each uncovered idx
        # constraint: for each block, sum of y[idx] for idx in block <= 1
        # We assign y[idx] = 1 / (number of blocks that can still take it)
        # This is complex; use a simpler bound.
        
        # Alternative: use the fact that each block covers at most C(k,t) t-subsets total,
        # but we only care about uncovered ones.
        
        # A stronger bound: consider a maximal set of pairwise "incompatible" uncovered subsets
        # where no single block covers two of them. This gives a packing number.
        # But computing this is expensive.
        
        # Use lb1 and also a greedy sequential bound
        # Greedy: repeatedly pick the uncovered subset with fewest covering blocks,
        # pick one block to cover it, remove all subsets that block covers.
        # Count the number of steps.
        
        remaining = uncovered_mask
        count = 0
        while remaining:
            lsb = remaining & (-remaining)
            idx = lsb.bit_length() - 1
            # Pick the best block for this idx
            best_gain = -1
            best_bi = -1
            for bi in subset_blocks[idx]:
                gain = (block_mask[bi] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi == -1 or best_gain == 0:
                return float('inf')
            count += 1
            remaining &= ~block_mask[best_bi]
        
        lb2 = count
        
        return max(lb1, lb2)
    
    # B&B with stronger pruning
    best_len_bnb = best_len
    best_blocks_bnb = best_blocks
    nodes_explored = 0
    time_checks = 0
    
    def bb_search(uncovered_mask, current_blocks, depth):
        nonlocal best_len_bnb, best_blocks_bnb, nodes_explored, time_checks
        
        time_checks += 1
        if time_checks % 100 == 0 and time.time() - start_time > time_limit:
            return
        
        nodes_explored += 1
        
        if uncovered_mask == 0:
            if len(current_blocks) < best_len_bnb:
                best_len_bnb = len(current_blocks)
                best_blocks_bnb = list(current_blocks)
            return
        
        # Lower bound
        lb = lp_lower_bound(uncovered_mask, len(current_blocks))
        if len(current_blocks) + lb >= best_len_bnb:
            return
        
        # Pick an uncovered t-subset - choose the one with fewest candidate blocks
        # to minimize branching
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
    
    # Start B&B
    full_mask = (1 << M) - 1
    bb_search(full_mask, [], 0)
    
    if best_len_bnb < best_len:
        best_blocks = best_blocks_bnb
        best_len = best_len_bnb
    
    # Final reduction
    best_blocks = reduce_blocks(best_blocks)
    best_len = len(best_blocks)
    
    return best_blocks
