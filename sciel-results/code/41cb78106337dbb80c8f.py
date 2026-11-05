import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build incidence structure
    # For each block, find which t-subsets it covers
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]

    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[s] for s in t_subs_in_block]
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)

    # Build sparse matrix: rows = blocks, cols = t-subsets, entry = 1 if block covers subset
    # This allows fast gain computation via matrix operations
    row_data = []
    col_data = []
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            row_data.append(b_idx)
            col_data.append(idx)

    incidence = csr_matrix(
        (np.ones(len(row_data), dtype=np.int32), (row_data, col_data)),
        shape=(num_k_blocks, num_t_subsets)
    )

    # Precompute block sizes (number of t-subsets each block covers)
    block_sizes = np.array([len(indices) for indices in block_t_indices], dtype=np.int32)

    # Greedy construction with vectorized gain computation
    def greedy_construct(seed, use_random_order=False):
        """Greedy covering construction with vectorized gain calculation."""
        rng = random.Random(seed)
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        selected_blocks = []
        
        # For efficiency, we'll use a different approach:
        # At each step, compute gain for all unselected blocks, pick the best
        # But this is O(num_k_blocks) per step, which is ~75582 * ~300 = ~22M operations
        # Let's be smarter: only consider blocks that cover at least one uncovered subset
        
        # Precompute: for each t-subset, list of blocks that cover it
        # We already have t_subset_to_blocks
        
        uncovered_count = num_t_subsets
        remaining_blocks = set(range(num_k_blocks))
        
        while uncovered_count > 0:
            # Find blocks that cover at least one uncovered t-subset
            # Compute gain for each candidate block
            # A block's gain = number of t-subsets it covers that are currently uncovered
            
            # Strategy: pick a random uncovered t-subset, get all blocks covering it,
            # then among those, pick the one with highest gain
            # But also consider blocks covering multiple uncovered subsets
            
            # More efficient: sample uncovered subsets, collect candidate blocks,
            # compute gains for candidates only
            
            # Pick several uncovered subsets to broaden candidate pool
            num_samples = 20
            candidate_blocks = set()
            
            for _ in range(num_samples):
                # Find an uncovered subset
                attempts = 0
                while attempts < 100:
                    idx = rng.randint(0, num_t_subsets - 1)
                    if coverage[idx] == 0:
                        break
                    attempts += 1
                else:
                    continue
                    
                # Add all blocks covering this subset
                for b_idx in t_subset_to_blocks[idx]:
                    if b_idx in remaining_blocks:
                        candidate_blocks.add(b_idx)
            
            if not candidate_blocks:
                break
            
            # Compute gains for candidate blocks
            cand_list = list(candidate_blocks)
            cand_indices = np.array(cand_list, dtype=np.int32)
            
            # For each candidate block, count how many of its t-subsets are uncovered
            # Use the incidence matrix
            cand_incidence = incidence[cand_indices, :]  # sparse matrix
            cand_covered = (coverage[0] == 0)  # boolean array of uncovered
            
            # Gain for each candidate = sum of (incidence * uncovered)
            # Since incidence is 0/1, gain = number of uncovered subsets in the block
            gains = np.array([
                np.sum(coverage[indices] == 0) 
                for indices in [block_t_indices[b] for b in cand_list]
            ], dtype=np.int32)
            
            # Pick the block with maximum gain
            best_pos = np.argmax(gains)
            if gains[best_pos] == 0:
                break
                
            best_block = cand_list[best_pos]
            selected_blocks.append(best_block)
            remaining_blocks.discard(best_block)
            
            # Update coverage
            idxs = block_t_indices[best_block]
            coverage[idxs] += 1
            uncovered_count -= gains[best_pos]
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        return selected_blocks

    # Reduction phase: try to remove blocks that aren't needed
    def reduce_covering(blocks, coverage):
        """Try to reduce the covering by removing redundant blocks."""
        blocks = list(blocks)
        coverage = coverage.copy()
        
        # Sort blocks by size (smaller blocks first) to find redundant ones faster
        # A block is redundant if removing it doesn't leave any t-subset uncovered
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                idxs = block_t_indices[b_idx]
                # Check if any t-subset in this block has coverage == 1
                if np.any(coverage[idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # This block is redundant
                    coverage[idxs] -= 1
                    changed = True
            blocks = new_blocks
            
            if time.time() > deadline - 5.0:
                break
        
        return blocks, coverage

    # Multi-restart greedy with different random seeds
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        seed = restart * 7919 + 12345
        blocks = greedy_construct(seed)
        
        if not blocks:
            continue
            
        # Compute coverage for reduction
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            coverage[block_t_indices[b_idx]] += 1
        
        # Verify all t-subsets are covered
        if np.any(coverage == 0):
            continue
            
        # Reduce
        reduced_blocks, _ = reduce_covering(blocks, coverage)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            
            # Early stop if we hit a good target
            if best_count <= 335:
                break

    # If no valid solution found, do a pure greedy without time pressure
    if best_blocks is None:
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        blocks = []
        
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
                
            uncovered = np.where(coverage == 0)[0]
            if len(uncovered) == 0:
                break
                
            # Pick a random uncovered subset
            u_idx = uncovered[random.randint(0, len(uncovered) - 1)]
            
            # Get all blocks covering this subset
            candidates = [b for b in t_subset_to_blocks[u_idx] if b not in blocks]
            if not candidates:
                break
                
            # Pick the block with highest gain
            best_block = None
            best_gain = -1
            for b_idx in candidates:
                gain = np.sum(coverage[block_t_indices[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is not None and best_gain > 0:
                blocks.append(best_block)
                coverage[block_t_indices[best_block]] += 1
        
        if blocks and np.all(coverage > 0):
            best_blocks = blocks
            best_count = len(blocks)

    if best_blocks is None:
        # Fallback: return empty (invalid, but shouldn't happen)
        return []

    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_blocks]
