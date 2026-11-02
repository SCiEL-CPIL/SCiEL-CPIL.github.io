"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Use lists of indices for numpy compatibility
    block_cover_lists = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_lists.append(indices)
    
    # Precompute t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_list in enumerate(block_cover_lists):
        for ts_idx in cover_list:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    # block_cover_matrix: shape (num_blocks, C(k,t)) - but C(k,t) is fixed
    # Actually, we can use a sparse representation or just lists for gain calculation
    # Let's stick to lists for flexibility but use numpy for the main greedy if possible.
    # For C(19,8,5), num_blocks = 75582, num_t_sets = 11628.
    # A dense matrix would be 75582 * 11628 * 8 bytes ~ 7GB. Too big.
    # So we must use the list-of-lists or sparse approach.
    
    best_solution = None
    best_size = float('inf')
    
    # Strategies: (beam_width, seed)
    # Dual-guided greedy: prioritize hard-to-cover t-sets
    strategies = [
        (1, 42),
        (3, 101),
        (5, 102),
        (10, 103),
        (20, 104),
        (5, 201),
        (10, 202),
        (20, 203),
        (50, 301),
        (100, 302),
    ]
    
    for beam_width, seed in strategies:
        if time.time() > deadline - 15.0:
            break
            
        sol = _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_lists, num_t_sets, t_set_to_blocks, deadline, 
                                 rng_seed=seed, beam_width=beam_width)
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
    # Try local improvement: remove redundant blocks
    if best_solution and time.time() < deadline - 20.0:
        improved = _local_improve(best_solution, block_cover_lists, t_set_to_blocks, num_t_sets, deadline)
        if improved and len(improved) < best_size:
            best_solution = improved
            best_size = len(improved)
        
    if best_solution is None:
        return []
        
    return best_solution

def _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_lists, num_t_sets, t_set_to_blocks, deadline, rng_seed=42, beam_width=1):
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    # Precompute initial block counts per t-set
    tset_block_counts = [len(t_set_to_blocks[i]) for i in range(num_t_sets)]
    # Track remaining blocks per t-set (decrement when a block is selected)
    remaining_counts = tset_block_counts[:]
    
    while uncovered:
        if time.time() > deadline - 2.0:
            return None
        
        candidate_blocks = set()
        
        # Pick beam_width t-sets from uncovered, weighted by difficulty
        # Difficulty = 1 / remaining_counts (higher for t-sets with fewer options)
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            # Sample with probability proportional to difficulty
            weights = [1.0 / max(remaining_counts[i], 1) for i in uncovered]
            # Normalize
            total_w = sum(weights)
            if total_w == 0:
                targets = rng.sample(list(uncovered), beam_width)
            else:
                probs = [w / total_w for w in weights]
                targets = rng.choices(list(uncovered), weights=probs, k=beam_width)
                # Deduplicate
                targets = list(set(targets))
                if len(targets) < beam_width:
                    # Fill with random
                    remaining_uncovered = list(uncovered - set(targets))
                    if remaining_uncovered:
                        targets.extend(rng.sample(remaining_uncovered, min(beam_width - len(targets), len(remaining_uncovered))))
            
        for target_tset_idx in targets:
            # Find best block covering this target, using weighted gain
            best_block_idx = -1
            max_weighted_gain = 0
            
            for b_idx in t_set_to_blocks[target_tset_idx]:
                # Weighted gain: sum of dual weights of uncovered t-sets covered by this block
                weighted_gain = 0.0
                for ts_idx in block_cover_lists[b_idx]:
                    if ts_idx in uncovered:
                        # Weight by difficulty of this t-set
                        weighted_gain += 1.0 / max(remaining_counts[ts_idx], 1)
                if weighted_gain > max_weighted_gain:
                    max_weighted_gain = weighted_gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)
        
        # Pick the best block among candidates using weighted gain
        final_best_block_idx = -1
        final_max_weighted_gain = 0
        
        for b_idx in candidate_blocks:
            weighted_gain = 0.0
            for ts_idx in block_cover_lists[b_idx]:
                if ts_idx in uncovered:
                    weighted_gain += 1.0 / max(remaining_counts[ts_idx], 1)
            if weighted_gain > final_max_weighted_gain:
                final_max_weighted_gain = weighted_gain
                final_best_block_idx = b_idx
        
        if final_best_block_idx == -1:
            break
            
        selected_blocks.append(all_blocks[final_best_block_idx])
        
        # Update uncovered and remaining counts
        covered = block_cover_lists[final_best_block_idx]
        for ts_idx in covered:
            if ts_idx in uncovered:
                remaining_counts[ts_idx] -= 1
        uncovered -= set(covered)
        
    return selected_blocks

def _local_improve(solution, block_cover_lists, t_set_to_blocks, num_t_sets, deadline):
    """Try to remove redundant blocks."""
    if not solution:
        return None
    
    # Convert solution to list of lists of indices for fast lookup
    sol_cover_lists = []
    for block in solution:
        indices = [t_set_to_blocks[0][0] for _ in range(0)] # dummy
        # Actually we need to map the block (tuple of points) to its t-set indices
        # We don't have a direct map from block tuple to indices in the current structure.
        # Let's recompute or pass the mapping.
        # For now, let's assume we can recompute or use a set.
        pass
    
    # Better approach: Use the block_cover_lists if we had the index.
    # But solution is a list of blocks (tuples).
    # We need to find the index of each block in all_blocks to use block_cover_lists.
    # This is expensive. Let's just recompute the coverage for the solution.
    
    # Build a set of t-set indices covered by each block in the solution
    sol_tsets = []
    for block in solution:
        tsets_in_block = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        sol_tsets.append(tsets_in_block)
        
    # Count coverage of each t-set by the solution
    coverage_count = np.zeros(num_t_sets, dtype=np.int32)
    for tsets_in_block in sol_tsets:
        for ts_idx in tsets_in_block:
            coverage_count[ts_idx] += 1
            
    # Try to remove blocks
    # A block is redundant if all its t-sets are covered by at least 2 blocks
    # i.e., min(coverage_count[s] for s in block) > 1
    
    changed = True
    new_solution = solution[:]
    new_sol_tsets = sol_tsets[:]
    
    while changed:
        if time.time() > deadline - 2.0:
            break
        changed = False
        for i in range(len(new_solution)):
            # Check if redundant
            is_redundant = True
            for ts_idx in new_sol_tsets[i]:
                if coverage_count[ts_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                # Update counts
                for ts_idx in new_sol_tsets[i]:
                    coverage_count[ts_idx] -= 1
                new_solution.pop(i)
                new_sol_tsets.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    if len(new_solution) < len(solution):
        return new_solution
    return None
# EVOLVE-BLOCK-END
