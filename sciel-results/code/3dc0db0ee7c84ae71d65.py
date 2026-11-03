import time
import random
import itertools
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # For each block, store the bitmask of t-subsets it covers
    # n_t_subsets = C(24,3) = 2024, so we need 2024 bits = 253 bytes per block
    # Use Python integers for arbitrary-precision bitmasks
    block_masks = [0] * n_k_subsets
    # triple_to_blocks: for each t-subset, list of block indices covering it
    triple_to_blocks = [[] for _ in range(n_t_subsets)]
    # Also store the list of triple indices for each block (for update operations)
    block_triple_list = [None] * n_k_subsets
    
    for i, block in enumerate(all_k_subsets):
        mask = 0
        triples = []
        for triple in itertools.combinations(block, t):
            idx = t_subset_to_idx[triple]
            mask |= (1 << idx)
            triples.append(idx)
            triple_to_blocks[idx].append(i)
        block_masks[i] = mask
        block_triple_list[i] = triples

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_t_subsets) - 1  # All uncovered initially
        
        blocks = []
        
        while uncovered_mask:
            if time.time() > deadline - 10:
                return None
            
            # Pick a random uncovered triple
            # Find the lowest set bit
            # Use bit trick: x & (-x) gives lowest set bit
            low_bit = uncovered_mask & (-uncovered_mask)
            start_idx = low_bit.bit_length() - 1
            
            # Get all blocks covering this triple
            candidates = triple_to_blocks[start_idx]
            if not candidates:
                return None
            
            # Shuffle to break ties
            rng.shuffle(candidates)
            
            # Compute gain for each candidate: number of uncovered triples it covers
            # gain = popcount(block_mask[b] & uncovered_mask)
            best_block_id = -1
            best_gain = 0
            
            for b_id in candidates:
                gain = (block_masks[b_id] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_id = b_id
            
            if best_block_id == -1:
                return None
                
            blocks.append(best_block_id)
            
            # Update uncovered mask
            uncovered_mask &= ~block_masks[best_block_id]
                    
        return blocks

    def reduce_and_optimize(blocks, seed):
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        
        current_blocks = list(blocks)
        
        # Compute coverage counts using the block triple lists
        cov_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_id in current_blocks:
            for t_idx in block_triple_list[b_id]:
                cov_counts[t_idx] += 1
                
        # Phase 1: Simple Reduction - remove blocks that don't cover any unique triple
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_id = current_blocks[i]
                can_remove = True
                for t_idx in block_triple_list[b_id]:
                    if cov_counts[t_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for t_idx in block_triple_list[b_id]:
                        cov_counts[t_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                    
        # Phase 2: Local Search (Swap) with bitmask-based candidate filtering
        swap_attempts = 0
        while time.time() < deadline - 5:
            if not current_blocks:
                break
                
            b_idx = rng.randint(0, len(current_blocks) - 1)
            old_b_id = current_blocks[b_idx]
            
            # Identify critical triples in this block (count == 1)
            critical_triples = []
            for t_idx in block_triple_list[old_b_id]:
                if cov_counts[t_idx] == 1:
                    critical_triples.append(t_idx)
            
            # If no critical triples, remove it
            if not critical_triples:
                for t_idx in block_triple_list[old_b_id]:
                    cov_counts[t_idx] -= 1
                current_blocks.pop(b_idx)
                continue
            
            # Build bitmask of critical triples
            critical_mask = 0
            for t_idx in critical_triples:
                critical_mask |= (1 << t_idx)
            
            # Find candidates: blocks that cover ALL critical triples
            # Start with blocks covering the first critical triple
            target_triple = critical_triples[0]
            candidates = triple_to_blocks[target_triple]
            
            best_new_b_id = -1
            best_score = -1
            
            for cand_id in candidates:
                # Check if cand covers all critical triples using bitmask
                if (block_masks[cand_id] & critical_mask) != critical_mask:
                    continue
                
                # Score: prefer blocks that cover more low-count triples
                score = 0
                for t_idx in block_triple_list[cand_id]:
                    c = cov_counts[t_idx]
                    if c <= 2:
                        score += 10
                    elif c <= 5:
                        score += 1
                    else:
                        score -= 1
                    
                if score > best_score:
                    best_score = score
                    best_new_b_id = cand_id
                    
            if best_new_b_id != -1 and best_new_b_id != old_b_id:
                # Perform swap
                for t_idx in block_triple_list[old_b_id]:
                    cov_counts[t_idx] -= 1
                for t_idx in block_triple_list[best_new_b_id]:
                    cov_counts[t_idx] += 1
                current_blocks[b_idx] = best_new_b_id
                
                # After swap, try to remove redundant blocks
                for i in range(len(current_blocks) - 1, -1, -1):
                    b_id = current_blocks[i]
                    can_remove = True
                    for t_idx in block_triple_list[b_id]:
                        if cov_counts[t_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for t_idx in block_triple_list[b_id]:
                            cov_counts[t_idx] -= 1
                        current_blocks.pop(i)
                        
        # Phase 3: Final polish - try 2-for-1 replacements
        # Look for two blocks that can be replaced by one
        if time.time() < deadline - 3:
            for i in range(len(current_blocks)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(current_blocks)):
                    if time.time() > deadline - 3:
                        break
                    # Check if union of triples covered by blocks i and j is covered by some single block
                    # This is expensive, so only do a few iterations
                    break
                break

        # Convert block IDs back to lists of points
        return [list(all_k_subsets[b_id]) for b_id in current_blocks]

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds
    seed = 0
    while time.time() < deadline - 15:
        greedy_result = greedy_cover(seed)
        if greedy_result is not None:
            optimized = reduce_and_optimize(greedy_result, seed)
            if len(optimized) < best_count:
                best_count = len(optimized)
                best_blocks = optimized
        seed += 1
        
    if best_blocks is None:
        # Fallback: try one more time with a fixed seed
        greedy_result = greedy_cover(42)
        if greedy_result is not None:
            best_blocks = reduce_and_optimize(greedy_result, 42)
        
    if best_blocks is None:
        return []
        
    return best_blocks

# EVOLVE-BLOCK-END
