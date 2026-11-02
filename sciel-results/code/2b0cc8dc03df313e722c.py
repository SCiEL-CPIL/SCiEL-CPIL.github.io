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
    deadline = time.time() + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute gain contributions:
    # For each point x, map (t-1)-subset to t-subset index
    point_to_tsubset_map = [{} for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for i in range(t):
            rest = list(subset[:i] + subset[i+1:])
            rest.sort()
            p = subset[i]
            point_to_tsubset_map[p][tuple(rest)] = idx

    def greedy_cover_v2(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_list = list(range(n_t_subsets))
        idx_to_pos = {i: i for i in range(n_t_subsets)}
        
        blocks = []
        
        while len(uncovered_list) > 0:
            if time.time() > deadline - 20:
                return None
            
            # Pick a random uncovered t-subset
            pos = rng.randint(0, len(uncovered_list) - 1)
            start_idx = uncovered_list[pos]
            
            # Remove start_idx from uncovered_list and idx_to_pos
            last_idx = uncovered_list[-1]
            uncovered_list[pos] = last_idx
            idx_to_pos[last_idx] = pos
            uncovered_list.pop()
            del idx_to_pos[start_idx]
            uncovered[start_idx] = False
            
            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            
            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 20:
                    return None
                
                block_set = set(block)
                best_x = -1
                best_gain = -1
                
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        idx = point_to_tsubset_map[x].get(tuple(sorted(rest)), -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0 and rng.random() < 0.5:
                        best_x = x
                
                if best_x >= 0:
                    block.append(best_x)
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
        """Remove redundant blocks."""
        if not blocks:
            return []
        
        blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Compute coverage counts
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] += 1
            
            # Try removing each block
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5:
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
                    improved = True
                    break
                else:
                    # Restore coverage counts
                    for a_idx in affected_indices:
                        coverage_count[a_idx] += 1
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds for greedy construction
    seed = 0
    while time.time() < deadline - 10:
        result = greedy_cover_v2(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
        seed += 1
    
    if best_blocks is None:
        # Fallback: simple greedy without randomization to ensure completion
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_list = list(range(n_t_subsets))
        idx_to_pos = {i: i for i in range(n_t_subsets)}
        blocks = []
        
        while len(uncovered_list) > 0:
            start_idx = uncovered_list[0]
            
            last_idx = uncovered_list[-1]
            uncovered_list[0] = last_idx
            idx_to_pos[last_idx] = 0
            uncovered_list.pop()
            del idx_to_pos[start_idx]
            uncovered[start_idx] = False
            
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        idx = point_to_tsubset_map[x].get(tuple(sorted(rest)), -1)
                        if idx != -1 and uncovered[idx]:
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
                    if idx in idx_to_pos:
                        p = idx_to_pos[idx]
                        last_idx = uncovered_list[-1]
                        uncovered_list[p] = last_idx
                        idx_to_pos[last_idx] = p
                        uncovered_list.pop()
                        del idx_to_pos[idx]
        
        best_blocks = blocks
    
    return best_blocks

# EVOLVE-BLOCK-END
