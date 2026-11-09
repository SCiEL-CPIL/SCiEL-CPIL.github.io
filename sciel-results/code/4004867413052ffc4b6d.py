import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using greedy set cover with
    efficient bitmask operations, followed by local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build block masks efficiently
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    # 3. Greedy set cover with multiple tie-breaking strategies
    def greedy_cover(tie_break='consolidate'):
        """
        Greedy set cover. tie_break: 'consolidate' prefers blocks overlapping more with current union,
        'diversify' prefers blocks overlapping less.
        """
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_indices = []
        current_union = 0
        
        while current_uncovered != 0:
            if time.time() > deadline - 3.0:
                break
            
            best_gain = -1
            best_block_idx = -1
            best_tie = -1 if tie_break == 'consolidate' else float('inf')
            
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    best_tie = count_bits(block_masks[i] & current_union)
                elif gain == best_gain and gain > 0:
                    tie = count_bits(block_masks[i] & current_union)
                    if tie_break == 'consolidate':
                        if tie > best_tie:
                            best_tie = tie
                            best_block_idx = i
                    else:  # diversify
                        if tie < best_tie:
                            best_tie = tie
                            best_block_idx = i
            
            if best_block_idx == -1:
                break
            
            chosen_indices.append(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            current_union |= block_masks[best_block_idx]
        
        return chosen_indices
    
    # 4. Remove redundant blocks
    def remove_redundant(chosen_indices):
        """Remove blocks that don't contribute unique coverage."""
        if not chosen_indices:
            return chosen_indices
        
        n = len(chosen_indices)
        masks = [block_masks[idx] for idx in chosen_indices]
        
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if time.time() > deadline - 3.0:
                    return [all_blocks[idx] for idx in chosen_indices]
                
                # Compute union of all other blocks
                union_others = 0
                for j in range(n):
                    if i != j:
                        union_others |= masks[j]
                
                # Check if mask_i is subset of union_others
                if (masks[i] & ~union_others) == 0:
                    # Remove block i
                    chosen_indices.pop(i)
                    masks.pop(i)
                    n -= 1
                    changed = True
                    break
        
        return [all_blocks[idx] for idx in chosen_indices]
    
    # 5. Local search: 1-exchange
    def local_search(chosen_indices, max_iter=10):
        """Try to replace a block with another that covers its unique subsets."""
        if not chosen_indices:
            return chosen_indices
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            n = len(chosen_indices)
            
            if time.time() > deadline - 5.0:
                break
            
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                
                mask_i = block_masks[chosen_indices[i]]
                
                # Compute union of all other blocks
                union_others = 0
                for j in range(n):
                    if i != j:
                        union_others |= block_masks[chosen_indices[j]]
                
                unique_i = mask_i & ~union_others
                if unique_i == 0:
                    # Redundant - remove
                    chosen_indices.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all of unique_i
                unique_count = count_bits(unique_i)
                best_block_idx = -1
                best_overlap = -1
                
                for j in range(num_blocks):
                    if j in chosen_indices:
                        continue
                    # Check if block j covers all of unique_i
                    if (block_masks[j] & unique_i) == unique_i:
                        overlap = count_bits(block_masks[j] & union_others)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_block_idx = j
                
                if best_block_idx != -1:
                    # Replace block i with best_block_idx
                    chosen_indices[i] = best_block_idx
                    improved = True
                    break
        
        return chosen_indices
    
    # 6. Run multiple greedy strategies and keep the best
    solutions = []
    
    # Strategy 1: Consolidate tie-breaking
    if time.time() < deadline - 10.0:
        sol1 = greedy_cover('consolidate')
        if sol1:
            sol1 = local_search(sol1, max_iter=5)
            solutions.append(sol1)
    
    # Strategy 2: Diversify tie-breaking
    if time.time() < deadline - 10.0:
        sol2 = greedy_cover('diversify')
        if sol2:
            sol2 = local_search(sol2, max_iter=5)
            solutions.append(sol2)
    
    # Pick the smallest solution
    if not solutions:
        # Fallback: just use consolidate
        sol = greedy_cover('consolidate')
        return [list(all_blocks[idx]) for idx in sol]
    
    best_solution = min(solutions, key=len)
    
    # Final local search pass
    best_solution = local_search(best_solution, max_iter=10)
    
    # Remove any remaining redundancies
    best_solution = remove_redundant(best_solution)
    
    return [list(block) for block in best_solution]
