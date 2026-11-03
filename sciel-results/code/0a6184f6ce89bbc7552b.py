import numpy as np
import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave 35s buffer)
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # c_k_t = C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute triple coverage frequency: how many blocks cover each triple
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
        
    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    def greedy_cover():
        """Run weighted greedy set cover."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Weighted gain: sum of weights of uncovered triples in block
            gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            
            # Find the block with maximum gain
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def get_coverage_counts(selected_blocks):
        """Compute coverage counts for a list of block indices."""
        counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            counts[block_triples[b_idx]] += 1
        return counts

    def prune_blocks(selected_blocks):
        """Remove redundant blocks from a valid covering."""
        if not selected_blocks:
            return selected_blocks
            
        counts = get_coverage_counts(selected_blocks)
        remaining = list(selected_blocks)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            for block_idx in remaining:
                if time.time() > deadline:
                    break
                    
                block_triple_indices = block_triples[block_idx]
                
                # Check if all triples in this block have coverage > 1
                can_remove = True
                for triple_idx in block_triple_indices:
                    if counts[triple_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block and update coverage counts
                    for triple_idx in block_triple_indices:
                        counts[triple_idx] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining

    def local_search_improve(initial_blocks, max_iters=200000):
        """
        Local search with swap moves (remove one block, add one block).
        Uses incremental tracking of uncovered count for speed.
        """
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_uncovered = int(np.sum(current_counts == 0))
        current_size = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = current_size
        best_uncovered = current_uncovered
        
        M = 1000  # Penalty for uncovered triples
        current_obj = current_size + M * current_uncovered
        best_obj = current_obj
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 1e-4
        
        all_block_indices = list(range(n_blocks))
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            if not current_blocks:
                break
                
            # Pick a random block to remove
            remove_idx_pos = random.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx_pos]
            
            # Pick a random block to add
            add_block = random.choice(all_block_indices)
            
            if add_block == remove_block:
                continue
                
            old_obj = current_obj
            
            # --- Remove ---
            rem_triples = block_triples[remove_block]
            for t_idx in rem_triples:
                if current_counts[t_idx] == 1:
                    current_uncovered += 1
                current_counts[t_idx] -= 1
                
            current_blocks.pop(remove_idx_pos)
            current_size -= 1
            
            # --- Add ---
            add_triples = block_triples[add_block]
            for t_idx in add_triples:
                if current_counts[t_idx] == 0:
                    current_uncovered -= 1
                current_counts[t_idx] += 1
                
            current_blocks.append(add_block)
            current_size += 1
            
            # --- Objective ---
            new_obj = current_size + M * current_uncovered
            
            # Update best if valid and smaller
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_uncovered = 0
                best_blocks = list(current_blocks)
                best_obj = new_obj
                
            # Acceptance
            delta = new_obj - old_obj
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    p = math.exp(-delta / T)
                    accept = random.random() < p
                else:
                    accept = False
            
            if not accept:
                # Revert
                current_counts[block_triples[add_block]] -= 1
                current_blocks.pop()
                
                current_counts[block_triples[remove_block]] += 1
                current_blocks.insert(remove_idx_pos, remove_block)
                
                current_size = len(current_blocks)
                # Recompute uncovered after revert
                # We need to track this properly. Let's just recompute.
                current_uncovered = int(np.sum(current_counts == 0))
                current_obj = old_obj
            else:
                current_obj = new_obj
                
            T *= cooling_rate
            
            # Periodically prune to reduce size if valid
            if i % 500 == 0 and current_uncovered == 0:
                pruned = prune_blocks(current_blocks)
                if len(pruned) < current_size:
                    current_blocks = pruned
                    current_counts = get_coverage_counts(current_blocks)
                    current_size = len(current_blocks)
                    current_uncovered = int(np.sum(current_counts == 0))
                    current_obj = current_size + M * current_uncovered
                    if current_size < best_size:
                        best_size = current_size
                        best_uncovered = 0
                        best_blocks = list(current_blocks)
                        best_obj = current_obj

        return best_blocks

    # 1. Get initial solution via greedy
    initial_blocks = greedy_cover()
    if not initial_blocks:
        return [list(b) for b in blocks[:100]]
        
    # Prune initial solution
    initial_blocks = prune_blocks(initial_blocks)
    
    # 2. Run local search for remaining time
    ls_blocks = local_search_improve(initial_blocks)
    
    # 3. Final Prune
    ls_blocks = prune_blocks(ls_blocks)
    
    # 4. Validation
    if ls_blocks:
        counts = get_coverage_counts(ls_blocks)
        if np.any(counts == 0):
            # Invalid, fallback to initial
            ls_blocks = initial_blocks
            ls_blocks = prune_blocks(ls_blocks)
            
    # Convert to list of lists
    result = [list(blocks[i]) for i in ls_blocks]
    
    return result
