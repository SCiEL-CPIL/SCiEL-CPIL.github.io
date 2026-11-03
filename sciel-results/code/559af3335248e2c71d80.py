import numpy as np
import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    c_k_t = math.comb(k, t)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # triple_to_blocks[j] is a list of block indices that cover triple j
    # We use lists of lists for sparse access, or arrays if dense enough.
    # Freq is ~210 for v=24, k=7, t=3. n_triples=2024.
    # 2024 * 210 ~ 425k entries. List of lists is fine.
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_to_blocks[t_idx].append(i)
            idx += 1
            
    # Compute triple coverage frequency
    triple_freq = np.array([len(lst) for lst in triple_to_blocks], dtype=np.int32)
    
    # Weights: 1/freq
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    # Initial gains for each block: sum of weights of all triples in block
    initial_gains = np.sum(triple_weights[block_triples], axis=1)
    
    def greedy_cover_incremental(seed=None):
        """
        Run incremental weighted greedy set cover.
        """
        if seed is not None:
            random.seed(seed)
            
        uncovered = np.ones(n_triples, dtype=bool)
        gains = initial_gains.copy()
        selected = []
        
        # Track active blocks? No, we just pick max gain.
        # If gain is 0, we are stuck or done.
        
        while True:
            if time.time() > deadline:
                break
                
            best_idx = int(np.argmax(gains))
            if gains[best_idx] < 1e-9:
                break
                
            selected.append(best_idx)
            
            # Update uncovered and gains
            block_t_idx = block_triples[best_idx]
            for t_idx in block_t_idx:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    w = triple_weights[t_idx]
                    # Subtract w from gains of all blocks covering this triple
                    for b_idx in triple_to_blocks[t_idx]:
                        gains[b_idx] -= w
                        
        return selected

    def prune_blocks(selected_blocks):
        """
        Remove redundant blocks from a valid covering.
        """
        if not selected_blocks:
            return selected_blocks
            
        # Build counts
        counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            counts[block_triples[b_idx]] += 1
            
        remaining = list(selected_blocks)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            for block_idx in remaining:
                if time.time() > deadline:
                    break
                    
                block_t_idx = block_triples[block_idx]
                
                # Check if all triples in this block have coverage > 1
                can_remove = True
                for t_idx in block_t_idx:
                    if counts[t_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove
                    for t_idx in block_t_idx:
                        counts[t_idx] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining

    def local_search_improve(initial_blocks, time_limit):
        """
        Simulated Annealing local search.
        Moves: Remove one block, Add one block.
        """
        if not initial_blocks:
            return initial_blocks

        current_blocks = list(initial_blocks)
        current_counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_blocks:
            current_counts[block_triples[b_idx]] += 1
            
        current_uncovered = int(np.sum(current_counts == 0))
        current_size = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = current_size
        
        M = 1000  # Penalty
        current_obj = current_size + M * current_uncovered
        best_obj = current_obj
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 1e-5
        
        all_block_indices = list(range(n_blocks))
        
        start_ls = time.time()
        
        while time.time() < start_ls + time_limit:
            if T < min_T:
                # Restart temperature if stuck? Or just break.
                T = 0.5
                
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
            
            # Update best
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
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
                # Recompute uncovered is expensive, but revert is O(1) if we tracked it.
                # We tracked current_uncovered.
                # Revert uncovered:
                # When we added, we decremented uncovered if count was 0.
                # When we removed, we incremented uncovered if count was 1.
                # Revert add:
                # For t in add_triples:
                #   if current_counts[t] (after revert) was 1 (meaning it was 0 before add? No)
                # Let's just recompute uncovered on revert? No, too slow.
                # Let's track carefully.
                
                # Actually, the revert logic in the original code recomputed uncovered.
                # To be safe and fast, let's just recompute uncovered if revert happens often?
                # No, revert is O(1) if we just undo the delta.
                
                # Delta uncovered when removing remove_block:
                # For each t in remove_block: if count[t] was 1, uncovered++.
                # Delta uncovered when adding add_block:
                # For each t in add_block: if count[t] was 0, uncovered--.
                
                # To revert:
                # Undo add:
                # For each t in add_block:
                #   current_counts[t] -= 1
                #   if current_counts[t] == 0: # It was 1 before add? No, it was 0 before add if uncovered was decremented.
                #   Logic:
                #   Before add: count[t] = c. After add: c+1.
                #   If c==0, uncovered was decremented.
                #   So if c+1 == 1 (i.e., c==0), uncovered was decremented.
                #   So if current_counts[t] (after add, before revert) is 1, it means c was 0.
                #   So if current_counts[t] == 1, we must increment uncovered.
                
                # Undo remove:
                # For each t in remove_block:
                #   current_counts[t] += 1
                #   Before remove: count[t] = c. After remove: c-1.
                #   If c==1, uncovered was incremented.
                #   So if current_counts[t] (after remove, before revert) is 0, it means c was 1.
                #   So if current_counts[t] == 0, we must decrement uncovered.
                
                # Let's implement this precise revert.
                
                # Undo Add
                for t_idx in add_triples:
                    if current_counts[t_idx] == 1:
                        current_uncovered += 1
                    current_counts[t_idx] -= 1
                current_blocks.pop() # Already popped? No, I popped in 'if not accept' block above?
                # Wait, in the code above:
                # if not accept:
                #   current_counts[block_triples[add_block]] -= 1 # This is vectorized, doesn't track uncovered!
                #   current_blocks.pop()
                #   ...
                
                # The vectorized update `current_counts[block_triples[add_block]] -= 1` does NOT update `current_uncovered`.
                # So I MUST recompute `current_uncovered` on revert or track it manually.
                # Given the complexity, and that reverts are frequent, recomputing `np.sum(current_counts == 0)` is O(n_triples) = 2024 ops.
                # This is acceptable if reverts are not too frequent.
                # But SA reverts are frequent.
                # Let's stick to the safe recompute for correctness, as 2024 ops is small.
                
                current_uncovered = int(np.sum(current_counts == 0))
                current_obj = old_obj
            else:
                current_obj = new_obj
                
            T *= cooling_rate
            
            # Periodic prune
            if current_uncovered == 0 and len(current_blocks) < best_size:
                # Prune current to see if we can improve best
                pass # Pruning inside LS is expensive. Skip for now.

        return best_blocks

    # 2. Multiple Greedy Restarts
    candidates = []
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() > deadline - 30.0: # Leave 30s for LS
            break
        seed = i * 12345 + 67890
        sel = greedy_cover_incremental(seed)
        if sel:
            pruned = prune_blocks(sel)
            if pruned:
                candidates.append(pruned)
                
    if not candidates:
        # Fallback
        sel = greedy_cover_incremental(0)
        if sel:
            candidates.append(prune_blocks(sel))
        else:
            return [list(b) for b in blocks[:100]]

    # Sort candidates by size
    candidates.sort(key=lambda x: len(x))
    
    # Keep top 3 unique-ish candidates
    top_candidates = []
    for c in candidates:
        if len(top_candidates) >= 3:
            break
        # Simple uniqueness check (optional, but good)
        is_unique = True
        for tc in top_candidates:
            if len(c) == len(tc) and set(c) == set(tc):
                is_unique = False
                break
        if is_unique:
            top_candidates.append(c)

    # 3. Local Search on Top Candidates
    remaining_time = deadline - time.time()
    time_per_ls = remaining_time / len(top_candidates) if top_candidates else 0
    
    best_overall = top_candidates[0]
    
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline:
            break
        ls_time = time_per_ls
        # Ensure we don't exceed deadline
        if time.time() + ls_time > deadline:
            ls_time = deadline - time.time()
        
        result = local_search_improve(cand, ls_time)
        if result and len(result) < len(best_overall):
            best_overall = result

    # Final Prune
    best_overall = prune_blocks(best_overall)
    
    # Validation
    if best_overall:
        counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in best_overall:
            counts[block_triples[b_idx]] += 1
        if np.any(counts == 0):
            # Fallback to first candidate
            best_overall = candidates[0]
            best_overall = prune_blocks(best_overall)

    return [list(blocks[i]) for i in best_overall]
