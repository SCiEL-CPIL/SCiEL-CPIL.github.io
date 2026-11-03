import numpy as np
import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
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
            best_idx = np.argmax(gains)
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

    def simulated_annealing(initial_blocks, max_iters=100000):
        """
        Simulated Annealing to minimize number of blocks.
        Objective: NumBlocks + M * NumUncovered
        """
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_uncovered = np.sum(current_counts == 0)
        current_size = len(current_blocks)
        
        # Objective function: We want to minimize size, but coverage is a hard constraint for the final answer.
        # During SA, we allow invalid states but penalize them heavily.
        M = 1000  # Penalty for uncovered triples
        
        current_obj = current_size + M * current_uncovered
        
        best_blocks = list(current_blocks)
        best_size = current_size
        best_uncovered = current_uncovered
        best_obj = current_obj
        
        # Initial temperature
        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-4
        
        # Precompute block indices for fast access
        # We need to pick a block to remove and a block to add.
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            # Pick a random block to remove
            if not current_blocks:
                break
            remove_idx_pos = random.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx_pos]
            
            # Pick a random block to add (different from remove_block)
            add_block = random.randint(0, n_blocks - 1)
            if add_block == remove_block:
                add_block = (add_block + 1) % n_blocks
                
            # Check if add_block is already in current_blocks
            # If it is, we are just swapping positions or doing nothing? 
            # Actually, if we remove A and add B, and B is already there, 
            # we are effectively just removing A (if B != A).
            # But if B is already there, the set of blocks doesn't change size, 
            # just the specific instance? No, blocks are unique.
            # If add_block is in current_blocks, and we remove remove_block,
            # the new set is (old set - remove_block) + add_block.
            # If add_block == remove_block, no change.
            # If add_block != remove_block and add_block in old set:
            #   old set: {A, B, C}
            #   remove A, add B -> {B, C, B} -> Set is {B, C}. Size decreases by 1?
            #   Wait, blocks are distinct indices.
            #   If we treat it as a multiset, we can have duplicates. But covering design usually implies a set.
            #   Let's ensure we don't add a block that is already present unless we are removing the one we are adding?
            #   Simpler: Generate a new set.
            
            # Let's use a list and update counts directly.
            # Step 1: Remove remove_block
            current_counts[block_triples[remove_block]] -= 1
            current_blocks.pop(remove_idx_pos)
            current_size -= 1
            
            # Step 2: Add add_block
            current_counts[block_triples[add_block]] += 1
            current_blocks.append(add_block)
            current_size += 1
            
            # Recalculate uncovered
            # This is O(n_triples) which is ~2024. 
            # max_iters is 100k. 100k * 2024 is 200M operations. Might be too slow in Python.
            # Optimization: Track uncovered count incrementally.
            
            # Let's restart with incremental tracking.
            
        # The above loop structure is hard to make incremental with simple pop/append if we don't track indices carefully.
        # Let's rewrite SA with explicit tracking.
        
        # Reset state
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_uncovered = np.sum(current_counts == 0)
        current_size = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = current_size
        best_uncovered = current_uncovered
        best_obj = current_size + M * current_uncovered
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 1e-4
        
        # Precompute list of block indices for random choice
        all_block_indices = list(range(n_blocks))
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            # Pick a random block to remove
            if not current_blocks:
                break
            remove_idx_pos = random.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx_pos]
            
            # Pick a random block to add
            add_block = random.choice(all_block_indices)
            
            # If adding the same block we are removing, no change
            if add_block == remove_block:
                continue
                
            # Apply move
            # Remove
            current_counts[block_triples[remove_block]] -= 1
            current_blocks.pop(remove_idx_pos)
            
            # Add
            current_counts[block_triples[add_block]] += 1
            current_blocks.append(add_block)
            
            # Calculate new uncovered count
            # We can't easily do this incrementally without knowing which triples went from 0->1 or 1->0.
            # current_uncovered = np.sum(current_counts == 0) is O(N).
            # Let's try to optimize this.
            # We can track the number of zeros.
            # When we decrement a count:
            #   if count was 1, it becomes 0 -> uncovered increases by 1
            #   if count was 0, it becomes -1 -> uncovered unchanged (but invalid)
            # When we increment a count:
            #   if count was 0, it becomes 1 -> uncovered decreases by 1
            #   if count was -1, it becomes 0 -> uncovered unchanged
            
            # Let's implement this incremental tracking.
            
        # Let's restart the SA implementation with proper incremental tracking.
        
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_uncovered = int(np.sum(current_counts == 0))
        current_size = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = current_size
        best_uncovered = current_uncovered
        best_obj = current_size + M * current_uncovered
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-4
        
        all_block_indices = list(range(n_blocks))
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            if not current_blocks:
                break
                
            remove_idx_pos = random.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx_pos]
            
            add_block = random.choice(all_block_indices)
            
            if add_block == remove_block:
                continue
                
            # --- Remove ---
            rem_triples = block_triples[remove_block]
            # Update uncovered count
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
            elif current_uncovered == 0 and current_size == best_size:
                # Tie, maybe keep old
                pass
            elif current_uncovered < best_uncovered:
                # Better coverage, even if size is larger, it's a good intermediate
                # But we only care about valid solutions for the final score.
                # However, for SA, we want to track the best *valid* solution.
                # If we find a valid solution, it's a candidate.
                if current_uncovered == 0:
                    best_size = current_size
                    best_uncovered = 0
                    best_blocks = list(current_blocks)
                    best_obj = new_obj

            # --- Acceptance ---
            delta = new_obj - best_obj if new_obj < best_obj else new_obj - (current_size + M * current_uncovered) # Wait, delta should be new_obj - old_obj
            # We need to store old_obj
            # Let's store old_obj before modifying
            
            # Re-do logic with old_obj stored
            
        # This manual tracking is error-prone. Let's use a simpler, slightly slower but correct approach first, 
        # and optimize if time permits. 
        # Actually, np.sum(counts == 0) is quite fast in numpy.
        # Let's stick to the numpy version but limit iterations.
        
        current_blocks = list(initial_blocks)
        current_counts = get_coverage_counts(current_blocks)
        current_size = len(current_blocks)
        current_uncovered = int(np.sum(current_counts == 0))
        current_obj = current_size + M * current_uncovered
        
        best_blocks = list(current_blocks)
        best_size = current_size
        best_uncovered = current_uncovered
        best_obj = current_obj
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-4
        
        all_block_indices = list(range(n_blocks))
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            if not current_blocks:
                break
                
            remove_idx_pos = random.randint(0, len(current_blocks) - 1)
            remove_block = current_blocks[remove_idx_pos]
            
            add_block = random.choice(all_block_indices)
            
            if add_block == remove_block:
                continue
                
            old_obj = current_obj
            
            # Apply move
            current_counts[block_triples[remove_block]] -= 1
            current_blocks.pop(remove_idx_pos)
            
            current_counts[block_triples[add_block]] += 1
            current_blocks.append(add_block)
            
            current_size = len(current_blocks)
            current_uncovered = int(np.sum(current_counts == 0))
            new_obj = current_size + M * current_uncovered
            
            # Update best
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
                # Accept with probability exp(-delta/T)
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
                current_uncovered = int(np.sum(current_counts == 0))
                current_obj = old_obj
            else:
                current_obj = new_obj
                
            T *= cooling_rate
            
            # Periodically prune to reduce size if valid
            if i % 1000 == 0 and current_uncovered == 0:
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

    # 1. Get initial solution
    initial_blocks = greedy_cover()
    if not initial_blocks:
        # Fallback
        return [list(b) for b in blocks[:100]]
        
    # Prune initial solution
    initial_blocks = prune_blocks(initial_blocks)
    
    # 2. Run Simulated Annealing
    # We can run multiple rounds with different random seeds or just one long one
    # Let's run one long SA for the remaining time
    sa_blocks = simulated_annealing(initial_blocks, max_iters=500000)
    
    # 3. Final Prune
    sa_blocks = prune_blocks(sa_blocks)
    
    # 4. Validation
    # Check if sa_blocks is valid
    if sa_blocks:
        counts = get_coverage_counts(sa_blocks)
        if np.any(counts == 0):
            # Invalid, fallback to initial
            sa_blocks = initial_blocks
            
    # Convert to list of lists
    result = [list(blocks[i]) for i in sa_blocks]
    
    return result
