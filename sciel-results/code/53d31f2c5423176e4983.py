import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # Precomputation
    # 1. All t-subsets (triples)
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # 2. All k-subsets (blocks) and the triples they cover
    # We only need blocks, but we need to map block -> list of triple indices
    # To save memory, we can generate blocks on the fly in the greedy phase, 
    # but for SA we need fast access. 
    # Let's precompute the mapping from block (tuple) to list of triple indices.
    # There are C(24,7) = 346,104 blocks.
    
    # Optimization: We don't need to store all blocks if we can generate them.
    # But for SA, we need to pick random blocks. 
    # Let's store the list of block-triple mappings.
    
    # To speed up precomputation, we can use the fact that a block covers 
    # all triples formed by its elements.
    
    # We will store:
    # block_list: list of tuples (the blocks)
    # block_triples: list of lists of triple indices
    
    # However, 346k * 35 ints is ~12M ints. ~48MB. Acceptable.
    
    # Let's try to do this efficiently.
    # Instead of storing all, maybe we can just use the greedy construction 
    # and then local search on the *found* blocks + candidates?
    # No, SA needs a pool of candidates.
    
    # Let's precompute.
    
    block_triples_map = {} # block_tuple -> list of triple_indices
    all_blocks = []
    
    # This loop might take a few seconds.
    for b in itertools.combinations(range(v), k):
        triples = []
        for ts in itertools.combinations(b, t):
            triples.append(tsubset_index[ts])
        block_triples_map[b] = triples
        all_blocks.append(b)
        
    n_blocks = len(all_blocks)
    
    # Map block index to block tuple and vice versa for fast lookup in SA
    # We'll work with indices of all_blocks for SA.
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Helper to get triple indices for a block index
    # block_triples_list[i] = list of triple indices for all_blocks[i]
    block_triples_list = [block_triples_map[b] for b in all_blocks]
    
    # Greedy Construction
    # cov_count: number of times each triple is covered
    cov_count = np.zeros(n_tsubsets, dtype=np.int32)
    selected_blocks = [] # list of block indices
    selected_set = set() # set of block indices for O(1) check
    
    # For greedy, we want to pick blocks that cover many uncovered triples.
    # We can iterate through all blocks, find the one with max gain, add it.
    # But C(24,7) is 346k. Doing this 76 times is 26M operations. Fast enough.
    
    # To make it faster, we can use a heuristic:
    # 1. Find an uncovered triple.
    # 2. Pick a block containing it that covers the most uncovered triples.
    
    # Let's use the "pick random uncovered triple" strategy to diversify.
    rng = random.Random(42)
    
    # Precompute for each triple, the list of block indices that contain it.
    # triple_blocks: list of lists of block indices
    triple_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, b in enumerate(all_blocks):
        for t_idx in block_triples_list[b_idx]:
            triple_blocks[t_idx].append(b_idx)
            
    # Greedy Phase
    while True:
        if time.time() - start_time > deadline - 30:
            break
            
        # Find an uncovered triple
        uncovered_indices = np.where(cov_count == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered triple
        t_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        
        # Get all blocks containing this triple
        candidate_blocks = triple_blocks[t_idx]
        
        # Find the block that covers the most uncovered triples
        best_block = -1
        best_gain = -1
        
        # Optimization: We can sample if candidate_blocks is large, but it's small (C(21,4)=5985 max)
        for b_idx in candidate_blocks:
            if b_idx in selected_set:
                continue
            # Count uncovered triples covered by this block
            gain = 0
            for t_i in block_triples_list[b_idx]:
                if cov_count[t_i] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            # Should not happen if coverage is possible
            break
            
        # Add the block
        selected_set.add(best_block)
        selected_blocks.append(best_block)
        for t_i in block_triples_list[best_block]:
            cov_count[t_i] += 1
            
    # Local Search: Simulated Annealing
    # Objective: Minimize len(selected_blocks). Secondary: Minimize locked triples (cov_count == 1).
    
    def get_score(blocks_list, cov_arr):
        n_locked = np.sum(cov_arr == 1)
        return len(blocks_list) * 10000 + n_locked
        
    current_blocks = list(selected_blocks)
    current_set = set(selected_blocks)
    current_cov = cov_count.copy()
    current_score = get_score(current_blocks, current_cov)
    
    best_blocks = list(current_blocks)
    best_set = set(current_blocks)
    best_cov = current_cov.copy()
    best_score = current_score
    
    T = 100.0
    cooling = 0.99
    min_T = 0.1
    
    steps = 0
    while time.time() - start_time < deadline - 5:
        steps += 1
        
        # Choose a move: Remove, Add, or Swap
        move_type = rng.choice(['remove', 'add', 'swap'])
        
        new_blocks = list(current_blocks)
        new_set = set(current_set)
        new_cov = current_cov.copy()
        
        valid_move = True
        delta_score = 0
        
        if move_type == 'remove':
            if len(current_blocks) == 0:
                continue
            # Pick a random block to remove
            b_idx = current_blocks[rng.randint(0, len(current_blocks) - 1)]
            
            # Check if removal is valid (no triple becomes 0)
            # A triple becomes 0 if its count was 1 and it is covered by b_idx.
            removable = True
            for t_i in block_triples_list[b_idx]:
                if current_cov[t_i] == 1:
                    removable = False
                    break
                    
            if not removable:
                continue
                
            # Perform removal
            new_set.remove(b_idx)
            new_blocks.remove(b_idx)
            for t_i in block_triples_list[b_idx]:
                new_cov[t_i] -= 1
                
        elif move_type == 'add':
            # Add a random block not in the set
            # This increases N, so usually bad, but might enable better swaps?
            # Actually, adding a block never helps reduce N directly. 
            # It only helps if we then remove two blocks. 
            # For simplicity, we skip 'add' in standard covering SA unless we allow temporary invalidity.
            # We will stick to valid moves. 'Add' is never beneficial for minimizing N in a valid state.
            continue
            
        else: # swap
            if len(current_blocks) == 0:
                continue
            b_old = current_blocks[rng.randint(0, len(current_blocks) - 1)]
            
            # Find triples that are "locked" by b_old (cov_count == 1)
            locked_triples = []
            for t_i in block_triples_list[b_old]:
                if current_cov[t_i] == 1:
                    locked_triples.append(t_i)
                    
            # Pick a random new block
            b_new = all_blocks[rng.randint(0, n_blocks - 1)]
            if b_new in current_set:
                continue
                
            # Check if b_new covers all locked_triples
            if locked_triples:
                # Check if b_new covers all t in locked_triples
                # We can do this by checking if all t are in block_triples_list[b_new]
                # To speed up, we can use a set for b_new's triples
                new_block_triples_set = set(block_triples_list[b_new])
                for t_i in locked_triples:
                    if t_i not in new_block_triples_set:
                        break
                else:
                    # Valid swap
                    # Perform swap
                    new_set.remove(b_old)
                    new_set.add(b_new)
                    new_blocks[new_blocks.index(b_old)] = b_new
                    
                    # Update cov
                    for t_i in block_triples_list[b_old]:
                        new_cov[t_i] -= 1
                    for t_i in block_triples_list[b_new]:
                        new_cov[t_i] += 1
                # If break, invalid
                if any(t_i not in new_block_triples_set for t_i in locked_triples):
                    continue
            
            else:
                # No locked triples, any swap is valid
                new_set.remove(b_old)
                new_set.add(b_new)
                new_blocks[new_blocks.index(b_old)] = b_new
                
                for t_i in block_triples_list[b_old]:
                    new_cov[t_i] -= 1
                for t_i in block_triples_list[b_new]:
                    new_cov[t_i] += 1
        
        # Calculate new score
        new_score = get_score(new_blocks, new_cov)
        delta_score = new_score - current_score
        
        # Acceptance
        if delta_score < 0 or (T > min_T and rng.random() < np.exp(-delta_score / T)):
            current_blocks = new_blocks
            current_set = new_set
            current_cov = new_cov
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_blocks = list(current_blocks)
                best_set = set(current_blocks)
                best_cov = current_cov.copy()
                
        T *= cooling
        
    # Return the best solution
    # Convert block indices back to lists of points
    result = [list(all_blocks[i]) for i in best_blocks]
    
    # Final cleanup: remove any redundant blocks in the best solution
    # Just in case the SA left some
    final_blocks = []
    final_cov = np.zeros(n_tsubsets, dtype=np.int32)
    for b_idx in best_blocks:
        final_blocks.append(b_idx)
        for t_i in block_triples_list[b_idx]:
            final_cov[t_i] += 1
            
    # Remove redundant
    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks) - 1, -1, -1):
            b_idx = final_blocks[i]
            removable = True
            for t_i in block_triples_list[b_idx]:
                if final_cov[t_i] == 1:
                    removable = False
                    break
            if removable:
                for t_i in block_triples_list[b_idx]:
                    final_cov[t_i] -= 1
                final_blocks.pop(i)
                changed = True
                
    result = [list(all_blocks[i]) for i in final_blocks]
    
    return result

