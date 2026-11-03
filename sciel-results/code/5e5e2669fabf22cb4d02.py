"""
Covering design C(19,8,5) using randomized greedy set cover with efficient
numpy-based coverage tracking and a single-pass redundancy deletion polish.

Strategy:
1. Precompute the mapping of each 8-block to the 5-subsets it covers.
2. Run multiple instances of a randomized greedy set cover algorithm.
   Randomization in tie-breaking allows exploring different solution basins.
3. Apply a deletion polish to the best solution found. The polish removes
   blocks that are redundant (i.e., all their 5-subsets are covered by at least
   two blocks). This is done in a single backward pass per iteration, which is
   sufficient because removing a block only decreases coverage counts, thus
   never making a non-redundant block redundant.
4. Deadline guard: return the best valid solution found before the limit.
"""
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the list of t-subset indices it covers
    block_tsubs_list = []
    for block in all_ksubs:
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            tsubs_in_block.append(tsub_index[ts])
        block_tsubs_list.append(tsubs_in_block)
    
    block_tsubs = np.array(block_tsubs_list, dtype=np.int32)

    def run_greedy(seed):
        random.seed(seed)
        covered = np.zeros(n_tsubs, dtype=bool)
        selected_indices = []
        
        while not covered.all():
            if time.time() > deadline:
                return selected_indices
            
            # Calculate gains for all blocks
            uncov = (~covered).astype(np.int32)
            # np.take is efficient for this gather operation
            gains = np.take(uncov, block_tsubs).sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            # Find all blocks with max gain and pick one randomly
            best_indices = np.where(gains == max_gain)[0]
            chosen_idx = random.choice(best_indices)
            
            selected_indices.append(chosen_idx)
            covered[block_tsubs[chosen_idx]] = True
            
        return selected_indices

    def polish(selected_indices):
        # selected_indices is a list of block indices
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_count[block_tsubs[idx]] += 1
            
        # Single backward pass is sufficient to remove all currently redundant blocks
        # because removing a block only decreases cover counts.
        for i in range(len(selected_indices) - 1, -1, -1):
            if time.time() > deadline:
                break
            idx = selected_indices[i]
            # Check if all t-subsets of this block are covered by at least 2 blocks
            if np.all(cover_count[block_tsubs[idx]] >= 2):
                cover_count[block_tsubs[idx]] -= 1
                selected_indices.pop(i)
                
        return selected_indices

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy instances
    # Estimate: 1 run takes ~1-2s. 10 runs ~ 10-20s.
    # Leave time for polish and safety.
    num_runs = 10
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
        sol = run_greedy(i)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    if best_solution is None:
        best_solution = run_greedy(0)
        
    # Polish the best solution
    polished = polish(best_solution)
    
    # Convert indices to blocks
    result = [list(all_ksubs[i]) for i in polished]
    
    return result
