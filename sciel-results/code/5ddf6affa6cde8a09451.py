import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    # Reverse index: for each t-set, which blocks cover it
    # We use a list of lists for efficiency in local search
    ts_to_blocks = [[] for _ in range(num_tsets)]

    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            idx = tset_to_idx[sub]
            BlockTsets[i, count] = idx
            ts_to_blocks[idx].append(i)
            count += 1

    # Convert BlockTsets to a list of lists for faster iteration in python loops if needed,
    # but keep the numpy array for vectorized gain calculation in greedy.
    # Actually, for the local search swap, we need to iterate over the t-sets of a block.
    # BlockTsets is (num_blocks, t_subsets_per_block).
    # Let's create a list of arrays for easier access.
    block_tsets_list = [BlockTsets[i] for i in range(num_blocks)]

    def polish_blocks(blocks):
        """Remove redundant blocks from the end of the list."""
        if not blocks:
            return blocks

        # Count how many times each t-set is covered
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[block_tsets_list[b]] += 1

        # Iterate from end to start
        i = len(blocks) - 1
        while i >= 0:
            if time.time() > deadline - 2.0:
                break
            b = blocks[i]
            # Check if any t-set is covered only by this block
            # If cover_count[ts] == 1 for any ts in block b, we cannot remove b
            can_remove = True
            for ts in block_tsets_list[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block b
                for ts in block_tsets_list[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
        return blocks

    def local_search_swap(blocks):
        """
        Try to replace a block with another block that covers all the t-sets
        uniquely covered by the original block.
        Returns (blocks, improved)
        """
        if not blocks:
            return blocks, False

        # Compute cover counts
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[block_tsets_list[b]] += 1

        improved = False
        # We iterate through the blocks. If we make a swap, we restart the loop
        # because the cover counts have changed.
        # To avoid infinite loops or excessive time, we limit the number of swaps or time.
        
        # We can randomize the order of checking blocks to diversify
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for idx_pos in indices:
            if time.time() > deadline - 2.0:
                break
                
            i = idx_pos
            if i >= len(blocks): continue # Safety, though indices are valid at start
            
            b = blocks[i]
            
            # Find t-sets uniquely covered by block b
            unique_ts = []
            for ts in block_tsets_list[b]:
                if cover_count[ts] == 1:
                    unique_ts.append(ts)
            
            if not unique_ts:
                continue

            # Find candidate blocks that cover ALL unique_ts
            # Start with the blocks covering the first unique t-set
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break

            if not candidates:
                continue

            # Remove the block itself from candidates
            candidates.discard(b)

            if not candidates:
                continue

            # Pick a candidate. 
            # Heuristic: Pick the one that covers the most "already covered" t-sets? 
            # Or just pick randomly to diversify?
            # Let's pick the one that maximizes the number of t-sets it covers 
            # that are currently covered by >1 block (increasing redundancy elsewhere is bad, 
            # but we just need valid coverage).
            # Actually, any valid replacement works for the count.
            # However, a "better" replacement might be one that allows further reductions.
            # For now, let's pick randomly from candidates to explore.
            
            # Optimization: Sort candidates by how many t-sets they share with the rest of the solution?
            # That's expensive. Let's just pick one.
            
            candidate_list = list(candidates)
            c = candidate_list[random.randint(0, len(candidate_list) - 1)]
            
            # Perform swap
            # Update cover_count
            for ts in block_tsets_list[b]:
                cover_count[ts] -= 1
            for ts in block_tsets_list[c]:
                cover_count[ts] += 1
            
            blocks[i] = c
            improved = True
            break # Restart the loop after a change
        
        return blocks, improved

    def run_greedy(seed=None):
        """
        Greedy set cover with random tie-breaking.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        # Pre-compute initial gains? No, gains change.
        # Gain of a block is the number of uncovered t-sets it covers.
        
        while uncovered.any():
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gain for all blocks
            # gains[b] = sum(uncovered[BlockTsets[b]])
            gains = uncovered[BlockTsets].sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly select one to diversify
            best_block_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            blocks.append(best_block_idx)
            
            # Update uncovered
            uncovered[BlockTsets[best_block_idx]] = False

        return blocks

    # 2. Main Search Loop
    best_blocks = None
    best_len = float('inf')

    # Initial deterministic greedy (seed 0)
    sol = run_greedy(seed=0)
    if sol is not None:
        sol = polish_blocks(sol)
        sol, _ = local_search_swap(sol)
        sol = polish_blocks(sol)
        best_blocks = sol[:]
        best_len = len(sol)

    # Randomized restarts
    # We want to spend most of the time here.
    restart_id = 1
    while time.time() < deadline - 10.0:
        sol = run_greedy(seed=restart_id)
        restart_id += 1
        
        if sol is None:
            break
        
        sol = polish_blocks(sol)
        
        # If this solution is better, or close enough, try local search
        if len(sol) <= best_len + 3:
            sol, _ = local_search_swap(sol)
            sol = polish_blocks(sol)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
                # Optional: try more local search steps on the new best?
                # Let's do one more round of swap+polish if we have time
                if time.time() < deadline - 5.0:
                    sol, _ = local_search_swap(sol)
                    sol = polish_blocks(sol)
                    if len(sol) < best_len:
                        best_len = len(sol)
                        best_blocks = sol[:]

    # Final polish on best solution
    if best_blocks is not None:
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks)
        best_blocks = polish_blocks(best_blocks)

    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
