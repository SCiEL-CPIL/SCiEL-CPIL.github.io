import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    For the specific task: C(19, 8, 5).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    # --- Precomputation ---
    # Enumerate all k-subsets of {0, ..., v-1}
    # For v=19, k=8: C(19,8) = 75582. This is manageable in memory.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Enumerate all t-subsets of {0, ..., v-1}
    # For v=19, t=5: C(19,5) = 11628.
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    
    # Map t-subsets to indices for O(1) lookup
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    
    # For each block, list the indices of the t-subsets it covers
    # block_ts_list[i] = list of ts indices covered by all_blocks[i]
    block_ts_list = []
    # Also build a reverse index: for each ts, which blocks cover it?
    # ts_blocks_list[j] = list of block indices that cover all_ts[j]
    ts_blocks_list = [[] for _ in range(n_ts)]
    
    for i, block in enumerate(all_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index[ts]
            ts_list.append(idx)
            ts_blocks_list[idx].append(i)
        block_ts_list.append(ts_list)
        
    # Precompute sets for faster intersection if needed, but lists are fine for now.
    # Convert to numpy arrays for vectorization? Maybe later if needed.
    
    def greedy_cover(strategy='rarest', seed=42, deadline=None):
        """
        Greedy set cover.
        strategy: 'rarest' (pick block covering most uncovered, tie-break by rarity?), 
                  'random' (random restart of greedy logic?)
        Actually, standard greedy: at each step, pick block with max gain (new uncovered ts covered).
        Optimization: maintain 'uncovered' bitset or boolean array.
        """
        if deadline is None:
            deadline = 115.0
            
        rng = random.Random(seed)
        uncovered = [1] * n_ts  # 1 if uncovered, 0 if covered
        covered_count = 0
        blocks_used = []
        
        # To speed up gain calculation, we can pre-sort or use numpy.
        # For v=19, k=8, t=5:
        # Each block covers C(8,5) = 56 ts.
        # Total ts = 11628.
        # Greedy steps ~ 300-400.
        # Each step: iterate n_blocks (75582), for each check 56 ts.
        # 75582 * 56 * 400 = ~1.7 billion operations. Too slow in pure Python.
        
        # We need a better approach.
        # Option 1: Use numpy for gain calculation.
        # Option 2: Use a priority queue or incremental updates.
        # Option 3: Limit the candidate set.
        
        # Let's try numpy vectorization for the gain step.
        # uncovered is a numpy array of 0/1.
        # For each block i, gain = sum(uncovered[j] for j in block_ts_list[i])
        # We can pre-store block_ts_list as a list of arrays?
        # Or, create a sparse matrix?
        # Actually, since n_blocks is ~75k and n_ts is ~11k, a dense matrix is 75k * 11k = 825M floats. Too big (3.3 GB).
        # Sparse is better.
        
        # Let's stick to optimized Python with early exit or sampling if full scan is too slow.
        # But 1.7B ops is definitely too slow for 120s.
        
        # Revised Strategy:
        # Instead of scanning ALL blocks every time, we can maintain a list of candidate blocks.
        # However, the set of candidates shrinks slowly.
        
        # Alternative: Use a "max gain" heuristic with limited search or randomized restarts.
        # But we need a valid cover.
        
        # Let's try a different greedy variant:
        # "Rarest First":
        # 1. Find the uncovered t-subset with the fewest covering blocks (among uncovered).
        # 2. Pick one of the blocks covering it that covers the most OTHER uncovered t-subsets.
        # This is often faster because the number of candidates for the rarest ts is small.
        
        # Let's implement Rarest First.
        
        # Data structures:
        # uncovered_count[ts_idx]: how many blocks cover this ts (static)
        # covered[ts_idx]: boolean
        # active_blocks[ts_idx]: list of blocks that cover this ts and are NOT yet used?
        # No, blocks can be reused? No, distinct blocks. But a block is either used or not.
        # We need to track which blocks are still available.
        
        # Let's maintain:
        # available_blocks: set of indices of all_blocks not yet chosen.
        # For each ts, a list of available blocks covering it.
        
        # This is getting complex to maintain dynamically.
        
        # Let's try a simpler, but effective heuristic:
        # Randomized Greedy with multiple restarts.
        # At each step, pick a random uncovered ts, then pick the best block covering it.
        
        # Even better:
        # Standard Greedy but with a "lookahead" or limited candidate pool.
        
        # Let's try to optimize the full scan using NumPy.
        # Create a sparse representation?
        # scipy.sparse is available.
        
        try:
            from scipy.sparse import csr_matrix
            # Build incidence matrix: rows=blocks, cols=ts
            # M[i, j] = 1 if block i covers ts j
            # n_blocks = 75582, n_ts = 11628
            # nnz = n_blocks * 56 = 4.2M. This is small enough for CSR.
            
            row_ind = []
            col_ind = []
            data = []
            
            for i, ts_list in enumerate(block_ts_list):
                for j in ts_list:
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(1)
                    
            M = csr_matrix((data, (row_ind, col_ind)), shape=(n_blocks, n_ts))
            
            def greedy_numpy(seed=42):
                global deadline
                uncovered = np.ones(n_ts, dtype=np.int8)
                blocks_used = []
                
                while np.sum(uncovered) > 0 and time.time() < deadline - 10:
                    # Calculate gain for all blocks
                    # gain[i] = sum(M[i, :] * uncovered)
                    # Since M is sparse, M.dot(uncovered) is efficient.
                    gains = M.dot(uncovered).A1  # Returns 1D array
                    
                    # Find max gain
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break  # No block covers any new ts? Should not happen if valid.
                        
                    # Pick a block with max gain.
                    # To add diversity, if multiple blocks have same max gain, pick randomly.
                    candidates = np.where(gains == max_gain)[0]
                    if len(candidates) > 1:
                        idx = rng.choice(candidates)
                    else:
                        idx = candidates[0]
                        
                    blocks_used.append(all_blocks[idx])
                    
                    # Update uncovered
                    # M[idx, :] contains the ts covered by this block.
                    # We need to zero out those positions in uncovered.
                    # Get non-zero columns of row idx.
                    # M.getrow(idx)
                    row = M.getrow(idx).tocsr()
                    cols = row.indices
                    uncovered[cols] = 0
                    
                return blocks_used

            # Let's test this approach mentally.
            # M.dot(uncovered) for CSR matrix:
            # Complexity is O(nnz_row) for each row? No, it's a full vector-matrix product.
            # It iterates over non-zeros. Total nnz is 4.2M.
            # So each greedy step takes ~4.2M ops.
            # 300 steps * 4.2M = 1.26B ops. Still might be tight in Python/NumPy overhead.
            # But NumPy is C-optimized. 1.26B simple integer adds/mults should be fast (< 1-2s).
            # Let's try it.
            
            # However, we need to be careful with the first run.
            # Let's run a few restarts with different seeds.
            
            best_blocks = None
            best_count = float('inf')
            
            # Run 5-10 restarts
            num_restarts = 10
            for r in range(num_restarts):
                if time.time() > deadline - 30:
                    break
                    
                seed = 42 + r
                blocks = greedy_numpy(seed=seed)
                
                if len(blocks) < best_count:
                    best_blocks = blocks
                    best_count = len(blocks)
                    
            if best_blocks is None:
                # Fallback: simple greedy
                best_blocks = greedy_numpy(seed=42)
                
            # --- Local Search / Post-processing ---
            # 1. Remove redundant blocks.
            # 2. Try to swap blocks to reduce count? (Hard for set cover, usually you can't swap to reduce count easily without complex local search).
            # 3. Try to find a smaller set by re-solving on remaining uncovered?
            
            def remove_redundant(blocks):
                # Convert to list of lists for mutability
                blocks = [list(b) for b in blocks]
                # Build coverage count
                # Use a dict or array.
                # Array of size n_ts.
                cover_count = np.zeros(n_ts, dtype=np.int32)
                
                # Map blocks to their ts lists for fast access
                # We need to map a tuple block to its index in all_blocks?
                # Or just recompute ts for each block?
                # Recalculating is C(8,5)=56 ops. Fast.
                
                for b in blocks:
                    b_set = set(b)
                    for ts in itertools.combinations(b, t):
                        idx = ts_index[ts]
                        cover_count[idx] += 1
                        
                new_blocks = []
                for b in blocks:
                    b_set = set(b)
                    redundant = True
                    for ts in itertools.combinations(b, t):
                        idx = ts_index[ts]
                        if cover_count[idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove from cover_count
                        for ts in itertools.combinations(b, t):
                            idx = ts_index[ts]
                            cover_count[idx] -= 1
                    else:
                        new_blocks.append(b)
                        
                return new_blocks

            # Apply redundancy removal to the best block set
            best_blocks = remove_redundant(best_blocks)
            
            # Iterative improvement:
            # If there's time, try local search:
            # - Randomly remove a block, see if cover breaks. If yes, try to fix by adding a different block.
            # - Or, run greedy again on the uncovered parts?
            
            # A stronger local search: "Swap"
            # Try replacing a block B with a block B' such that:
            # 1. All ts uniquely covered by B are also covered by B'.
            # 2. B' covers some ts that are currently over-covered (count > 1), potentially allowing other blocks to be removed later?
            # This is complex.
            
            # Let's try a simpler "shake" and re-greedy.
            # 1. Take the current cover.
            # 2. Randomly remove 1-5 blocks.
            # 3. Run greedy to cover the newly uncovered ts.
            # 4. If the total count is lower, keep it.
            
            if time.time() < deadline - 15:
                current_blocks = best_blocks
                current_count = len(current_blocks)
                improved = True
                max_improve_iter = 50
                
                for _ in range(max_improve_iter):
                    if time.time() > deadline - 15:
                        break
                    if not improved:
                        break
                    improved = False
                    
                    # Pick 1-3 random blocks to remove
                    n_remove = rng.randint(1, 3)
                    if n_remove >= current_count:
                        break
                        
                    remove_idx = rng.sample(range(current_count), n_remove)
                    remove_set = set(remove_idx)
                    
                    # Identify ts that become uncovered
                    # First, compute coverage of remaining blocks
                    cover_count = np.zeros(n_ts, dtype=np.int32)
                    for i, b in enumerate(current_blocks):
                        if i in remove_set:
                            continue
                        b_set = set(b)
                        for ts in itertools.combinations(b, t):
                            idx = ts_index[ts]
                            cover_count[idx] += 1
                            
                    # Now, which ts are uncovered?
                    # We need to know which ts were ONLY covered by the removed blocks.
                    # Let's compute the coverage of the removed blocks specifically.
                    # Actually, easier:
                    # 1. Build cover_count for ALL blocks.
                    # 2. Subtract cover_count of removed blocks.
                    # 3. If result < 0, that ts was not covered by others? No, cover_count was for all.
                    # If cover_count_all[ts] - cover_count_removed[ts] == 0, then it's uncovered.
                    
                    # Let's redo:
                    cover_all = np.zeros(n_ts, dtype=np.int32)
                    for b in current_blocks:
                        b_set = set(b)
                        for ts in itertools.combinations(b, t):
                            idx = ts_index[ts]
                            cover_all[idx] += 1
                            
                    cover_rem = np.zeros(n_ts, dtype=np.int32)
                    for i in remove_idx:
                        b = current_blocks[i]
                        b_set = set(b)
                        for ts in itertools.combinations(b, t):
                            idx = ts_index[ts]
                            cover_rem[idx] += 1
                            
                    # Uncovered ts are those where cover_all[ts] == cover_rem[ts]
                    # i.e., the removed blocks were the ONLY ones covering them.
                    uncovered_mask = (cover_all == cover_rem).astype(np.int8)
                    
                    if np.sum(uncovered_mask) == 0:
                        # All removed blocks were redundant!
                        current_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
                        current_count = len(current_blocks)
                        improved = True
                        continue
                        
                    # Now, run greedy to cover the uncovered_mask
                    # We need to restrict the greedy search to ONLY cover the uncovered ts.
                    # And we prefer blocks that are already in the set? No, we want to minimize ADDITIONS.
                    # So, run greedy on the subset of ts defined by uncovered_mask.
                    
                    # Create a new incidence matrix restricted to these ts?
                    # Or just use the full matrix but mask the gains.
                    
                    # Gains should only count the uncovered ts.
                    # gains = M.dot(uncovered_mask)
                    
                    # We want to add FEW blocks.
                    # Greedy is good at this.
                    
                    # However, we must ensure we don't add a block that is already in current_blocks.
                    # We can just let greedy pick, and then check for duplicates?
                    # Or, mark used blocks as having gain -1?
                    
                    # Let's create a set of used block indices.
                    used_indices = set()
                    # We need to map blocks to indices.
                    # all_blocks is a list of tuples.
                    # Create a dict: block_tuple -> index
                    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
                    
                    for b in current_blocks:
                        idx = block_to_idx[tuple(b)]
                        used_indices.add(idx)
                        
                    # Modify M or gains to penalize used blocks?
                    # If we add a used block, it's a duplicate. We can just skip it in the final merge.
                    # But greedy might pick it.
                    # Let's just run greedy and then remove duplicates.
                    
                    # Custom greedy for specific uncovered set
                    blocks_to_add = []
                    current_uncovered = uncovered_mask.copy()
                    
                    while np.sum(current_uncovered) > 0 and time.time() < deadline - 15:
                        gains = M.dot(current_uncovered).A1
                        # Penalize used blocks?
                        # If a block is already used, adding it again is useless.
                        # So set gains[used_indices] = 0?
                        # No, if it covers new things, it's still valid, just redundant in the final set.
                        # But we want to MINIMIZE the number of NEW blocks.
                        # So yes, set gains of used blocks to 0 (or -inf) to force selection of new blocks?
                        # Actually, if a used block covers the most, it might be the best choice.
                        # But we can't add it again.
                        # So we MUST pick a new block.
                        # So, set gains of used blocks to -1?
                        # And ensure there is always a valid new block.
                        
                        # Let's just mask used blocks.
                        gains_masked = gains.copy()
                        for idx in used_indices:
                            gains_masked[idx] = 0  # Effectively ignore them for selection?
                            # Wait, if ALL candidates are used, we are stuck.
                            # But if the cover was valid, and we only removed some, there must be new blocks that cover the rest?
                            # Not necessarily. The "new" blocks might be the only ones.
                            # So we should allow used blocks to be selected, but then we don't add them.
                            # Actually, if we select a used block, we don't add it to `blocks_to_add`.
                            # But we still update `current_uncovered`.
                            # This effectively means we are "re-using" the block's coverage.
                            # This is correct!
                            
                        # So, no masking needed for selection logic, just don't add duplicates.
                        
                        max_gain = np.max(gains)
                        if max_gain == 0:
                            break # Stuck?
                            
                        candidates = np.where(gains == max_gain)[0]
                        idx = rng.choice(candidates)
                        
                        # If this block is not already used, add it to the list of blocks to add
                        if idx not in used_indices:
                            blocks_to_add.append(all_blocks[idx])
                            used_indices.add(idx)
                        
                        # Update current_uncovered
                        row = M.getrow(idx).tocsr()
                        cols = row.indices
                        current_uncovered[cols] = 0
                        
                    # Merge
                    new_blocks = current_blocks
                    # Remove the ones we dropped
                    new_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
                    # Add the new ones
                    new_blocks.extend(blocks_to_add)
                    
                    if len(new_blocks) < current_count:
                        current_blocks = new_blocks
                        current_count = len(current_blocks)
                        improved = True
                        
                best_blocks = current_blocks
                
            return best_blocks

        except ImportError:
            # Fallback to pure Python if scipy is not available (unlikely, but safe)
            # Use the simple greedy from before, but it might be slow.
            # Let's just return a basic greedy result.
            
            def greedy_py(seed=42):
                rng = random.Random(seed)
                uncovered = [1] * n_ts
                blocks_used = []
                
                while True:
                    remaining = [i for i, u in enumerate(uncovered) if u]
                    if not remaining:
                        break
                    if time.time() > deadline - 10:
                        break
                    
                    # Pick random uncovered ts
                    target_ts = rng.choice(remaining)
                    
                    # Find best block covering it
                    best_block = None
                    best_gain = -1
                    for b_idx in ts_blocks_list[target_ts]:
                        gain = 0
                        for ts_idx in block_ts_list[b_idx]:
                            if uncovered[ts_idx]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = b_idx
                            
                    if best_block is None:
                        break
                        
                    blocks_used.append(all_blocks[best_block])
                    for ts_idx in block_ts_list[best_block]:
                        uncovered[ts_idx] = 0
                        
                return blocks_used

            best_blocks = greedy_py(42)
            # Remove redundant
            def remove_redundant_py(blocks):
                cover_count = defaultdict(int)
                for b in blocks:
                    b_set = set(b)
                    for ts in itertools.combinations(b, t):
                        cover_count[ts] += 1
                        
                new_blocks = []
                for b in blocks:
                    b_set = set(b)
                    redundant = True
                    for ts in itertools.combinations(b, t):
                        if cover_count[ts] == 1:
                            redundant = False
                            break
                    if redundant:
                        for ts in itertools.combinations(b, t):
                            cover_count[ts] -= 1
                    else:
                        new_blocks.append(b)
                return new_blocks

            best_blocks = remove_redundant_py(best_blocks)
            return best_blocks

# End of construct
