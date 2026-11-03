import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block_tsubs: for each block, list of t-subset indices it covers
    # Also build the incidence matrix A (sparse)
    # A is n_tsubsets x n_blocks
    row_indices = []
    col_indices = []
    block_tsubs = [[] for _ in range(n_blocks)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            tsubs.append(idx)
            row_indices.append(idx)
            col_indices.append(bi)
        block_tsubs[bi] = tsubs
        
    data = np.ones(len(row_indices), dtype=np.float64)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_tsubsets, n_blocks))
    
    # For faster local search, we need to quickly check if a block covers certain t-subsets
    # We can use the block_tsubs list.
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(seed=0):
        """Greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Precompute gain for each block (initially number of t-subsets it covers)
        gain = np.array([len(tsubs) for tsubs in block_tsubs], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(chosen) # Store index for efficiency
            
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Update gains for all blocks covering this t-subset
                    # This is expensive if done naively. 
                    # However, n_blocks is ~50k. 
                    # We can optimize by precomputing tsub_blocks
                    # But we didn't build tsub_blocks explicitly as a list of lists for speed of construction?
                    # Let's build tsub_blocks for efficient gain updates.
            
            gain[chosen] = -1
        
        return blocks_used

    # Let's rebuild with tsub_blocks for efficient gain updates in greedy
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    def greedy_cover_fast(seed=0):
        """Optimized Greedy set cover."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Initial gain
        gain = np.array([len(tsubs) for tsubs in block_tsubs], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(chosen)
            
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            
            gain[chosen] = -1
        
        return blocks_used

    def local_search(block_indices, seed=0):
        """
        Local search:
        1. Remove redundant blocks.
        2. Try 1-for-1 swaps to enable more removals.
        """
        if not block_indices:
            return block_indices
            
        # 1. Redundancy Removal
        # Count coverage of each t-subset by current blocks
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_indices:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        kept = [True] * len(block_indices)
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(block_indices):
                if not kept[i]:
                    continue
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    changed = True
                    
        current_blocks = [block_indices[i] for i in range(len(block_indices)) if kept[i]]
        
        # 2. 1-for-1 Swap Phase
        # We try to swap a block B with B' such that B' covers all unique t-subsets of B.
        # This might allow other blocks to become redundant.
        # To keep it fast, we limit the number of swap attempts.
        
        improved = True
        max_swap_iterations = 50 # Limit to prevent hanging
        iteration = 0
        
        while improved and iteration < max_swap_iterations and time.time() < deadline - 5:
            iteration += 1
            improved = False
            
            # Recalculate cover counts for current state
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in current_blocks:
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] += 1
                
            # Identify blocks that are "critical" (cover at least one unique t-subset)
            # We want to swap non-critical or critical blocks if it helps.
            # Actually, we can try swapping any block.
            
            # To speed up, we only try swaps for a subset of blocks (e.g., first 100)
            # Or we try all if the list is small.
            # Let's try all blocks in the current solution.
            
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 5:
                    break
                
                # Find t-subsets uniquely covered by bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                
                # If no unique t-subsets, bi is redundant, but we already removed those.
                # So unique_tsubs is not empty.
                
                # Find candidate blocks B' that cover all unique_tsubs
                # And are not in current_blocks
                
                # Optimization: Precompute for each t-subset, the blocks covering it.
                # We need intersection of tsub_blocks[ts] for ts in unique_tsubs.
                
                # Start with the rarest unique t-subset to minimize intersection size
                # Sort unique_tsubs by frequency (length of tsub_blocks[ts])
                unique_tsubs_sorted = sorted(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                
                if not unique_tsubs_sorted:
                    continue
                    
                candidates = set(tsub_blocks[unique_tsubs_sorted[0]])
                for ts_idx in unique_tsubs_sorted[1:]:
                    candidates &= set(tsub_blocks[ts_idx])
                    if not candidates:
                        break
                
                # Remove current blocks from candidates
                current_set = set(current_blocks)
                candidates -= current_set
                
                if not candidates:
                    continue
                
                # Try a random candidate
                # If the swap leads to a reduction in size, accept it.
                # We check this by simulating the swap and running redundancy removal.
                
                # To avoid O(N^2) behavior, we only try a few random candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                for new_bi in cand_list[:5]: # Try up to 5 candidates
                    # Simulate swap
                    # Remove bi, add new_bi
                    # Check if any block becomes redundant
                    
                    # We can approximate: 
                    # If new_bi covers more "shared" t-subsets, it might help.
                    # But exact check is safer.
                    
                    # Let's just accept the swap if it doesn't increase size (it doesn't) 
                    # and hope that subsequent redundancy removal helps.
                    # To be more aggressive, we can check if the new block allows 
                    # at least one other block to become redundant.
                    
                    # This check is expensive. Let's just do the swap and 
                    # run redundancy removal in the next iteration of the while loop.
                    # But that might be slow.
                    
                    # Alternative: Only accept if new_bi covers at least one t-subset 
                    # that was covered by 2 or more blocks, and bi was critical.
                    # This is heuristic.
                    
                    # Let's stick to a simpler heuristic: 
                    # Accept swap if new_bi is "better" in terms of coverage overlap.
                    # Or just accept the first valid swap and restart redundancy removal.
                    
                    # Let's accept the first valid swap found.
                    current_blocks[i] = new_bi
                    improved = True
                    break
                if improved:
                    break # Restart redundancy removal loop
                    
        return current_blocks

    # Strategy: Multiple Greedy + Local Search
    num_trials = 20
    for seed in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        blocks_idx = greedy_cover_fast(seed)
        if not blocks_idx:
            continue
            
        # Convert indices to actual blocks for now, but keep indices for local search
        blocks_idx = local_search(blocks_idx, seed)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = [all_blocks[i] for i in blocks_idx]
            
        if best_len <= 140: # Good enough?
            break

    # If we have a solution, try MILP with warm start
    if best_blocks and time.time() < deadline - 10:
        # Prepare MILP
        c = np.ones(n_blocks, dtype=np.float64)
        lb = np.ones(n_tsubsets, dtype=np.float64)
        ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_blocks, dtype=np.int8)
        bounds = Bounds(0, 1)
        
        # Warm start: x[i] = 1 if block i is in best_blocks
        x0 = np.zeros(n_blocks)
        for block in best_blocks:
            # Find index of block
            # This is O(N) search. We can precompute block_to_idx
            # But we didn't. Let's just use the indices from the last local search if available.
            # Actually, best_blocks is a list of tuples.
            # We need the indices.
            # Let's keep the indices of the best solution found in the loop.
            pass
        
        # We lost the indices. Let's re-derive or store them.
        # Let's modify the loop to store best_indices.
        
    # Re-run with index tracking
    best_indices = None
    best_len = float('inf')
    
    for seed in range(num_trials):
        if time.time() > deadline - 30:
            break
            
        blocks_idx = greedy_cover_fast(seed)
        if not blocks_idx:
            continue
            
        blocks_idx = local_search(blocks_idx, seed)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_indices = blocks_idx
            best_blocks = [all_blocks[i] for i in blocks_idx]
            
        if best_len <= 140:
            break

    if best_indices and time.time() < deadline - 15:
        c = np.ones(n_blocks, dtype=np.float64)
        lb = np.ones(n_tsubsets, dtype=np.float64)
        ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_blocks, dtype=np.int8)
        bounds = Bounds(0, 1)
        
        x0 = np.zeros(n_blocks)
        for i in best_indices:
            x0[i] = 1.0
            
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                options={'time_limit': int(deadline - time.time() - 5), 'mip_gap': 0.01}
            )
            
            if result.status == 0 or result.status == 1:
                x = result.x
                selected_idx = [i for i in range(n_blocks) if x[i] > 0.5]
                if selected_idx and len(selected_idx) < best_len:
                    best_len = len(selected_idx)
                    best_blocks = [all_blocks[i] for i in selected_idx]
        except Exception:
            pass

    if best_blocks is None:
        # Fallback
        blocks_idx = greedy_cover_fast(0)
        blocks_idx = local_search(blocks_idx, 0)
        best_blocks = [all_blocks[i] for i in blocks_idx]
        
    return best_blocks
