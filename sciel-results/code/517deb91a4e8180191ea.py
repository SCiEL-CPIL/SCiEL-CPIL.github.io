import itertools
import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # 1. Precomputation
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute block t-sub indices and bitmasks
    block_tsub_indices = []
    block_masks = [0] * num_kblocks
    
    for i, b in enumerate(all_kblocks):
        indices = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsub_indices.append(indices)
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 2. Greedy Construction with Candidate Pool
    uncovered_mask = (1 << num_tsubs) - 1
    selected_blocks = []
    candidate_pool = set()
    
    # Top K blocks to add to pool at each step
    TOP_K = 50
    
    while uncovered_mask:
        if time.monotonic() > deadline - 20.0:
            break
            
        best_gain = 0
        best_block_idx = -1
        # Store top K gains for candidate pool
        # Using a list of (gain, idx) and sorting is O(N log N).
        # N=50k. 50k log 50k ~ 50k * 16 = 800k ops.
        # Doing this 150 times is 120M ops. Might be slow in Python.
        # Alternative: Just track the top K in a heap or list.
        
        # Faster: Just iterate and keep track of top K.
        # Since we just need the best block AND a pool of good blocks,
        # we can do a single pass to find best, and a second pass to get top K?
        # Or one pass to find best and collect candidates?
        
        # Let's just find the best block first.
        # Then, separately, we can add the best block and some random/other good blocks?
        # Actually, the "top K" idea requires sorting or partial selection.
        # Let's simplify: Add the best block to the pool.
        # Also, add blocks that cover the *most* uncovered t-subsets among the remaining?
        # That's what we're doing.
        
        # To get a diverse pool, we can add the best block, and then
        # maybe the 2nd best, 3rd best?
        # Let's just add the best block to the pool.
        # And to ensure the pool is large enough, we can pre-select a set of "high degree" blocks?
        # All blocks have the same degree C(7,4)=35.
        
        # Let's stick to: Add the best block to the pool.
        # After greedy, the pool will have ~150 blocks.
        # This is too small for ILP to find a *better* solution (it can only select from the pool).
        # If the optimal solution uses a block NOT in the greedy path, we miss it.
        
        # Better: During the first few steps, add more candidates.
        # Or, simply use the first 1000 blocks generated? No, they are arbitrary.
        
        # Let's try: Add the best block, and also the blocks that are "neighbors" of the best block.
        # Neighbors: blocks sharing a t-subset with the best block.
        # This requires `tsub_to_blocks` precomputation.
        
    # Let's precompute tsub_to_blocks
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for j, b in enumerate(all_kblocks):
        for idx in block_tsub_indices[j]:
            tsub_to_blocks[idx].append(j)

    # Reset Greedy
    uncovered_mask = (1 << num_tsubs) - 1
    selected_blocks = []
    candidate_pool = set()
    
    while uncovered_mask:
        if time.monotonic() > deadline - 20.0:
            break
            
        best_gain = 0
        best_block_idx = -1
        
        for i in range(num_kblocks):
            common = block_masks[i] & uncovered_mask
            if common:
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    
        if best_gain == 0:
            break
            
        selected_blocks.append(best_block_idx)
        candidate_pool.add(best_block_idx)
        
        # Add neighbors of best_block_idx to candidate pool
        # Neighbors are blocks that share a t-subset with best_block_idx
        for idx in block_tsub_indices[best_block_idx]:
            for neighbor in tsub_to_blocks[idx]:
                candidate_pool.add(neighbor)
                
        # Limit pool size to avoid huge ILP
        if len(candidate_pool) > 5000:
            # Remove some? Keep the ones in selected_blocks and recent additions?
            # For now, just let it grow, we will trim later.
            pass

        uncovered_mask &= ~block_masks[best_block_idx]
        
        if uncovered_mask == 0:
            break
            
    # 3. Prune
    counts = [0] * num_tsubs
    for b_idx in selected_blocks:
        for idx in block_tsub_indices[b_idx]:
            counts[idx] += 1
            
    selected_set = set(selected_blocks)
    changed = True
    while changed and time.monotonic() < deadline - 15.0:
        changed = False
        for b_idx in sorted(selected_set, reverse=True):
            can_remove = True
            for idx in block_tsub_indices[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.discard(b_idx)
                for idx in block_tsub_indices[b_idx]:
                    counts[idx] -= 1
                changed = True
                
    selected_blocks = sorted(selected_set)
    
    # 4. ILP on Candidate Pool
    remaining = deadline - time.monotonic()
    if remaining > 15.0:
        # Ensure all selected blocks are in candidate pool
        for b in selected_blocks:
            candidate_pool.add(b)
            
        # Trim candidate pool if too large
        if len(candidate_pool) > 3000:
            # Prioritize blocks in selected_blocks, then others
            # Simple: just take a subset
            # We need to map indices
            # Let's just take the first 3000?
            # Better: Keep selected_blocks and add random others?
            # Or keep the ones with highest "overlap" with selected_blocks?
            # For simplicity, if > 3000, we might skip ILP or use a smaller subset.
            # Let's try to keep the most "central" ones.
            # Actually, if the pool is too big, ILP will be slow.
            # Let's cap it at 2000.
            # We must ensure selected_blocks are included.
            sel_set = set(selected_blocks)
            others = [b for b in candidate_pool if b not in sel_set]
            # Shuffle others?
            import random
            random.seed(42)
            random.shuffle(others)
            candidate_pool = sel_set.union(set(others[:2000 - len(sel_set)]))
            
        if len(candidate_pool) > 0:
            # Map candidate_pool to 0..N-1
            cand_list = sorted(candidate_pool)
            cand_to_idx = {b: i for i, b in enumerate(cand_list)}
            num_cand = len(cand_list)
            
            # Build reduced incidence matrix
            rows = []
            cols = []
            for j, b_idx in enumerate(cand_list):
                for idx in block_tsub_indices[b_idx]:
                    rows.append(idx)
                    cols.append(j)
                    
            A_red = csc_matrix(( [1]*len(rows), (rows, cols) ), shape=(num_tsubs, num_cand))
            
            # x0 for ILP
            x0 = np.zeros(num_cand)
            for b_idx in selected_blocks:
                if b_idx in cand_to_idx:
                    x0[cand_to_idx[b_idx]] = 1.0
            
            try:
                c = np.ones(num_cand)
                constraints = LinearConstraint(A_red, 1, np.inf)
                integrality = np.ones(num_cand)
                bounds = Bounds(0, 1)
                
                ilp_time = max(5.0, remaining - 5.0)
                res = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    x0=x0,
                    time_limit=ilp_time
                )
                
                if res.x is not None:
                    x_sol = np.round(res.x)
                    if np.all((x_sol == 0) | (x_sol == 1)):
                        coverage = A_red @ x_sol
                        if np.all(coverage >= 1):
                            new_count = int(np.sum(x_sol))
                            old_count = len(selected_blocks)
                            if new_count < old_count:
                                new_selected_indices = [cand_list[i] for i in range(num_cand) if x_sol[i] == 1]
                                # Re-polish
                                counts2 = [0] * num_tsubs
                                for b_idx in new_selected_indices:
                                    for idx in block_tsub_indices[b_idx]:
                                        counts2[idx] += 1
                                sel_set2 = set(new_selected_indices)
                                changed2 = True
                                while changed2 and time.monotonic() < deadline - 5.0:
                                    changed2 = False
                                    for b_idx in sorted(sel_set2, reverse=True):
                                        can_rem = True
                                        for idx in block_tsub_indices[b_idx]:
                                            if counts2[idx] == 1:
                                                can_rem = False
                                                break
                                        if can_rem:
                                            sel_set2.discard(b_idx)
                                            for idx in block_tsub_indices[b_idx]:
                                                counts2[idx] -= 1
                                            changed2 = True
                                selected_blocks = sorted(sel_set2)
                                
            except Exception:
                pass

    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    return result_blocks
