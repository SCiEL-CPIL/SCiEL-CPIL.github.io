import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import heapq

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
    
    # Build tsub_blocks for efficient gain updates
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    best_blocks = None
    best_indices = None
    best_len = float('inf')
    
    def greedy_cover_heap(seed=0):
        """Greedy set cover using a max-heap for gains."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Initial gain: number of t-subsets each block covers
        gain = np.array([len(tsubs) for tsubs in block_tsubs], dtype=np.int32)
        
        # Heap stores (-gain, block_id)
        heap = [(-gain[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Pop max gain block
            neg_g, bi = heapq.heappop(heap)
            g = -neg_g
            
            # Check if gain is still valid (lazy deletion)
            if gain[bi] != g or gain[bi] == 0:
                continue
            
            if g == 0:
                break
            
            blocks_used.append(bi)
            gain[bi] = 0 # Mark as used/removed from consideration
            
            # Update uncovered and gains
            for ts_idx in block_tsubs[bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Update gains for all blocks covering this t-subset
                    for other_bi in tsub_blocks[ts_idx]:
                        if gain[other_bi] > 0:
                            gain[other_bi] -= 1
                            heapq.heappush(heap, (-gain[other_bi], other_bi))
        
        return blocks_used

    def remove_redundant_blocks(block_indices):
        """Remove redundant blocks from the solution."""
        if not block_indices:
            return block_indices
        
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
                    
        return [block_indices[i] for i in range(len(block_indices)) if kept[i]]

    def local_search_2opt(block_indices, seed=0, max_rounds=20):
        """
        Local search with 1-for-1 and 2-for-2 swaps.
        """
        if not block_indices:
            return block_indices
        
        current_blocks = list(block_indices)
        rng = random.Random(seed)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 15:
                break
            
            # Phase 1: Remove redundant blocks (1-opt removal)
            current_blocks = remove_redundant_blocks(current_blocks)
            
            # Compute cover counts
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in current_blocks:
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] += 1
            
            current_set = set(current_blocks)
            improved = False
            
            # Phase 2: Try 1-for-1 swaps
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 15:
                    break
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]
                if not unique_tsubs:
                    continue
                
                unique_tsubs_sorted = sorted(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[unique_tsubs_sorted[0]])
                for ts_idx in unique_tsubs_sorted[1:]:
                    candidates &= set(tsub_blocks[ts_idx])
                    if not candidates:
                        break
                
                candidates -= current_set
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for new_bi in cand_list[:5]:
                    current_blocks[i] = new_bi
                    current_set.discard(bi)
                    current_set.add(new_bi)
                    improved = True
                    break
                
                if improved:
                    break
            
            if improved:
                continue # Restart round
            
            # Phase 3: Try 2-for-2 swaps
            # We try to remove two blocks and add two others
            n_curr = len(current_blocks)
            if n_curr < 2:
                break
                
            # Limit pairs to check to avoid O(N^2) explosion
            # Randomly sample pairs if n_curr is large
            if n_curr > 50:
                pairs = [tuple(sorted(rng.sample(range(n_curr), 2))) for _ in range(100)]
            else:
                pairs = list(itertools.combinations(range(n_curr), 2))
            
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline - 15:
                    break
                    
                bi = current_blocks[i]
                bj = current_blocks[j]
                
                # Calculate unique t-subsets covered by bi or bj but not by others
                # unique_tsubs = (T_bi U T_bj) \ (Union of T_x for x in S \ {bi, bj})
                
                # Efficient check:
                # A t-subset is unique to {bi, bj} if cover_count[ts] == 2 AND (ts in T_bi OR ts in T_bj) AND (ts not in T_x for x in S \ {bi, bj})
                # Actually, if cover_count[ts] == 2, it is covered by exactly 2 blocks in S.
                # If those 2 blocks are bi and bj, then it is unique to the pair.
                
                unique_tsubs = []
                # We need to know which blocks cover each ts.
                # We have cover_count. If cover_count[ts] == 2, we need to check if bi and bj are the ones.
                
                # To do this efficiently, we can just iterate over T_bi and T_bj.
                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                
                for ts in set_bi | set_bj:
                    if cover_count[ts] == 2:
                        # Check if the two blocks covering ts are bi and bj
                        # We don't have the list of blocks per ts easily accessible in O(1) for membership of specific bi/bj
                        # But we can check: is ts in T_bi? is ts in T_bj?
                        # If ts in T_bi and ts in T_bj, then since count is 2, they must be the only ones.
                        # If ts in T_bi and not T_bj, then it's covered by bi and some other block.
                        # If ts in T_bj and not T_bi, then it's covered by bj and some other block.
                        
                        # Wait, if cover_count[ts] == 2, and ts is in T_bi, then bi is one of the two.
                        # If ts is also in T_bj, then bj is the other. So unique to pair.
                        # If ts is in T_bi but not T_bj, then the other block is not bj. So NOT unique to pair.
                        
                        if ts in set_bj:
                            unique_tsubs.append(ts)
                        # If ts in set_bi but not set_bj, it's covered by bi and someone else. Not unique to pair.
                        # If ts in set_bj but not set_bi, it's covered by bj and someone else. Not unique to pair.
                        # If ts in neither, it's not covered by bi or bj.
                        
                if not unique_tsubs:
                    # We can remove both bi and bj
                    del current_blocks[i]
                    del current_blocks[j]
                    current_set.discard(bi)
                    current_set.discard(bj)
                    improved = True
                    break
                
                # If unique_tsubs is not empty, we need to find 2 new blocks C, D
                # such that C U D covers unique_tsubs.
                
                # Heuristic: Pick the t-subset in unique_tsubs with fewest candidate blocks.
                # Candidates must cover all unique_tsubs.
                # Actually, C and D together must cover all unique_tsubs.
                
                # Let's find candidate blocks for each unique ts.
                # This is expensive. Let's limit to a few random unique_tsubs to define a "core".
                
                # Simpler heuristic:
                # Pick one unique ts, say ts_star.
                # Any solution (C,D) must have at least one block covering ts_star.
                # Let candidates_ts_star = tsub_blocks[ts_star] - current_set.
                # For each C in candidates_ts_star:
                #    remaining_tsubs = unique_tsubs - T_C
                #    If remaining_tsubs is empty, D can be anything (but we need to add 2, so maybe D is dummy? No, we must add 2 blocks).
                #    If remaining_tsubs is not empty, we need D to cover remaining_tsubs.
                #    Candidates for D = intersection of tsub_blocks[x] for x in remaining_tsubs, minus current_set and {C}.
                
                ts_star = unique_tsubs[0]
                cand_c = [b for b in tsub_blocks[ts_star] if b not in current_set]
                rng.shuffle(cand_c)
                
                found_swap = False
                for C in cand_c[:10]: # Limit candidates
                    # Check if C covers some unique tsubs
                    set_C = set(block_tsubs[C])
                    remaining = [ts for ts in unique_tsubs if ts not in set_C]
                    
                    if not remaining:
                        # C covers all. We need another block D.
                        # We can pick any D? No, we must add 2 blocks to replace 2.
                        # If C covers all, we can add C and any other block?
                        # But we want to minimize size. Adding C and D where D is redundant?
                        # If we add C and D, and D is redundant, we can remove D later.
                        # So just pick a random D from all blocks?
                        # Better: pick D such that it doesn't break anything.
                        # Let's just pick a random block not in current_set.
                        # Or better, pick D that covers some of the *other* tsubs to help future steps?
                        # For now, just pick a random valid block.
                        D = random.choice([b for b in range(n_blocks) if b not in current_set and b != C])
                        # Perform swap
                        current_blocks[i] = C
                        current_blocks[j] = D
                        current_set.discard(bi)
                        current_set.discard(bj)
                        current_set.add(C)
                        current_set.add(D)
                        improved = True
                        found_swap = True
                        break
                    
                    # Find D that covers all 'remaining'
                    # Intersection of tsub_blocks for each ts in remaining
                    cand_d = set(tsub_blocks[remaining[0]])
                    for ts in remaining[1:]:
                        cand_d &= set(tsub_blocks[ts])
                        if not cand_d:
                            break
                    
                    cand_d -= current_set
                    cand_d.discard(C)
                    
                    if cand_d:
                        D = random.choice(list(cand_d))
                        current_blocks[i] = C
                        current_blocks[j] = D
                        current_set.discard(bi)
                        current_set.discard(bj)
                        current_set.add(C)
                        current_set.add(D)
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not improved:
                break # No more improvements in 1-opt or 2-opt
        
        # Final redundancy removal
        current_blocks = remove_redundant_blocks(current_blocks)
        return current_blocks

    # Strategy: Multiple Greedy + Local Search
    num_trials = 20
    for seed in range(num_trials):
        if time.time() > deadline - 30:
            break
            
        blocks_idx = greedy_cover_heap(seed)
        if not blocks_idx:
            continue
            
        # Apply local search
        blocks_idx = local_search_2opt(blocks_idx, seed, max_rounds=10)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_indices = blocks_idx
            best_blocks = [all_blocks[i] for i in blocks_idx]
            
            if best_len <= 140: # Threshold for early exit
                break

    # Try MILP with warm start if we have time
    if best_indices and time.time() < deadline - 10:
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
                options={'time_limit': int(deadline - time.time() - 5), 'mip_gap': 0.0}
            )
            
            if result.status in (0, 1):
                x = result.x
                selected_idx = [i for i in range(n_blocks) if x[i] > 0.5]
                if selected_idx and len(selected_idx) < best_len:
                    best_len = len(selected_idx)
                    best_indices = selected_idx
                    best_blocks = [all_blocks[i] for i in selected_idx]
        except Exception:
            pass

    if best_blocks is None:
        # Fallback
        blocks_idx = greedy_cover_heap(0)
        blocks_idx = local_search_2opt(blocks_idx, 0, max_swap_rounds=5)
        best_blocks = [all_blocks[i] for i in blocks_idx]
        
    return best_blocks
