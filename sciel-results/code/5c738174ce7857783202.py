import itertools
import time
import random
import heapq
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build incidence data
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
    
    # For local search, we need quick access to which blocks cover a t-subset
    # Store as lists for easy set operations if needed, but numpy is faster for counting
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Convert to numpy arrays for vectorized gain updates
    block_tsubs_arr = [np.array(ts_list, dtype=np.int32) for ts_list in block_tsubs]
    tsub_blocks_arr = [np.array(bi_list, dtype=np.int32) for bi_list in tsub_blocks]
    
    # Initial gains
    initial_gains = np.array([len(ts_list) for ts_list in block_tsubs], dtype=np.int32)

    best_blocks = None
    best_indices = None
    best_len = float('inf')

    def greedy_cover_heap(seed=0):
        """
        Greedy set cover using a priority queue (heap) for efficiency.
        Uses lazy deletion to handle stale gain values.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        gain = initial_gains.copy()
        blocks_used = []
        
        # Heap entries: (-gain, block_index, version)
        # We need a way to invalidate old entries.
        # Since gains only decrease, if we pop an item with gain G, but current gain is G' < G, we discard it.
        # We don't strictly need a version counter if we check gain[block_index] == -popped_gain.
        
        heap = [(-gain[i], i) for i in range(n_blocks) if gain[i] > 0]
        heapq.heapify(heap)
        
        while np.any(uncovered):
            if time.time() > deadline - 2:
                break
                
            if not heap:
                break
                
            neg_g, bi = heapq.heappop(heap)
            g = -neg_g
            
            # Lazy deletion check
            if gain[bi] != g:
                continue
            if g == 0:
                continue
            
            # Check if this block actually covers any *new* t-subsets
            # The gain calculation in the heap assumes 'gain' is the count of NEW tsubs covered.
            # The update logic below ensures gain[bi] is updated correctly.
            
            blocks_used.append(bi)
            
            # Update uncovered set and gains
            chosen_tsubs = block_tsubs_arr[bi]
            for ts_idx in chosen_tsubs:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # This t-subset is now covered. 
                    # Decrease gain for all blocks that covered it.
                    blocks_covering = tsub_blocks_arr[ts_idx]
                    # Vectorized decrement
                    # Note: gain is int32. 
                    gain[blocks_covering] -= 1
                    # Push updated gains for affected blocks if they are still active
                    # We don't push every single one to avoid heap bloat, 
                    # but we must ensure the heap has *some* valid entry for active blocks.
                    # Actually, a simpler strategy for heap greedy:
                    # We push the new gain for *every* block affected.
                    # This creates duplicates, but lazy deletion handles it.
                    for nb in blocks_covering:
                        if gain[nb] > 0 and gain[nb] != -1:
                            heapq.heappush(heap, (-gain[nb], nb))
            
            # Mark this block as used (negative gain indicates used/invalid)
            gain[bi] = -1
            
        return blocks_used

    def remove_redundant_blocks(block_indices):
        if not block_indices:
            return block_indices
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_indices:
            cover_count[block_tsubs_arr[bi]] += 1
            
        kept = [True] * len(block_indices)
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(block_indices):
                if not kept[i]:
                    continue
                redundant = True
                for ts_idx in block_tsubs_arr[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    cover_count[block_tsubs_arr[bi]] -= 1
                    changed = True
                    
        return [block_indices[i] for i in range(len(block_indices)) if kept[i]]

    def local_search_advanced(block_indices, seed=0, max_rounds=30):
        """
        Advanced local search:
        1. Remove redundant blocks.
        2. Try 2-for-1 swaps (replace 2 blocks with 1).
        3. Try 1-for-1 swaps (replace 1 block with 1).
        """
        if not block_indices:
            return block_indices
        
        current_blocks = list(block_indices)
        rng = random.Random(seed)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            # Phase 1: Redundancy Removal
            current_blocks = remove_redundant_blocks(current_blocks)
            
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for bi in current_blocks:
                cover_count[block_tsubs_arr[bi]] += 1
            
            current_set = set(current_blocks)
            improved = False
            
            # Phase 2: 2-for-1 Swap
            # Try to find two blocks i, j such that their union is covered by a single block x.
            # This is expensive O(N^2). We will sample pairs.
            n_curr = len(current_blocks)
            if n_curr >= 2:
                # Sample a few pairs
                num_samples = min(50, n_curr * 5)
                sampled_pairs = set()
                for _ in range(num_samples):
                    i, j = rng.sample(range(n_curr), 2)
                    if (i, j) in sampled_pairs or (j, i) in sampled_pairs:
                        continue
                    sampled_pairs.add((i, j))
                    
                    bi = current_blocks[i]
                    bj = current_blocks[j]
                    
                    # T-subsets uniquely covered by bi or bj
                    # A t-subset is 'exposed' by removing bi and bj if it was only covered by them.
                    # Actually, we need to cover (Unique_i \ Union_j) U (Unique_j \ Union_i) with one block.
                    # Simpler: Let U_bi = tsubs in bi. Let U_bj = tsubs in bj.
                    # Let S_i = tsubs in bi with count 1 (when only bi is considered? No, count in current set).
                    # Let C_i = {ts in bi | count[ts] == 1} -> These MUST be covered by the new block x.
                    # Let C_j = {ts in bj | count[ts] == 1} -> These MUST be covered by the new block x.
                    # Wait, if a t-subset has count 2 (covered by bi and bj), removing them leaves count 0.
                    # So x must cover ALL t-subsets in bi U bj that have count <= 2?
                    # No. If count[ts] == 2, and bi, bj are the ones, then removing them makes count 0. So x must cover ts.
                    # If count[ts] == 1, and it's in bi, removing bi makes it 0. So x must cover ts.
                    # So x must cover: (bi U bj) \ {ts | count[ts] > 2 OR (count[ts]==2 and ts not in bi U bj? Impossible)}
                    # Essentially, x must cover all ts in (bi U bj) such that count[ts] <= 2.
                    
                    union_tsubs = np.union1d(block_tsubs_arr[bi], block_tsubs_arr[bj])
                    required_tsubs = union_tsubs[cover_count[union_tsubs] <= 2]
                    
                    if len(required_tsubs) == 0:
                         # If no t-subsets are strictly required, we can replace with any block? 
                         # No, we need to maintain coverage. If we remove bi, bj, any ts in bi U bj with count 2 becomes 0.
                         # So required_tsubs is exactly those with count <= 2.
                         # If required_tsubs is empty, it means all ts in bi U bj have count >= 3.
                         # In that case, we can replace bi, bj with ANY block? No, we just need to pick a block.
                         # Actually, if required_tsubs is empty, we can just pick the first candidate or even a random block?
                         # But we want to minimize size. If we replace 2 with 1, size drops by 1.
                         # If required_tsubs is empty, any block works? No, we just need *a* block.
                         # But wait, if required_tsubs is empty, it implies bi and bj are "redundant" in a pair sense?
                         # We can just pick a random block from the available ones? 
                         # Better: pick a block that covers the most *other* uncovered things? 
                         # For now, if required_tsubs is empty, we can replace with a random block from all_blocks.
                         # But let's stick to finding a specific block that covers required_tsubs.
                         pass

                    if len(required_tsubs) > 0:
                        # Find candidates that cover all required_tsubs
                        # Start with rarest
                        freqs = [len(tsub_blocks[ts]) for ts in required_tsubs]
                        order = np.argsort(freqs)
                        required_sorted = required_tsubs[order]
                        
                        candidates = set(tsub_blocks[required_sorted[0]])
                        for ts_idx in required_sorted[1:]:
                            candidates &= set(tsub_blocks[ts_idx])
                            if not candidates:
                                break
                        
                        # Remove current blocks
                        candidates -= current_set
                        
                        if candidates:
                            # Pick one
                            new_bi = rng.choice(list(candidates))
                            current_blocks[i] = new_bi
                            # Remove j
                            del current_blocks[j]
                            current_set.discard(bi)
                            current_set.discard(bj)
                            current_set.add(new_bi)
                            improved = True
                            break
                    else:
                        # Replace with a random block? Or just drop one? 
                        # If we drop one, we might break coverage if count was exactly 2 for some ts.
                        # But we established count >= 3 for all ts in union.
                        # So we can just drop one block?
                        # If we drop one, size drops by 1.
                        # Let's drop j.
                        del current_blocks[j]
                        current_set.discard(bj)
                        improved = True
                        break
            
            if improved:
                # Restart round with updated state
                continue
            
            # Phase 3: 1-for-1 Swap
            indices_order = list(range(len(current_blocks)))
            rng.shuffle(indices_order)
            
            for i in indices_order:
                if time.time() > deadline - 5:
                    break
                
                bi = current_blocks[i]
                bi_tsubs = block_tsubs_arr[bi]
                unique_tsubs = bi_tsubs[cover_count[bi_tsubs] == 1]
                
                if len(unique_tsubs) == 0:
                    continue
                
                unique_tsubs_sorted = unique_tsubs[np.argsort([len(tsub_blocks[ts]) for ts in unique_tsubs])]
                
                candidates = set(tsub_blocks[unique_tsubs_sorted[0]])
                for ts_idx in unique_tsubs_sorted[1:]:
                    candidates &= set(tsub_blocks[ts_idx])
                    if not candidates:
                        break
                
                candidates -= current_set
                
                if candidates:
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
            
            if not improved:
                break
        
        return remove_redundant_blocks(current_blocks)

    # Main Loop
    num_trials = 40
    for seed in range(num_trials):
        if time.time() > deadline - 25:
            break
        
        blocks_idx = greedy_cover_heap(seed)
        if not blocks_idx:
            continue
        
        blocks_idx = local_search_advanced(blocks_idx, seed, max_rounds=25)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_indices = blocks_idx
            best_blocks = [all_blocks[i] for i in blocks_idx]
            
            if best_len <= 145 and time.time() < deadline - 20:
                polished = local_search_advanced(blocks_idx, seed + 1000, max_rounds=50)
                if len(polished) < best_len:
                    best_len = len(polished)
                    best_indices = polished
                    best_blocks = [all_blocks[i] for i in polished]
                    
        if best_len <= 142:
            break

    # MILP Polish
    if best_indices and time.time() < deadline - 20:
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
            remaining_time = int(deadline - time.time() - 8)
            if remaining_time > 5:
                result = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    x0=x0,
                    options={'time_limit': remaining_time, 'mip_gap': 0.01}
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
        blocks_idx = greedy_cover_heap(0)
        blocks_idx = local_search_advanced(blocks_idx, 0, max_rounds=10)
        best_blocks = [all_blocks[i] for i in blocks_idx]
        
    return best_blocks
