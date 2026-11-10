import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_t_subsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of t-subset indices that contain x
    # This is useful for quick updates, but for bitmask we need the full structure.
    
    # Bitmask representation for speed
    # Since v=19, we can use an integer bitmask for the set of points in a block.
    # However, checking if a t-subset is covered by a block is faster with sets or precomputed lists.
    # Let's stick to the efficient approach from inspiration: precompute t-subsets for each potential block?
    # No, C(19,8) is too big.
    # Instead, precompute for each block (if we had them) the t-subsets they cover.
    # But we generate blocks on the fly.
    
    # Key optimization from inspiration: 
    # Use a "coverage count" array and a bitmask-like structure or just fast set operations.
    # The current program uses sets of tuples. This is okay but can be slow.
    # Let's use the approach: 
    # 1. Greedy construction with randomized tie-breaking and "rarity" heuristic.
    # 2. Local search to remove redundant blocks and swap blocks to reduce count.
    
    # Precompute: for each point x, which t-subsets contain it?
    # point_tsub_indices[x] = list of indices of t-subsets containing x
    point_tsub_indices = [[] for _ in range(v)]
    for ts in all_t_subsets:
        for x in ts:
            point_tsub_indices[x].append(tsub_to_idx[ts])
    
    # For faster gain calculation in greedy:
    # We want to know, for a candidate block B, how many *uncovered* t-subsets it covers.
    # gain(B) = | { ts in B : ts is uncovered } |
    # We can maintain a boolean array `uncovered` of size n_tsubs.
    # To compute gain, we iterate over all t-subsets of B (C(k,t) of them) and check `uncovered[idx]`.
    # C(8,5) = 56. This is fast.
    
    def compute_gain(block, uncovered):
        """Compute how many uncovered t-subsets are in block."""
        gain = 0
        for ts in itertools.combinations(block, t):
            if uncovered[tsub_to_idx[ts]]:
                gain += 1
        return gain

    def greedy_once(seed, deadline):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                return None
            
            # Pick a t-subset to cover. 
            # Heuristic: Pick one that is "hard" to cover? 
            # Or just random. Random is simple and works well with restarts.
            # Let's try picking the one with the fewest covering blocks? 
            # That's expensive to compute. 
            # Let's stick to random selection of an uncovered t-subset.
            uncovered_indices = np.where(uncovered)[0]
            start_idx = rng.choice(uncovered_indices)
            start_ts = all_t_subsets[start_idx]
            
            # Find the best block containing start_ts
            # start_ts has t elements. We need to add k-t elements from the remaining v-t points.
            s_set = set(start_ts)
            remaining_points = [p for p in range(v) if p not in s_set]
            
            best_block = None
            best_gain = -1
            
            # Generate all combinations of k-t points from remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(start_ts + combo))
                
                # Calculate gain
                gain = compute_gain(block, uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    # Early break if we cover all possible? Max gain is C(k,t) but limited by uncovered.
                    # If best_gain is high, we might break early? No, we need max.
            
            if best_block is None:
                # Should not happen
                break
                
            blocks.append(list(best_block))
            
            # Update uncovered
            for ts in itertools.combinations(best_block, t):
                uncovered[tsub_to_idx[ts]] = False
            
        return blocks

    def local_search(blocks, deadline):
        """
        Try to remove redundant blocks and swap blocks to reduce count.
        """
        # Build coverage counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            subs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                cover_count[idx] += 1
                subs.append(idx)
            block_tsubs.append(subs)
        
        improved = True
        current_blocks = [list(b) for b in blocks]
        
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Remove redundant blocks
            # A block is redundant if all its t-subsets are covered by other blocks (count > 1)
            new_blocks = []
            for i, block in enumerate(current_blocks):
                redundant = True
                for idx in block_tsubs[i]:
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(block)
                else:
                    # Decrement cover counts for this block's t-subsets
                    for idx in block_tsubs[i]:
                        cover_count[idx] -= 1
                    improved = True
            
            if len(new_blocks) < len(current_blocks):
                current_blocks = new_blocks
                # Rebuild block_tsubs and cover_count? 
                # It's safer to rebuild from scratch after removal to avoid bugs.
                # But that's expensive. Let's just rebuild if we removed any.
                # Actually, the cover_count is already updated. We just need to update block_tsubs.
                # But the indices in block_tsubs correspond to the old list.
                # Let's just rebuild everything to be safe.
                cover_count = np.zeros(n_tsubs, dtype=np.int32)
                block_tsubs = []
                for block in current_blocks:
                    subs = []
                    for ts in itertools.combinations(block, t):
                        idx = tsub_to_idx[ts]
                        cover_count[idx] += 1
                        subs.append(idx)
                    block_tsubs.append(subs)
                continue # Restart loop to check for more redundancies
            
            # 2. Try to swap blocks? 
            # This is complex. Let's stick to removal for now.
            # If no removals, try a simple random restart within local search?
            # Or just stop.
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run greedy restarts
    # How many? 
    # Greedy takes ~1-2s. Local search takes ~0.5s.
    # Total ~2.5s per iteration.
    # We have 145s. So ~50 iterations.
    
    num_restarts = 0
    while time.time() < time_limit - 5.0:
        seed = num_restarts
        # Greedy
        blocks = greedy_once(seed, time_limit - 10.0)
        if blocks is None:
            continue
        
        # Local Search
        if time.time() < time_limit - 5.0:
            blocks = local_search(blocks, time_limit - 5.0)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        num_restarts += 1
        
        # If we hit the lower bound, stop
        if best_len <= 264:
            break

    if best_blocks is None:
        # Fallback
        blocks = greedy_once(0, time_limit - 1.0)
        if blocks:
            best_blocks = blocks
        else:
            # Emergency fallback: return empty list? No, must be valid.
            # But if greedy fails, we are in trouble.
            # Greedy should always succeed.
            best_blocks = []

    return best_blocks
