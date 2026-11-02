import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Precompute all t-subsets as bitmasks
    # v=19, t=5. C(19,5) = 11628.
    all_t_subs = []
    for sub in itertools.combinations(range(v), t):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        all_t_subs.append(mask)
    n_t_subs = len(all_t_subs)
    
    # Map from bitmask to index for fast lookup? 
    # Or just keep the list of masks.
    # For a block, we need to know which t-subsets it covers.
    # A block has C(8,5) = 56 t-subsets.
    
    # Helper to get t-subset masks of a block mask
    def get_block_t_masks(block_mask):
        # Get the set bits of block_mask
        points = []
        temp = block_mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            points.append(idx)
            temp ^= lsb
        
        masks = []
        for sub in itertools.combinations(points, t):
            mask = 0
            for p in sub:
                mask |= (1 << p)
            masks.append(mask)
        return masks

    # 1. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        # counts[i] is the number of blocks covering t-sub i
        counts = [0] * n_t_subs
        blocks = [] # list of block masks
        uncovered_count = n_t_subs
        
        # To speed up greedy, we can pick a random uncovered subset and build a block around it
        # But calculating gain for every candidate block is expensive.
        # Instead, pick a random uncovered subset S.
        # Generate random blocks containing S and pick the one with max gain.
        
        while uncovered_count > 0 and time.time() < deadline - 5:
            # Find an uncovered subset
            # This is slow if we scan all. 
            # Better: maintain a list of uncovered indices?
            # For simplicity and speed in greedy, let's just pick a random index and check if it's uncovered.
            # Or, since we start with 0, we know which are uncovered.
            # Let's maintain a list `uncovered_indices`.
            
            # Actually, for the first few blocks, we can just pick random points.
            # Let's use a simpler greedy:
            # Pick a random point, extend to k.
            # This is slow.
            
            # Let's use the "pick uncovered subset" strategy properly.
            # We need a way to quickly find an uncovered subset.
            # We can maintain a list of indices where counts[i] == 0.
            
            if not hasattr(greedy_construct, 'uncovered_list'):
                greedy_construct.uncovered_list = list(range(n_t_subs))
            else:
                # This logic is flawed in a closure. Let's use local variables.
                pass
            
        # Let's rewrite greedy to be self-contained and efficient.
        
        # Re-implementation of Greedy
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered and time.time() < deadline - 10:
            # Pick a random uncovered subset
            idx = random.choice(list(uncovered))
            target_mask = all_t_subs[idx]
            target_points = [i for i in range(v) if (target_mask >> i) & 1]
            
            # We must include these 't' points in the block
            block_points = set(target_points)
            
            # We need to add k-t more points
            # Candidates are the remaining v-t points
            candidates = [p for p in range(v) if p not in block_points]
            
            # Greedily select the best k-t points to maximize coverage of other uncovered subsets
            # This is a max-coverage problem on the remaining points.
            # Given small k-t (3), we can try all combinations? C(14,3) = 364.
            # For each combination, calculate gain.
            
            best_gain = -1
            best_combo = None
            
            # To speed up, we can sample a few combos or use greedy selection of points.
            # Greedy selection of points:
            # Add point that covers most uncovered subsets.
            
            # Let's try greedy point selection for speed.
            for _ in range(k - t):
                best_p = -1
                best_p_gain = -1
                for p in candidates:
                    # Calculate gain of adding p
                    # New subsets formed are those containing p and t-1 from current block_points
                    gain = 0
                    # Current block_points has size s. New subsets contain p and t-1 from block_points.
                    # Number of such subsets is C(s, t-1).
                    # We need to check if these subsets are uncovered.
                    # Since block_points changes, this is complex.
                    
                    # Simpler: Just pick random candidates and check?
                    # Or, precompute for each point p, which subsets it helps?
                    # Too complex.
                    
                    # Let's just pick a random point from candidates for now to ensure speed,
                    # or use a simple heuristic: pick point that appears in many uncovered subsets?
                    # We can precompute "frequency" of points in uncovered subsets.
                    pass
                # Fallback: Random
                best_p = random.choice(candidates)
                block_points.add(best_p)
                candidates.remove(best_p)
            
            block_mask = 0
            for p in block_points:
                block_mask |= (1 << p)
            
            blocks.append(block_mask)
            
            # Update counts
            block_t_masks = get_block_t_masks(block_mask)
            for mask in block_t_masks:
                # Find index of mask
                # This is slow. We need a map.
                pass

    # The above greedy is too slow/complex to write correctly in short time.
    # Let's stick to the previous program's structure but optimize the local search.
    # The previous program scored 1.14.
    # The key improvement is likely in the Local Search moves.
    
    # Let's use the previous program's greedy (which worked) but replace local search with a stronger one.
    
    # Re-creating the efficient data structures from the previous good run (implicit in score 1.14)
    
    # Precompute t-subsets indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    def get_block_t_sub_indices(block):
        return [t_sub_index[sub] for sub in itertools.combinations(block, t)]

    # Greedy (same as before, proven to work)
    def greedy_construct_simple(seed):
        random.seed(seed)
        counts = [0] * n_t_subs
        blocks = []
        uncovered = set(range(n_t_subs))
        
        while uncovered and time.time() < deadline - 20:
            idx = random.choice(list(uncovered))
            target_sub = all_t_subs[idx]
            block = list(target_sub)
            
            while len(block) < k:
                best_p = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (p,)))
                        sub_idx = t_sub_index.get(sub, -1)
                        if sub_idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                    elif gain == best_gain and best_p != -1:
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p == -1:
                    break
                block.append(best_p)
            
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                idx2 = t_sub_index[sub]
                counts[idx2] += 1
                if counts[idx2] == 1:
                    uncovered.discard(idx2)
                    
        return blocks

    # Strong Local Search
    def local_search_strong(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_sub_index[sub]] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        # Precompute for each block, the list of t-subset indices it covers
        block_t_indices = [get_block_t_sub_indices(b) for b in blocks]
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try removals
            for i in range(len(blocks)):
                b = blocks[i]
                indices = block_t_indices[i]
                removable = True
                for idx in indices:
                    if counts[idx] <= 1:
                        removable = False
                        break
                if removable:
                    for idx in indices:
                        counts[idx] -= 1
                    blocks.pop(i)
                    block_t_indices.pop(i)
                    improved = True
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = [list(x) for x in blocks]
                    break # Restart loop
            
            if not improved:
                # 2. Try 1-swaps
                # Pick a random block
                if not blocks:
                    break
                i = random.randint(0, len(blocks)-1)
                b = blocks[i]
                x = random.choice(b)
                
                # Find a y not in b
                y_candidates = [p for p in range(v) if p not in b]
                if not y_candidates:
                    continue
                y = random.choice(y_candidates)
                
                # Check if swap is valid (maintains coverage)
                # Subsets of b containing x are at risk
                # Subsets of new b containing y are gained
                # We need counts[idx] > 1 for all idx in b that contain x
                # AND we don't care about y's coverage for validity, as long as we don't lose x's unique coverage.
                
                valid = True
                # Calculate indices to decrement (those with x)
                dec_indices = []
                for sub in itertools.combinations(b, t):
                    if x in sub:
                        idx = t_sub_index[sub]
                        if counts[idx] <= 1:
                            valid = False
                            break
                        dec_indices.append(idx)
                
                if valid:
                    # Calculate indices to increment (those with y in new block)
                    new_b = [p for p in b if p != x]
                    new_b.append(y)
                    new_b.sort()
                    inc_indices = []
                    for sub in itertools.combinations(new_b, t):
                        if y in sub:
                            idx = t_sub_index[sub]
                            inc_indices.append(idx)
                    
                    # Perform swap
                    for idx in dec_indices:
                        counts[idx] -= 1
                    for idx in inc_indices:
                        counts[idx] += 1
                    
                    blocks[i] = new_b
                    block_t_indices[i] = inc_indices + [j for j in block_t_indices[i] if j not in set(dec_indices)] # This is inefficient. Recalculate.
                    block_t_indices[i] = get_block_t_sub_indices(new_b)
                    improved = True # Accept structure change to explore
            
            # 3. Try 2-swaps (if stuck)
            # This is expensive, so only try if no improvement in last X iterations?
            # For now, let's stick to 1-swap + removal.
            
        return best_blocks

    # Run greedy and then local search
    best_initial = None
    best_len = float('inf')
    
    # Run greedy multiple times
    for i in range(5):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct_simple(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_initial = blocks
            
    if best_initial is None:
        best_initial = greedy_construct_simple(0)
        
    # Local search
    final_blocks = local_search_strong(best_initial, deadline - 5)
    
    return final_blocks
