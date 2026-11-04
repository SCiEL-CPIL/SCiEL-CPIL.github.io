import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute point pairs for faster triplet ID lookup?
    # For t=3, triplet is (a, b, c). ID can be mapped.
    # Using a dictionary is O(1) average, which is fine for v=24.

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    def greedy_construct(v, k, t, seed=0, bias=0.0):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet to start the block
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                # Pairs in current block (for t=3)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Mix greedy and random based on bias
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass # keep best_point
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple Greedy Constructions
    # Use a mix of pure greedy and randomized greedy
    for seed in range(30):
        if time.time() > deadline - 60:
            break
        bias = 0.0 if seed % 3 == 0 else 0.2 + 0.05 * (seed % 10)
        blocks = greedy_construct(v, k, t, seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)
        best_count = len(best_blocks)

    # Phase 2: Local Search Optimization
    current_blocks = list(best_blocks)
    
    # Precompute triplet IDs for each block to speed up local search
    def get_block_triplets(block):
        return [get_id(triplet) for triplet in itertools.combinations(block, t)]

    block_triplet_ids = [get_block_triplets(b) for b in current_blocks]
    
    # Count coverage
    count = np.zeros(n_triplets, dtype=np.int32)
    for ids in block_triplet_ids:
        for tid in ids:
            count[tid] += 1

    def update_counts(block_ids, delta):
        for tid in block_ids:
            count[tid] += delta

    def is_redundant(block_ids, block_index):
        # Check if all triplets in this block are covered by other blocks
        # i.e., count[tid] > 1 for all tid in block_ids
        for tid in block_ids:
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant():
        removed = False
        new_blocks = []
        new_ids = []
        for i, b in enumerate(current_blocks):
            if is_redundant(block_triplet_ids[i], i):
                update_counts(block_triplet_ids[i], -1)
                removed = True
            else:
                new_blocks.append(b)
                new_ids.append(block_triplet_ids[i])
        return new_blocks, new_ids, removed

    def try_swap_2_for_1():
        n = len(current_blocks)
        if n < 2:
            return current_blocks, block_triplet_ids, False
            
        # We iterate through all pairs
        # To speed up, we can precompute the "unique" triplets for each block?
        # But counts change.
        
        for i in range(n):
            if time.time() > deadline - 5:
                return current_blocks, block_triplet_ids, False
            ids_i = block_triplet_ids[i]
            b_i = current_blocks[i]
            
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return current_blocks, block_triplet_ids, False
                ids_j = block_triplet_ids[j]
                b_j = current_blocks[j]
                
                # Find critical triplets: those that would be uncovered if we removed i and j
                # A triplet is critical if count[tid] == (1 if tid in ids_i else 0) + (1 if tid in ids_j else 0)
                # And tid is in ids_i or ids_j.
                
                # Optimization: Use sets for intersection?
                # Let's just iterate through the union of ids_i and ids_j.
                # Since k=7, t=3, each block has 35 triplets. Union is at most 70.
                
                critical_tids = []
                # We need to be careful with double counting in the union iteration if we use sets, 
                # but here we just need to check each unique tid in the union.
                union_ids = set(ids_i) | set(ids_j)
                
                for tid in union_ids:
                    c_i = 1 if tid in ids_i else 0
                    c_j = 1 if tid in ids_j else 0
                    if count[tid] == c_i + c_j:
                        critical_tids.append(tid)
                
                if not critical_tids:
                    # Both blocks are redundant together
                    update_counts(ids_i, -1)
                    update_counts(ids_j, -1)
                    new_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                    new_ids = [ids for idx, ids in enumerate(block_triplet_ids) if idx != i and idx != j]
                    return new_blocks, new_ids, True
                
                # Check if critical_tids can be covered by a single block
                # The set of points involved in critical_tids must be <= k
                critical_points = set()
                for tid in critical_tids:
                    critical_points.update(all_triplets[tid])
                
                if len(critical_points) <= k:
                    # We can form a new block containing critical_points
                    new_block = list(critical_points)
                    # Fill with random points to maximize coverage?
                    # For simplicity and determinism in local search, we can just fill randomly 
                    # or greedily. Greedy is better.
                    
                    # Greedy fill:
                    # Current partial block: new_block
                    # We want to maximize the number of triplets covered by the new block 
                    # that are NOT already covered by other blocks (or just maximize total coverage).
                    # Actually, since we are replacing i and j, we just need to ensure 
                    # the critical ones are covered. The rest is bonus.
                    
                    # Let's just fill with random points for now, or points from the original blocks?
                    # Using points from i and j might help preserve structure.
                    candidate_points = set(current_blocks[i]) | set(current_blocks[j])
                    candidate_points.difference_update(new_block)
                    
                    rng = random.Random(i * 100 + j)
                    # Shuffle candidates
                    candidates = list(candidate_points)
                    rng.shuffle(candidates)
                    
                    for p in candidates:
                        if len(new_block) >= k:
                            break
                        new_block.append(p)
                    
                    # If still not full, add random points
                    if len(new_block) < k:
                        all_points = set(range(v))
                        all_points.difference_update(new_block)
                        remaining = list(all_points)
                        rng.shuffle(remaining)
                        for p in remaining:
                            if len(new_block) >= k:
                                break
                            new_block.append(p)
                    
                    new_block.sort()
                    new_ids = get_block_triplets(new_block)
                    
                    # Verify validity: 
                    # The new block must cover all critical_tids. It does by construction.
                    # Are there any other triplets that become uncovered?
                    # No, because we only removed i and j. Any triplet not in critical_tids 
                    # was covered by at least one other block, or by both i and j (count >= 3).
                    # Wait, if a triplet was covered by i and j and one other, count=3.
                    # Remove i, j -> count=1. Still covered.
                    # If covered by i and j only, count=2. It is in critical_tids.
                    # So yes, valid.
                    
                    update_counts(ids_i, -1)
                    update_counts(ids_j, -1)
                    update_counts(new_ids, 1)
                    
                    new_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                    new_ids_list = [ids for idx, ids in enumerate(block_triplet_ids) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    new_ids_list.append(new_ids)
                    
                    return new_blocks, new_ids_list, True

        return current_blocks, block_triplet_ids, False

    def try_swap_1_for_1():
        n = len(current_blocks)
        if n == 0:
            return current_blocks, block_triplet_ids, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return current_blocks, block_triplet_ids, False
            ids_i = block_triplet_ids[i]
            b_i = current_blocks[i]
            
            # Find unique triplets of i
            unique_tids = [tid for tid in ids_i if count[tid] == 1]
            
            if not unique_tids:
                continue # Redundant, handled by removal phase
            
            # Can we find a replacement block that covers unique_tids?
            # The points in unique_tids must be <= k
            unique_points = set()
            for tid in unique_tids:
                unique_points.update(all_triplets[tid])
            
            if len(unique_points) > k:
                continue # Cannot cover all unique triplets in one block
            
            # Try to construct a replacement block
            new_block = list(unique_points)
            
            # Fill the rest to maximize coverage of non-unique triplets?
            # Or just fill randomly?
            # Greedy fill is better.
            
            # Calculate current coverage of other triplets by b_i
            # We want to pick points that cover triplets that are currently covered by b_i 
            # but will be lost if we don't cover them in the new block?
            # No, if a triplet is covered by b_i and other blocks, it's fine.
            # We just need to cover unique_tids.
            # To be safe, let's pick points that maximize the number of triplets 
            # covered by the new block that are currently covered by b_i.
            
            # Simple heuristic: Pick points from b_i first, then others.
            candidates = [p for p in b_i if p not in new_block]
            rng = random.Random(i)
            rng.shuffle(candidates)
            
            for p in candidates:
                if len(new_block) >= k:
                    break
                new_block.append(p)
            
            if len(new_block) < k:
                all_points = set(range(v))
                all_points.difference_update(new_block)
                remaining = list(all_points)
                rng.shuffle(remaining)
                for p in remaining:
                    if len(new_block) >= k:
                        break
                    new_block.append(p)
            
            new_block.sort()
            new_ids = get_block_triplets(new_block)
            
            # Check if new_block covers all unique_tids
            new_id_set = set(new_ids)
            if not all(tid in new_id_set for tid in unique_tids):
                continue # Should not happen if unique_points <= k
            
            # Perform swap
            update_counts(ids_i, -1)
            update_counts(new_ids, 1)
            
            new_blocks = list(current_blocks)
            new_blocks[i] = new_block
            new_ids_list = list(block_triplet_ids)
            new_ids_list[i] = new_ids
            
            return new_blocks, new_ids_list, True

        return current_blocks, block_triplet_ids, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        # 1. Remove redundants
        current_blocks, block_triplet_ids, removed = try_remove_redundant()
        if removed:
            progress_made = True
            continue # Restart loop as structure changed significantly
        
        # 2. Swap 2 for 1
        current_blocks, block_triplet_ids, swapped = try_swap_2_for_1()
        if swapped:
            progress_made = True
            continue
        
        # 3. Swap 1 for 1
        current_blocks, block_triplet_ids, swapped = try_swap_1_for_1()
        if swapped:
            progress_made = True

    # Final cleanup
    current_blocks, block_triplet_ids, _ = try_remove_redundant()

    return current_blocks

if __name__ == "__main__":
    pass
