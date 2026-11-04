import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute triplet IDs for all possible (t-1)-subset + point combinations
    # For each (t-1)-subset, store a dict mapping point -> triplet_id
    pair_to_triplet_ids = {}
    for p in range(v):
        for rest in itertools.combinations(range(v), t - 1):
            if p not in rest:
                triplet = tuple(sorted(rest + (p,)))
                tid = triplet_to_id[triplet]
                key = tuple(sorted(rest))
                if key not in pair_to_triplet_ids:
                    pair_to_triplet_ids[key] = {}
                pair_to_triplet_ids[key][p] = tid
    
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Greedy Construction with Incremental Block Building
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        # List of uncovered triplet IDs
        uncovered_ids = list(range(n_triplets))
        uncovered_set = set(uncovered_ids)
        blocks = []
        
        while uncovered_ids:
            if time.time() > deadline - 15:
                break
                
            # Pick a random uncovered triplet to seed the block
            # Using index to allow O(1) removal later
            idx = rng.randrange(len(uncovered_ids))
            start_tid = uncovered_ids[idx]
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered_set:
                            gain += 1
                            
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break
                            
                if max_gain == -1:
                    best_point = rng.choice(candidates)
                elif max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            to_remove = []
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if tid in uncovered_set:
                    to_remove.append(tid)
            
            for tid in to_remove:
                uncovered_set.discard(tid)
                # Find index and remove from list (swapping with last for O(1))
                # This is O(N) worst case per removal, but total removals is N_triplets
                # We can just mark as removed and filter later, or use a dict?
                # Simpler: just rebuild uncovered_ids if it gets too sparse?
                # Or just use the set for membership, and list for random choice?
                # If we use list for random choice, we need to remove.
                # Let's just use the set for "is it uncovered" and a list for "pick one".
                # To pick one: 
                # 1. If len(uncovered_ids) is small, pick random.
                # 2. Remove from list.
                
                # Efficient removal from list:
                # Find index
                # Actually, doing linear search for removal is slow.
                # Let's just use the set `uncovered_set` for truth.
                # And for picking a random one, we can just pick a random index from `uncovered_ids` 
                # and check if it's in `uncovered_set`. If not, pick another?
                # Or better: maintain a list of "active" IDs.
                # Since N_triplets is 2024, linear search is actually fine (2024 ops).
                
                # Let's stick to simple: 
                # Remove from list
                if tid in uncovered_ids:
                    # Find and remove
                    # This is O(N)
                    uncovered_ids = [x for x in uncovered_ids if x != tid]
                    # Wait, this creates a new list every time. Expensive.
                    
            # Better approach for uncovered tracking:
            # Use a boolean array `is_uncovered`
            # And a list `uncovered_list`
            # When covering, set `is_uncovered[tid]=False`
            # When picking, pick random index from `uncovered_list`, 
            # while not `is_uncovered[id]`, pick again? 
            # Or just filter `uncovered_list` periodically.
            
            # Let's restart the uncovered tracking logic in the function for clarity and speed.
            
        return blocks

    # Refactored Greedy with Boolean Array
    def greedy_construct_fast(seed=0, ratio_mode=False, random_tiebreak_prob=0.0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        blocks = []
        
        # Index for O(1) swap-remove in uncovered_list
        # pos[i] = index of i in uncovered_list
        pos = np.arange(n_triplets)
        
        while uncovered_list:
            if time.time() > deadline - 20:
                break
                
            # Pick random uncovered
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_score = -1.0
                candidates = [p for p in range(v) if p not in block_set]
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                num_pairs = len(pairs_in_block)
                
                # Use precomputed pair_to_triplet_ids for speed
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        key = tuple(sorted(pair))
                        if key in pair_to_triplet_ids:
                            tid = pair_to_triplet_ids[key].get(p)
                            if tid is not None and is_uncovered[tid]:
                                gain += 1
                            
                    if ratio_mode:
                        score = gain / (1.0 + len(block_set) * 0.1)
                    else:
                        score = float(gain)
                        
                    if score > best_score:
                        best_score = score
                        best_point = p
                        if gain == num_pairs:
                            break
                            
                # Randomized tie-breaking: with some probability, pick a random candidate
                # among those with score close to best to explore diverse solutions
                if random_tiebreak_prob > 0 and rng.random() < random_tiebreak_prob:
                    # Find candidates with score >= best_score * 0.8
                    good_candidates = []
                    for p in candidates:
                        gain = 0
                        for pair in pairs_in_block:
                            key = tuple(sorted(pair))
                            if key in pair_to_triplet_ids:
                                tid = pair_to_triplet_ids[key].get(p)
                                if tid is not None and is_uncovered[tid]:
                                    gain += 1
                        if ratio_mode:
                            score = gain / (1.0 + len(block_set) * 0.1)
                        else:
                            score = float(gain)
                        if score >= best_score * 0.8:
                            good_candidates.append(p)
                    if good_candidates:
                        best_point = rng.choice(good_candidates)
                elif best_point == -1:
                    best_point = rng.choice(candidates)
                elif best_score == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()
                    pos[tid] = -1

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds with varied strategies
    num_seeds = 180
    for seed in range(num_seeds):
        if time.time() > deadline - 35:
            break
        # Cycle through different strategy combinations
        strategy = seed % 6
        if strategy < 2:
            ratio_mode = False
            random_tiebreak_prob = 0.0
        elif strategy < 4:
            ratio_mode = True
            random_tiebreak_prob = 0.0
        else:
            ratio_mode = False
            random_tiebreak_prob = 0.3  # Use randomized tie-breaking to explore diverse solutions
        
        blocks = greedy_construct_fast(seed, ratio_mode=ratio_mode, random_tiebreak_prob=random_tiebreak_prob)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct_fast(0)

    current_blocks = list(best_blocks)
    
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                count[get_id(triplet)] += delta

    update_counts(current_blocks, 1)

    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_merge(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute unique triplets for each block to avoid recomputation
        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            if not block_unique[i]:
                continue

            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    return blocks, False
                if not block_unique[j]:
                    continue
                
                all_unique_ids = set(block_unique[i]) | set(block_unique[j])
                
                points_needed = set()
                for tid in all_unique_ids:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) <= k:
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    update_counts([blocks[i], blocks[j]], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    
                    return new_blocks, True

        return blocks, False

    def try_swap_2_for_1(blocks):
        """Replace two blocks with one block if the union of their unique triplets
        fits in a single k-subset."""
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute unique triplets per block
        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            if not block_unique[i]:
                continue
            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    return blocks, False
                if not block_unique[j]:
                    continue

                all_unique = set(block_unique[i]) | set(block_unique[j])
                points_needed = set()
                for tid in all_unique:
                    points_needed.update(all_triplets[tid])

                if len(points_needed) <= k:
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([blocks[i], blocks[j]], -1)
                    update_counts([new_block], 1)

                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    progress_made = True
    while progress_made and time.time() < deadline - 5:
        progress_made = False
        
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True

        if not progress_made:
            current_blocks, merged = try_swap_2_for_1(current_blocks)
            if merged:
                progress_made = True

    return current_blocks

if __name__ == "__main__":
    pass
