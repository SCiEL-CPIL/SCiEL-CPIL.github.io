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
    
    # Precompute triplet ID lookup for fast access during greedy construction
    # For t=3, we can encode (a,b,c) with a<b<c as a*325 + b*12 + c for v=24
    # But let's use a dict for generality
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]
    
    # Precompute for each point p and each (t-1)-subset S, the triplet ID of S ∪ {p}
    # This is expensive for large v, so we'll compute on-the-fly but cache per (p, frozenset)
    triplet_id_cache = {}
    def get_triplet_id_cached(p, subset_tuple):
        """Get triplet ID for point p and (t-1)-subset subset_tuple"""
        key = (p, subset_tuple)
        if key not in triplet_id_cache:
            triplet = tuple(sorted(list(subset_tuple) + [p]))
            triplet_id_cache[key] = triplet_to_id[triplet]
        return triplet_id_cache[key]
    
    # Precompute for each point p and each (t-1)-subset, the list of triplet IDs
    # that contain both p and the (t-1)-subset
    # For efficiency: for each point p, precompute which triplets contain p
    point_triplets = [set() for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_triplets[p].add(tid)

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

    # Refactored Greedy with Boolean Array and smooth scoring
    def greedy_construct_fast(seed=0, mode='abs'):
        """
        mode: 'abs' - absolute gain, 'smooth' - smooth log-sum-exp based scoring,
              'diversity' - penalize points that overlap heavily with existing blocks
        """
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_count = n_triplets
        blocks = []
        block_point_sets = []  # list of sets for each block
        
        # For smooth scoring, track coverage counts for diversity penalty
        coverage_counts = np.zeros(n_triplets, dtype=np.int32)
        
        # Precompute for each point: which (t-1) subsets it forms triplets with
        # Actually, let's precompute: for point p and subset S of size (t-1),
        # the triplet ID of S ∪ {p}
        # This is expensive to precompute fully, so compute on fly but cache
        
        while uncovered_count > 0:
            if time.time() > deadline - 20:
                break
                
            # Pick random uncovered
            # Find a random uncovered triplet
            attempts = 0
            while attempts < 100:
                start_tid = rng.randrange(n_triplets)
                if is_uncovered[start_tid]:
                    break
                attempts += 1
            else:
                # Fallback: find first uncovered
                for tid in range(n_triplets):
                    if is_uncovered[tid]:
                        start_tid = tid
                        break
            
            start_triplet = all_triplets[start_tid]
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_score = -1.0
                candidates = [p for p in range(v) if p not in block_set]
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                num_pairs = len(pairs_in_block)
                
                # Compute gains for all candidates
                gains = {}
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        tid = get_triplet_id_cached(p, pair)
                        if is_uncovered[tid]:
                            gain += 1
                    gains[p] = gain
                
                if mode == 'smooth':
                    # Use log-sum-exp style smooth max to break ties
                    # Score = gain + small_bonus_for_diversity
                    # Diversity: prefer points that are less "central" (cover fewer already-covered triplets)
                    for p in candidates:
                        gain = gains[p]
                        # Count how many triplets this point would add that are already covered
                        redundant = 0
                        for pair in pairs_in_block:
                            tid = get_triplet_id_cached(p, pair)
                            if not is_uncovered[tid]:
                                redundant += 1
                        # Smooth score: emphasize gain, penalize redundancy softly
                        score = gain * 10.0 - redundant * 0.5 + rng.random() * 0.1
                        if score > best_score:
                            best_score = score
                            best_point = p
                elif mode == 'diversity':
                    # Penalize points that overlap with existing blocks
                    for p in candidates:
                        gain = gains[p]
                        # Count overlap with existing blocks
                        overlap = 0
                        for existing_block in block_point_sets:
                            if p in existing_block:
                                overlap += 1
                        score = gain * 5.0 - overlap * 1.0 + rng.random() * 0.5
                        if score > best_score:
                            best_score = score
                            best_point = p
                else:  # 'abs'
                    for p in candidates:
                        gain = gains[p]
                        score = float(gain)
                        if score > best_score:
                            best_score = score
                            best_point = p
                            if gain == num_pairs:
                                break
                
                if best_point == -1:
                    best_point = rng.choice(candidates)
                elif best_score <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            block_point_sets.append(set(block))
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    uncovered_count -= 1
                coverage_counts[tid] += 1

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds with different modes
    # Cycle through modes: abs, smooth, diversity
    modes = ['abs', 'smooth', 'diversity']
    # Use more seeds in the earlier phase, then refine
    num_seeds = 180
    for seed in range(num_seeds):
        if time.time() > deadline - 45:
            break
        mode = modes[seed % 3]
        blocks = greedy_construct_fast(seed, mode=mode)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct_fast(0, mode='abs')

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

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = set()
            for triplet in itertools.combinations(b1, t):
                tid = get_id(triplet)
                ids_b1.add(tid)
                
            unique_b1 = [tid for tid in ids_b1 if count[tid] == 1]
            
            if not unique_b1:
                continue

            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = set()
                for triplet in itertools.combinations(b2, t):
                    tid = get_id(triplet)
                    ids_b2.add(tid)
                    
                unique_b2 = [tid for tid in ids_b2 if count[tid] == 1]
                
                all_unique_ids = set(unique_b1) | set(unique_b2)
                
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
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks.append(b)
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

    def try_replace_1_for_1(blocks):
        """Try to replace a single block with another block that covers all its
        unique triplets and potentially more. This helps escape local minima."""
        n = len(blocks)
        if n < 1:
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
            unique_b = block_unique[i]
            if not unique_b:
                continue

            # The new block must cover all unique triplets of block i
            points_needed = set()
            for tid in unique_b:
                points_needed.update(all_triplets[tid])

            if len(points_needed) > k:
                continue

            # Try to find a better block: one that covers unique_b and also
            # covers some triplets that are currently covered by other blocks
            # Strategy: start with points_needed, then greedily add points that
            # maximize additional coverage of triplets with count >= 2

            current_block = list(blocks[i])
            current_block_set = set(current_block)
            
            # Compute what triplets the current block covers
            current_triplet_ids = set()
            for triplet in itertools.combinations(current_block, t):
                tid = get_id(triplet)
                current_triplet_ids.add(tid)

            # Build new block starting from points_needed
            new_block = list(points_needed)
            new_block_set = set(new_block)
            
            # Greedily add points to maximize coverage of triplets with count >= 2
            while len(new_block) < k:
                candidates = [p for p in range(v) if p not in new_block_set]
                if not candidates:
                    break
                
                best_point = -1
                best_gain = 0
                pairs_in_block = list(itertools.combinations(new_block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        tid = get_triplet_id_cached(p, pair)
                        # Count how many additional coverings this provides
                        # We want to cover triplets that are currently covered by this block
                        # but might not be covered by the new block otherwise
                        if tid in current_triplet_ids:
                            if count[tid] >= 2:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    # Add a random point
                    best_point = random.choice(candidates)
                
                new_block.append(best_point)
                new_block_set.add(best_point)
            
            new_block.sort()
            
            # Verify: the new block must cover all unique triplets of block i
            new_triplet_ids = set()
            for triplet in itertools.combinations(new_block, t):
                tid = get_id(triplet)
                new_triplet_ids.add(tid)
            
            if not set(unique_b).issubset(new_triplet_ids):
                continue

            # Check if this replacement is beneficial: it should cover at least
            # as many triplets as the old block, ideally more
            old_covered = len(current_triplet_ids)
            new_covered = len(new_triplet_ids)
            
            # Only replace if it's strictly better or equal (to avoid cycling)
            if new_covered > old_covered:
                update_counts([blocks[i]], -1)
                update_counts([new_block], 1)
                new_blocks = blocks[:i] + [new_block] + blocks[i+1:]
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

        if not progress_made:
            current_blocks, replaced = try_replace_1_for_1(current_blocks)
            if replaced:
                progress_made = True

    # Final validation: ensure all triplets are covered
    # If not, we need to add blocks to cover missing triplets
    final_count = np.zeros(n_triplets, dtype=np.int32)
    for block in current_blocks:
        for triplet in itertools.combinations(block, t):
            final_count[get_id(triplet)] += 1
    
    # Find uncovered triplets
    uncovered_tids = [tid for tid in range(n_triplets) if final_count[tid] == 0]
    
    if uncovered_tids:
        # Add blocks to cover uncovered triplets
        remaining_blocks = list(current_blocks)
        is_uncov = np.ones(n_triplets, dtype=bool)
        is_uncov[:n_triplets] = (final_count == 0)
        uncov_count = len(uncovered_tids)
        
        while uncov_count > 0 and time.time() < deadline - 2:
            # Pick a random uncovered triplet
            start_tid = random.choice(uncovered_tids)
            start_triplet = all_triplets[start_tid]
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        tid = get_triplet_id_cached(p, pair)
                        if is_uncov[tid]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1:
                    best_point = random.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            remaining_blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncov[tid]:
                    is_uncov[tid] = False
                    uncov_count -= 1
        
        current_blocks = remaining_blocks

    return current_blocks

if __name__ == "__main__":
    pass
