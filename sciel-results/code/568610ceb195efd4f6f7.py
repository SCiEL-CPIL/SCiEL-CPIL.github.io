import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    def get_tid(triplet):
        return triplet_to_id[tuple(sorted(triplet))]
    
    # Precompute for each point, the pairs of other points that form a triplet with it
    # pairs_for_point[p] = list of (a, b, triplet_id) such that {p, a, b} is a triplet
    pairs_for_point = [[] for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for i in range(t):
            p = triplet[i]
            others = [x for j, x in enumerate(triplet) if j != i]
            # Since t=3, others has size 2
            a, b = others[0], others[1]
            pairs_for_point[p].append((a, b, tid))

    # Precompute triplet IDs for a block (faster than itertools each time)
    def block_triplet_ids(block):
        ids = []
        for i in range(len(block)):
            for j in range(i+1, len(block)):
                for m in range(j+1, len(block)):
                    trip = (block[i], block[j], block[m])
                    ids.append(get_tid(trip))
        return ids

    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet to start
            start_tid = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for a, b, tid in pairs_for_point[p]:
                        if a in block_set and b in block_set:
                            if tid in uncovered:
                                gain += 1
                    if gain > 0:
                        gains[p] = gain
                
                if not gains:
                    # No point increases coverage, pick random
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Pick point with max gain. Tie-break randomly.
                    max_gain = max(gains.values())
                    best_candidates = [p for p, g in gains.items() if g == max_gain]
                    best_point = rng.choice(best_candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_tid(triplet)
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_construct(0)
    
    # 1.5. Bipartite matching refinement: re-select blocks to maximize distinct coverage
    def bipartite_refine(blocks, max_time=15.0):
        """Use bipartite matching to select a subset of blocks that maximizes distinct triplet coverage."""
        start = time.time()
        n_triplets = len(all_triplets)
        
        # Build incidence: for each block, which triplets it covers
        block_triplets = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(get_tid(triplet))
            block_triplets.append(ids)
        
        # Start with all blocks, try to remove redundant ones using matching
        # We want to find a minimum subset that covers all triplets
        # Use a greedy approach with matching: repeatedly find the block that covers the most uncovered triplets
        
        uncovered = set(range(n_triplets))
        selected = []
        remaining_indices = list(range(len(blocks)))
        
        while uncovered and remaining_indices:
            if time.time() - start > max_time:
                break
            
            # Find block that covers most uncovered triplets
            best_idx = None
            best_gain = 0
            for idx in remaining_indices:
                gain = len(block_triplets[idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0:
                    # Tie-break: prefer blocks that share more with already-selected
                    pass
            
            if best_idx is None or best_gain == 0:
                break
            
            selected.append(best_idx)
            uncovered -= block_triplets[best_idx]
            remaining_indices.remove(best_idx)
        
        if not uncovered and len(selected) < len(blocks):
            return [blocks[i] for i in selected]
        return blocks
    
    refined = bipartite_refine(best_blocks, max_time=15.0)
    if len(refined) < len(best_blocks):
        best_blocks = refined

    # 2. Local Search Optimization
    # Maintain count of how many blocks cover each triplet
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_tid(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_tid(triplet)] == 1:
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

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # Precompute triplet ID sets for each block
        block_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(get_tid(triplet))
            block_ids.append(ids)
        
        # Try all pairs
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_ids[i]
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = block_ids[j]
                
                # Find critical triplets: those covered ONLY by b1 and/or b2
                critical_points = set()
                
                # Triplets in b1
                for tid in ids_b1:
                    c = count[tid]
                    cov_b1 = 1
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        critical_points.update(all_triplets[tid])
                        
                # Triplets in b2 (not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    cov_b2 = 1
                    if c == cov_b2:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both redundant
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # Can form one block from critical points
                    new_block = list(critical_points)
                    # Fill greedily
                    point_freq = np.zeros(v, dtype=np.int32)
                    for idx, b in enumerate(blocks):
                        if idx == i or idx == j:
                            continue
                        for p in b:
                            point_freq[p] += 1
                            
                    candidates = [p for p in range(v) if p not in new_block]
                    candidates.sort(key=lambda p: point_freq[p], reverse=True)
                    
                    for p in candidates:
                        if len(new_block) >= k:
                            break
                        new_block.append(p)
                    
                    new_block.sort()
                    
                    # Verify validity: new_block must cover all critical triplets
                    # Since critical_points are in new_block, and critical triplets are subsets of critical_points,
                    # they are covered.
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks, rng):
        """Try replacing one block with another that covers its critical triplets."""
        n = len(blocks)
        if n < 2:
            return blocks, False
        
        # Precompute triplet ID sets
        block_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(get_tid(triplet))
            block_ids.append(ids)
        
        # Try random pairs of blocks to swap
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                return blocks, False
            
            b1 = blocks[i]
            ids_b1 = block_ids[i]
            
            # Find critical triplets for b1 (covered only by b1)
            critical_tids = set()
            for tid in ids_b1:
                if count[tid] == 1:
                    critical_tids.add(tid)
            
            if not critical_tids:
                continue  # b1 is redundant, should have been removed
            
            # Find critical points
            critical_points = set()
            for tid in critical_tids:
                critical_points.update(all_triplets[tid])
            
            if len(critical_points) > k:
                continue
            
            # Try to find a replacement block that covers all critical triplets
            for j in indices:
                if j == i:
                    continue
                if time.time() > deadline - 5:
                    return blocks, False
                
                b2 = blocks[j]
                ids_b2 = block_ids[j]
                
                # Check if b2 covers all critical triplets
                covers_all = all(tid in ids_b2 for tid in critical_tids)
                
                if covers_all:
                    # Check if removing b1 and adding b2 doesn't break coverage
                    # b2's non-critical triplets must be covered by other blocks
                    new_block = list(critical_points)
                    # Fill with points from b2 not in critical_points
                    b2_set = set(b2)
                    for p in b2:
                        if p not in new_block:
                            new_block.append(p)
                            if len(new_block) >= k:
                                break
                    
                    # If we still need more points, add random
                    if len(new_block) < k:
                        remaining = [p for p in range(v) if p not in new_block]
                        rng.shuffle(remaining)
                        for p in remaining:
                            new_block.append(p)
                            if len(new_block) >= k:
                                break
                    
                    new_block.sort()
                    
                    # Verify: new block must cover all critical triplets
                    new_ids = set()
                    for triplet in itertools.combinations(new_block, t):
                        new_ids.add(triplet_to_id[triplet])
                    
                    if all(tid in new_ids for tid in critical_tids):
                        # Check that removing b1 doesn't leave any triplet uncovered
                        # (critical_tids are covered by new_block, so we're fine)
                        update_counts([b1], -1)
                        update_counts([new_block], 1)
                        
                        new_blocks = [b for idx, b in enumerate(blocks) if idx != i]
                        new_blocks.append(new_block)
                        return new_blocks, True
        
        return blocks, False

    def simulated_annealing(blocks, rng, max_steps=1000, start_temp=1.0, end_temp=0.01):
        """Use simulated annealing to escape local optima."""
        n = len(blocks)
        if n < 2:
            return blocks
        
        block_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(get_tid(triplet))
            block_ids.append(ids)
        
        current = list(blocks)
        current_ids = [set(ids) for ids in block_ids]
        current_cost = len(current)
        best = list(current)
        best_cost = current_cost
        best_ids = [set(ids) for ids in current_ids]
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (step / max_steps)
            
            # Random move: remove a block, then try to re-add it or a better one
            i = rng.randint(0, len(current) - 1)
            
            # Try removing block i
            old_block = current[i]
            old_ids = current_ids[i]
            
            # Check if block i is redundant
            is_red = all(count[tid] > 1 for tid in old_ids)
            
            if is_red:
                # Remove it
                update_counts([old_block], -1)
                current.pop(i)
                current_ids.pop(i)
                new_cost = len(current)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = list(current)
                    best_ids = [set(ids) for ids in current_ids]
            else:
                # Block is critical; try swapping with another block
                j = rng.randint(0, len(current) - 1)
                if j == i:
                    continue
                
                new_block = list(current[j])
                rng.shuffle(new_block)
                # Randomly modify the block
                if rng.random() < 0.5:
                    # Swap a point
                    p_in = new_block[0]
                    p_out = rng.randint(0, v - 1)
                    if p_out not in new_block:
                        new_block[0] = p_out
                        new_block.sort()
                
                # Check if new_block covers old_block's critical triplets
                new_ids = set()
                for triplet in itertools.combinations(new_block, t):
                    new_ids.add(get_tid(triplet))
                
                critical_tids = {tid for tid in old_ids if count[tid] == 1}
                covers = all(tid in new_ids for tid in critical_tids)
                
                if covers:
                    # Check if removing old_block and adding new_block is valid
                    # (new_block's triplets must be covered by other blocks or itself)
                    update_counts([old_block], -1)
                    # Tentatively add new_block
                    for tid in new_ids:
                        count[tid] += 1
                    
                    # Check validity
                    valid = all(count[tid] >= 1 for tid in new_ids)
                    
                    if valid:
                        new_cost = len(current)  # Same number of blocks
                        # Accept based on temperature (always accept same cost, reject worse)
                        if new_cost <= current_cost or rng.random() < np.exp(-(new_cost - current_cost) / max(temp, 1e-10)):
                            current[i] = new_block
                            current_ids[i] = new_ids
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best = list(current)
                                best_ids = [set(ids) for ids in current_ids]
                    else:
                        # Revert
                        for tid in new_ids:
                            count[tid] -= 1
                        update_counts([old_block], 1)
        
        # Restore best
        # First revert current to best
        for idx, b in enumerate(current):
            if idx < len(best):
                update_counts([b], -1)
        for idx, b in enumerate(best):
            if idx >= len(current):
                update_counts([b], 1)
            else:
                # Check if different
                if sorted(b) != sorted(current[idx]):
                    update_counts([b], 1)
        
        # Simpler: just return best and rebuild counts
        count.fill(0)
        update_counts(best, 1)
        return best

    # Main Optimization Loop
    progress_made = True
    max_iterations = 200
    stagnant_count = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 2:
            break
        progress_made = False
        
        # 1. Remove redundants
        while True:
            current_blocks, removed = try_remove_redundant(current_blocks)
            if not removed:
                break
            progress_made = True
        
        # 2. Try 2-for-1 swaps
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True
        
        # 3. Try 1-for-1 swaps
        if not progress_made:
            rng = random.Random(iteration * 1000 + 42)
            current_blocks, swapped = try_swap_1_for_1(current_blocks, rng)
            if swapped:
                progress_made = True
        
        # 4. Simulated Annealing if stuck
        if not progress_made and iteration % 5 == 4:
            if time.time() < deadline - 10:
                rng = random.Random(iteration * 7919 + 13)
                sa_blocks = simulated_annealing(current_blocks, rng, max_steps=200)
                if len(sa_blocks) < len(current_blocks):
                    current_blocks = sa_blocks
                    progress_made = True
                else:
                    # Rebuild counts
                    count.fill(0)
                    update_counts(current_blocks, 1)
        
        # 5. Periodic bipartite refinement
        if not progress_made and iteration % 10 == 9:
            if time.time() < deadline - 15:
                refined = bipartite_refine(current_blocks, max_time=10.0)
                if len(refined) < len(current_blocks):
                    current_blocks = refined
                    count.fill(0)
                    update_counts(current_blocks, 1)
                    progress_made = True
        
        if progress_made:
            stagnant_count = 0
        else:
            stagnant_count += 1
        
        if stagnant_count >= 15:
            break

    # Final pass to ensure no redundants remain
    current_blocks, _ = try_remove_redundant(current_blocks)
    
    return current_blocks
