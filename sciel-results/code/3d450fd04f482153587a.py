import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
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
                    ids.append(triplet_to_id[trip])
        return ids
    
    # Precompute all candidate blocks (C(24,7) = 346104) and their triplet coverage
    # This is expensive but enables faster local search
    all_candidate_blocks = list(itertools.combinations(range(v), k))
    n_candidates = len(all_candidate_blocks)
    
    # For each candidate block, precompute the set of triplet IDs it covers
    # We'll use a list of frozensets for fast membership testing
    candidate_triplet_sets = []
    for block in all_candidate_blocks:
        ids = set()
        for triplet in itertools.combinations(block, t):
            ids.add(triplet_to_id[triplet])
        candidate_triplet_sets.append(ids)

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
                tid = triplet_to_id[triplet]
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # 2. Local Search Optimization
    # Maintain count of how many blocks cover each triplet
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[triplet_to_id[triplet]] == 1:
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
    
    # Precompute for each block in current_blocks, the set of triplet IDs it covers
    def get_block_ids(block):
        ids = set()
        for triplet in itertools.combinations(block, t):
            ids.add(triplet_to_id[triplet])
        return ids

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # Precompute triplet ID sets for each block
        block_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(triplet_to_id[triplet])
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
                ids.add(triplet_to_id[triplet])
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

    def hypergraph_local_search(blocks, rng, max_iterations=100):
        """
        Formulate as k-way hypergraph vertex cover.
        Nodes = triplets (C(24,3) = 2024), Hyperedges = candidate blocks (C(24,7) = 346104).
        We want to select minimum hyperedges to cover all nodes.
        
        Strategy: Start with current solution. Try to replace one selected block with 
        a different candidate block that covers all its critical triplets and has 
        better overlap properties.
        """
        n = len(blocks)
        if n < 2:
            return blocks
        
        current = list(blocks)
        best = list(current)
        best_cost = len(current)
        
        # Precompute triplet ID sets for current blocks
        block_ids = [get_block_ids(b) for b in current]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Pick a random block to try to replace
            i = rng.randint(0, len(current) - 1)
            old_block = current[i]
            old_ids = block_ids[i]
            
            # Find critical triplets (covered only by this block)
            critical_tids = {tid for tid in old_ids if count[tid] == 1}
            
            if not critical_tids:
                # Block is redundant, remove it
                update_counts([old_block], -1)
                current.pop(i)
                block_ids.pop(i)
                best = list(current)
                best_cost = len(current)
                continue
            
            # Find all candidate blocks that cover all critical triplets
            # This is expensive, so we sample
            n_samples = 50
            found_replacement = False
            
            for _ in range(n_samples):
                # Sample a random candidate block
                cand_idx = rng.randint(0, n_candidates - 1)
                cand_block = all_candidate_blocks[cand_idx]
                cand_ids = candidate_triplet_sets[cand_idx]
                
                # Check if it covers all critical triplets
                if not all(tid in cand_ids for tid in critical_tids):
                    continue
                
                # Check if this candidate is already in our solution
                if cand_block in current:
                    continue
                
                # Tentatively replace
                update_counts([old_block], -1)
                for tid in cand_ids:
                    count[tid] += 1
                
                # Check validity
                valid = all(count[tid] >= 1 for tid in cand_ids)
                
                if valid:
                    # Accept the replacement
                    current[i] = list(cand_block)
                    block_ids[i] = cand_ids
                    found_replacement = True
                    break
                else:
                    # Revert
                    for tid in cand_ids:
                        count[tid] -= 1
                    update_counts([old_block], 1)
            
            if not found_replacement:
                # Try a different strategy: remove two blocks and add one
                if len(current) >= 2:
                    j = rng.randint(0, len(current) - 1)
                    if j == i:
                        continue
                    
                    old_block2 = current[j]
                    old_ids2 = block_ids[j]
                    
                    # Find critical triplets for both blocks
                    crit1 = {tid for tid in old_ids if count[tid] == 1}
                    crit2 = {tid for tid in old_ids2 if count[tid] == 1}
                    
                    # If either is redundant, just remove it
                    if not crit1:
                        update_counts([old_block], -1)
                        current.pop(i)
                        block_ids.pop(i)
                        best = list(current)
                        best_cost = len(current)
                        continue
                    if not crit2:
                        update_counts([old_block2], -1)
                        current.pop(j)
                        block_ids.pop(j)
                        best = list(current)
                        best_cost = len(current)
                        continue
                    
                    # Find critical points for both
                    crit_points = set()
                    for tid in crit1 | crit2:
                        crit_points.update(all_triplets[tid])
                    
                    if len(crit_points) <= k:
                        # Try to find a single block that covers all critical triplets
                        for _ in range(30):
                            cand_idx = rng.randint(0, n_candidates - 1)
                            cand_block = all_candidate_blocks[cand_idx]
                            cand_ids = candidate_triplet_sets[cand_idx]
                            
                            if all(tid in cand_ids for tid in crit1 | crit2):
                                # Tentatively replace both with this one
                                update_counts([old_block, old_block2], -1)
                                for tid in cand_ids:
                                    count[tid] += 1
                                
                                valid = all(count[tid] >= 1 for tid in cand_ids)
                                
                                if valid:
                                    # Remove both, add one
                                    current.pop(j)  # Remove j first (higher index)
                                    block_ids.pop(j)
                                    current.pop(i)  # Now remove i
                                    block_ids.pop(i)
                                    current.append(list(cand_block))
                                    block_ids.append(cand_ids)
                                    best = list(current)
                                    best_cost = len(current)
                                    break
                                else:
                                    for tid in cand_ids:
                                        count[tid] -= 1
                                    update_counts([old_block, old_block2], 1)
        
        # Rebuild counts from best
        count.fill(0)
        update_counts(best, 1)
        return best

    # Main Optimization Loop
    progress_made = True
    max_iterations = 150
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 3:
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
        
        # 4. Hypergraph local search if stuck
        if not progress_made and iteration % 3 == 2:
            if time.time() < deadline - 10:
                rng = random.Random(iteration * 7919 + 13)
                hs_blocks = hypergraph_local_search(current_blocks, rng, max_iterations=50)
                if len(hs_blocks) < len(current_blocks):
                    current_blocks = hs_blocks
                    progress_made = True
                else:
                    # Rebuild counts
                    count.fill(0)
                    update_counts(current_blocks, 1)
        
        if not progress_made:
            break

    # Final pass to ensure no redundants remain
    current_blocks, _ = try_remove_redundant(current_blocks)
    
    return current_blocks
