import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    start_time = time.time()
    # Hard deadline to ensure we return before 120s (evaluator kills at 150s, keep margin)
    deadline = start_time + 105.0

    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed (vectorized access)
    # idx_3d[a, b, c] is the index of the sorted triple (a,b,c)
    # We only care about distinct a,b,c.
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]
                
    def get_triple_idx(a, b, c):
        return idx_3d[a, b, c]

    def greedy_cover(random_state, weighted=False):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        # For weighted greedy, compute initial weights based on "rarity"
        # Initially all triples are equally rare, so weights are uniform.
        # But as we cover triples, the remaining uncovered triples become the focus.
        # The "weight" for a triple can be seen as 1 if uncovered, 0 if covered.
        # Standard greedy maximizes count.
        # Weighted greedy might prioritize triples that are "harder" to cover.
        # A common heuristic is to pick the triple with the fewest candidate blocks.
        # Here, we can approximate by picking the uncovered triple that appears
        # in the fewest potential blocks? That's expensive.
        # Alternative: Randomly pick an uncovered triple, but with bias towards
        # those that have been "neglected" or are central?
        # Let's stick to the standard "first uncovered" or "random uncovered"
        # but improve the point selection.
        
        while num_uncovered > 0:
            # Find an uncovered triple
            # Using np.argmax on the uncovered array is fast.
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
                
            # Get the triple
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            # Mark as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                
                if not candidates:
                    break
                    
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                # Compute gains for all candidates
                best_gain = -1
                best_point = -1
                
                # Precompute pairs in current block
                block_pairs = list(itertools.combinations(block, 2))
                
                # For each candidate, count how many uncovered triples it covers
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered status for all triples covered by the new point
                # for each pair in block that includes best_point
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def greedy_cover_weighted(random_state):
        """
        Greedy that prioritizes covering triples that are 'rare' or 'hard'.
        We approximate 'hard' by looking at how many blocks in the current
        partial solution *could* cover it, but that's complex.
        Instead, we use a simpler heuristic:
        When selecting the next triple to cover, pick one that has the
        fewest potential completions to a k-block?
        Actually, let's try a different angle:
        When adding a point to the current block, instead of just maximizing
        immediate gain, we might consider the *quality* of the coverage.
        
        Let's try a different strategy for the initial triple selection:
        Instead of the first uncovered triple, pick a random uncovered triple.
        This helps break symmetry.
        """
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        points = list(range(v))
        
        # Maintain a list of uncovered indices for fast random selection
        # Initially all indices
        uncovered_indices = list(range(num_triples))
        
        while uncovered_indices:
            # Pick a random uncovered triple
            idx_pick = rng.randint(0, len(uncovered_indices) - 1)
            start_idx = uncovered_indices[idx_pick]
            # Remove it from the list
            uncovered_indices.pop(idx_pick)
            
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            # Mark as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                
                if not candidates:
                    break
                    
                rng.shuffle(candidates)
                
                best_gain = -1
                best_point = -1
                
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered status
                # We need to remove newly covered triples from uncovered_indices
                # This is O(N) per step if we filter the list, which is fine for small N.
                # But we can just update the array and let the random pick skip covered ones?
                # No, if we pick a covered one, we waste a turn.
                # Let's just update the array and filter the list occasionally?
                # Or simpler: just update the array. When we pick a random index,
                # if it's covered, we just pick another?
                # Actually, maintaining the list is safer.
                
                # Let's optimize the update:
                # Instead of filtering the whole list every time, we can just
                # update the `uncovered` array and when we pick a random index,
                # we check if it's uncovered. If not, we pick again.
                # This is inefficient if many are covered.
                
                # Better: Just update the array. The `uncovered_indices` list
                # becomes stale. Let's abandon the list and just use the array.
                # To pick a random uncovered triple:
                # 1. Pick random index in [0, num_triples)
                # 2. If uncovered, use it.
                # 3. If not, try again.
                # This works if the fraction of uncovered is not too small.
                # Initially 100%, later <1%. This degrades.
                
                # Let's stick to the list but update it efficiently.
                # When we cover a triple, we don't immediately remove it from the list.
                # When we pick a random index, we check if it's covered.
                # If covered, we swap it with the last element and pop.
                
                # Let's implement this "lazy deletion" in the main loop below.
                # For now, let's just update the array.
                
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            # We don't remove from uncovered_indices here.
                            # We will handle it when picking.
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        # Count frequency of each triple
        counts = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def local_search_trim(blocks, random_state, time_limit=2.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        Also try to swap blocks? No, just deletion is safe.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def swap_search(blocks, random_state, time_limit=2.0):
        """
        Try to replace a block with another block that covers the same unique triples
        but allows other blocks to be removed?
        This is complex. Let's stick to deletion for now.
        """
        pass

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        # Check if we have enough time for a full iteration
        if time.time() + 1 > deadline:
            break
            
        # Alternate between standard and weighted (random triple pick) greedy
        if seed_counter % 2 == 0:
            blocks = greedy_cover(seed_counter)
        else:
            blocks = greedy_cover_weighted(seed_counter)
        
        # Quick check: if we already have a very good solution, try to trim
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        # Attempt to trim
        if time.time() < deadline - 1:
            trimmed = trim_blocks(blocks)
            if len(trimmed) < best_len:
                best_len = len(trimmed)
                best_blocks = list(trimmed)
                
            # Attempt local search if we are close to the known best
            if best_len <= 76 and time.time() < deadline - 2:
                ls_blocks = local_search_trim(trimmed, seed_counter, time_limit=1.0)
                if len(ls_blocks) < best_len:
                    best_len = len(ls_blocks)
                    best_blocks = list(ls_blocks)
                    
        if best_len == 69:
            break
            
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return empty if something went wrong, though greedy should always produce something
        return []
