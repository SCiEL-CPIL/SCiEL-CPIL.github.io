import itertools
import time
import random
import sys

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    start_time = time.time()
    # Safety margin: 150s limit, use 140s deadline to be safe
    deadline = start_time + 140.0
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed: idx_3d[a][b][c] = index of triple (a,b,c)
    # -1 if not a valid triple
    idx_3d = [[[-1]*v for _ in range(v)] for _ in range(v)]
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a][b][c] = triple_to_idx[(t1, t2, t3)]
        
    def get_triple_idx(a, b, c):
        return idx_3d[a][b][c]
        
    def greedy_cover(random_state, time_check=None):
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            if time_check and time_check():
                # If time is up, we need to ensure validity. 
                # The current partial blocks might not cover everything.
                # For simplicity in the loop, we assume we finish the current block.
                # But if we break here, we return invalid. 
                # Better to check before starting a new block.
                pass
            
            # Pick first uncovered for determinism within a run?
            # Let's pick a random one to help with local minima.
            # We need to find one.
            # Bytearray scan is fast.
            
            start_idx = 0
            while start_idx < num_triples and uncovered[start_idx] == 0:
                start_idx += 1
                
            if start_idx >= num_triples:
                break # Should not happen if num_uncovered > 0
                
            # Get the triple
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            # Mark this triple as covered?
            # It will be covered by the block we build.
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block_set]
                
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Count triples (x, u, v) where u, v in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            # Check if triple (x, p1, p2) is uncovered
                            idx = get_triple_idx(x, p1, p2)
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    
                if best_point == -1:
                    # Should not happen if num_uncovered > 0 and v > k
                    # Fallback to random
                    best_point = rng.choice(candidates) if candidates else -1
                    if best_point == -1:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered status for all triples covered by the new point x
                # These are triples (x, p1, p2) for p1, p2 in block (excluding x)
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        # Triple is (best_point, p1, p2)
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        # Count frequency of each triple
        counts = [0] * num_triples
        
        for block in blocks:
            # Iterate all triples in the block
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
                # Check if any triple in this block has count 1
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def local_search_trim(blocks, random_state, time_limit=5.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            # Shuffle active blocks indices
            active_indices = [i for i in range(len(blocks)) if active[i]]
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                # Check if block i is essential
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove it
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    # We use a slightly tighter deadline for the loop to allow for final processing
    loop_deadline = deadline - 5.0
    
    while time.time() < loop_deadline:
        # Run greedy
        blocks = greedy_cover(seed_counter)
        
        # Trim
        blocks = trim_blocks(blocks)
        
        # Local search trim
        # Use a short time limit for local search to allow more restarts
        blocks = local_search_trim(blocks, seed_counter, time_limit=1.0)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit the lower bound, stop?
        # Lower bound is 69.
        if best_len == 69:
            break
            
        seed_counter += 1
        
    # If we have time, do a more intensive local search on the best block found
    if best_blocks is not None and time.time() < deadline - 2.0:
        # Try a longer local search on the best solution
        final_blocks = local_search_trim(best_blocks, seed_counter, time_limit=deadline - time.time() - 2.0)
        if len(final_blocks) < best_len:
            best_blocks = final_blocks
            best_len = len(best_blocks)

    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return empty or a trivial covering?
        # Should not happen.
        return []
