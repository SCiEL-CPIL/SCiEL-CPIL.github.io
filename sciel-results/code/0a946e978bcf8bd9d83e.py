import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    
    # Helper to get triples in a block
    def block_triples(block):
        return set(itertools.combinations(block, t))
    
    # Helper to check if a block is valid (distinct elements)
    def is_valid_block(block):
        return len(set(block)) == len(block) and all(0 <= x < v for x in block)

    def greedy_construct(random_seed, time_budget):
        """
        Construct a covering using a randomized greedy approach.
        At each step, pick a block that covers the most uncovered triples,
        with some randomization to create diversity.
        """
        random.seed(random_seed)
        uncovered = set(all_tsubs)
        blocks = []
        end_time = start_time + time_budget
        
        # Precompute coverage potential for all possible blocks? 
        # C(24,7) is ~346k. For each, 35 triples. 12M ops. 
        # We can't precompute all gains for all blocks every step.
        # We will compute gain on the fly for a subset of candidates or use a heuristic.
        
        # A better heuristic: 
        # 1. Pick a random uncovered triple.
        # 2. Generate a random block containing it.
        # 3. Optimize that block locally to maximize coverage of uncovered triples.
        
        while uncovered:
            if time.time() > end_time:
                break
            
            # Pick a random uncovered triple to anchor the new block
            # This ensures we don't get stuck in a local optimum of "covering only popular triples"
            anchor_triple = random.choice(list(uncovered))
            
            # Build a block around this anchor
            # Start with the anchor triple
            block = list(anchor_triple)
            random.shuffle(block)
            
            # Add k-3 more points to maximize coverage of uncovered triples
            for _ in range(k - t):
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Shuffle to break ties and introduce randomness
                random.shuffle(candidates)
                
                # To speed up, we might not check all v-k candidates if v is large,
                # but for v=24, checking all is fine.
                for x in candidates:
                    # Calculate gain: how many new uncovered triples are covered
                    gain = 0
                    # Triples in new block containing x: choose t-1 from existing block
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            uncovered.difference_update(block_triples(block))
            
        return blocks

    def local_search(blocks, time_budget):
        """
        Intensive local search: Deletion and Swap.
        """
        end_time = start_time + time_budget
        blocks = [sorted(b) for b in blocks]
        
        # Build coverage map
        coverage = {ts: 0 for ts in all_tsubs}
        for block in blocks:
            for ts in block_triples(block):
                coverage[ts] += 1
        
        def try_delete(idx):
            block = blocks[idx]
            for ts in block_triples(block):
                if coverage[ts] <= 1:
                    return False
            for ts in block_triples(block):
                coverage[ts] -= 1
            blocks.pop(idx)
            return True

        def try_swap(idx, old_x, new_x):
            if new_x in blocks[idx]:
                return False
            block = blocks[idx]
            old_triples = block_triples(block)
            new_block = sorted([p for p in block if p != old_x] + [new_x])
            new_triples = block_triples(new_block)
            
            # Check if any triple loses coverage
            for ts in old_triples:
                if ts not in new_triples and coverage[ts] <= 1:
                    return False
            
            # Apply
            for ts in old_triples:
                coverage[ts] -= 1
            for ts in new_triples:
                coverage[ts] += 1
            blocks[idx] = new_block
            return True

        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # 1. Try deletions
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                if try_delete(idx):
                    improved = True
            
            # 2. Try swaps if no deletion or to enable further deletions
            if time.time() > end_time:
                break
            
            # Limit swaps to avoid excessive time
            num_swaps_attempted = 0
            max_swaps = len(blocks) * 10 # Heuristic limit
            
            while num_swaps_attempted < max_swaps and time.time() < end_time:
                num_swaps_attempted += 1
                idx = random.randint(0, len(blocks)-1)
                old_x = random.choice(blocks[idx])
                new_x = random.randint(0, v-1)
                
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                
                if try_swap(idx, old_x, new_x):
                    # After a swap, try to delete some blocks
                    indices2 = list(range(len(blocks)))
                    random.shuffle(indices2)
                    for idx2 in indices2:
                        if try_delete(idx2):
                            improved = True
                            break
                            
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    # Multi-start strategy
    # Use a few diverse seeds and allocate time to each
    num_starts = 5
    time_per_start = (time_limit - 5) / num_starts
    
    for i in range(num_starts):
        if time.time() > time_limit - 10: # Stop if too close to end
            break
            
        seed = 1000 + i * 137 # Distinct seeds
        # Phase 1: Construct
        init_blocks = greedy_construct(seed, time_per_start * 0.4)
        
        if not init_blocks:
            continue
            
        # If construction timed out and didn't cover everything, skip or finish?
        # The greedy loop breaks on time. If uncovered remains, it's invalid.
        # We need to ensure validity. 
        # Let's check if it's a valid cover.
        # Actually, the greedy loop above only breaks if time runs out.
        # If it breaks, `uncovered` is not empty. 
        # We must finish it deterministically if it's incomplete, or skip this start.
        
        # Check validity quickly
        # Rebuild coverage to check
        cov = {ts: 0 for ts in all_tsubs}
        for b in init_blocks:
            for ts in block_triples(b):
                cov[ts] += 1
        if any(cov[ts] == 0 for ts in all_tsubs):
             # Incomplete cover. Finish it greedily (deterministic)
             # This might take a while, but usually few blocks are needed.
             uncovered = {ts for ts in all_tsubs if cov[ts] == 0}
             while uncovered:
                 anchor = min(uncovered)
                 block = list(anchor)
                 for _ in range(k - t):
                     best_x = -1
                     best_gain = -1
                     for x in range(v):
                         if x in block: continue
                         gain = 0
                         for rest in itertools.combinations(block, t-1):
                             ts = tuple(sorted(rest + (x,)))
                             if ts in uncovered:
                                 gain += 1
                         if gain > best_gain:
                             best_gain = gain
                             best_x = x
                     block.append(best_x)
                 block.sort()
                 init_blocks.append(block)
                 for ts in block_triples(block):
                     if ts in uncovered:
                         uncovered.remove(ts)
        
        if len(init_blocks) >= best_size:
            continue
            
        # Phase 2: Local Search
        # Allocate remaining time for this start to local search
        # Be careful not to exceed total time limit
        remaining_for_ls = time_limit - (time.time() - start_time) - 5
        if remaining_for_ls <= 0:
            continue
            
        optimized_blocks = local_search(init_blocks, remaining_for_ls * 0.8)
        
        if len(optimized_blocks) < best_size:
            best_size = len(optimized_blocks)
            best_blocks = optimized_blocks
            
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        blocks = []
        uncovered = set(all_tsubs)
        while uncovered:
            anchor = min(uncovered)
            block = list(anchor)
            for _ in range(k - t):
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(block_triples(block))
        return blocks

    return best_blocks
