"""Improved covering design solver using fast greedy and local search."""
import time
import random
import itertools


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubs[p].append(i)
    
    # Point degree for tie-breaking (lower degree = more peripheral = potentially better for diversity?)
    point_deg = [len(point_tsubs[p]) for p in range(v)]
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = bytearray(num_tsubsets)  # 1 = uncovered
        for i in range(num_tsubsets):
            uncovered[i] = 1
        
        blocks = []
        remaining = num_tsubsets
        
        while remaining > 0:
            if time.time() > deadline:
                return None  # timeout
            
            # Pick an uncovered t-subset to seed the block
            # Random selection is often fine, but picking one with fewer "extensions" might be better.
            # For speed, we just pick a random uncovered one.
            idx = rng.randrange(num_tsubsets)
            while uncovered[idx] == 0:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            block_set = set(block)
            
            # Track current coverage of this block to calculate incremental gain efficiently?
            # Since k is small (8), we can just recompute gain for each candidate.
            # Gain of adding x to B = number of t-subsets containing x that are:
            # 1. Uncovered
            # 2. Not fully contained in B (i.e., they need x and possibly others, but since we add x, 
            #    we need to check if the rest of the t-subset is in B U {x})
            # Actually, a t-subset is covered by B U {x} if it is a subset of B U {x}.
            # Since t=5 and we are adding x, the t-subset must contain x and 4 other points from B.
            # So gain(x) = count of uncovered t-subsets S such that x in S and S \ {x} subset of B.
            
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_gain = -1
                best_point = -1
                best_tie = float('inf') # Prefer lower degree? Or random?
                candidates = [x for x in range(v) if x not in block_set]
                
                # Shuffle candidates to break ties randomly
                rng.shuffle(candidates)
                
                for x in candidates:
                    # Calculate gain
                    gain = 0
                    # Iterate through t-subsets containing x
                    for ts_idx in point_tsubs[x]:
                        if uncovered[ts_idx] == 0:
                            continue
                        # Check if ts is covered by block U {x}
                        # ts contains x. Check if other 4 points are in block.
                        ts = all_tsubsets[ts_idx]
                        # Since x is in ts, we can just check if all points in ts are in block_set U {x}
                        # Optimized: check if ts is subset of block_set + {x}
                        # We know x is in ts. So we need ts - {x} subset of block_set.
                        # Since ts has size 5, ts - {x} has size 4.
                        covered = True
                        for p in ts:
                            if p == x:
                                continue
                            if p not in block_set:
                                covered = False
                                break
                        if covered:
                            gain += 1
                    
                    # Tie-breaking: 
                    # 1. Higher gain
                    # 2. If gain equal, prefer point with lower degree (more peripheral) 
                    #    to encourage diversity in future steps? Or random?
                    # Let's use random tie-breaking to allow diverse restarts.
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_tie = rng.random()
                    elif gain == best_gain and gain > 0:
                        # Random tie break
                        if rng.random() < 0.5:
                            best_point = x
                            best_tie = rng.random()
                    # If gain is 0 for all, we pick the first one (or random)
                    # The loop above handles gain > -1. If all gains are 0, best_gain becomes 0.
                
                if best_point == -1:
                    # Should not happen if remaining > 0 and v > k
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # For each t-subset, if it's uncovered and subset of block, mark covered
            block_mask_set = set(block)
            for ts_idx in range(num_tsubsets):
                if uncovered[ts_idx] == 0:
                    continue
                ts = all_tsubsets[ts_idx]
                # Check if ts is subset of block
                # Since v is small, set intersection is fast
                if all(p in block_mask_set for p in ts):
                    uncovered[ts_idx] = 0
                    remaining -= 1
        
        return blocks
    
    def local_search(blocks, seed, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        rng = random.Random(seed)
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Precompute coverage lists for each block
        # block_covered[i] = list of t-subset indices covered by block i
        block_covered = []
        for b in blocks:
            b_set = set(b)
            covered_list = []
            for ts_idx, ts in enumerate(all_tsubsets):
                if all(p in b_set for p in ts):
                    covered_list.append(ts_idx)
            block_covered.append(covered_list)
        
        # Count coverage
        cover_count = [0] * num_tsubsets
        for cl in block_covered:
            for ts_idx in cl:
                cover_count[ts_idx] += 1
        
        kept = [True] * n
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            for bi in order:
                if time.time() > time_limit:
                    break
                if not kept[bi]:
                    continue
                
                # Check if block bi is redundant
                redundant = True
                for ts_idx in block_covered[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    kept[bi] = False
                    improved = True
                    # Update counts
                    for ts_idx in block_covered[bi]:
                        cover_count[ts_idx] -= 1
        
        new_blocks = [b for i, b in enumerate(blocks) if kept[i]]
        return new_blocks
    
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        blocks = local_search(blocks, 0, time_limit=deadline - 10)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 20:
        blocks = greedy_cover(seed)
        if blocks is not None:
            # Only do local search if we have time and potential to improve
            # Or always do it if it's fast
            blocks = local_search(blocks, seed, time_limit=deadline - 5)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        return []
    
    return best_blocks
