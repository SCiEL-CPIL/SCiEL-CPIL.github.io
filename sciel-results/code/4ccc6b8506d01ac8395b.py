import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    # Evaluator kills after 150s, but task contract says 120s. 
    # We set a safe deadline to ensure we return a valid result.
    deadline = start_time + 115.0 

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each point x, the list of t-subsets containing x? 
    # No, better to precompute for each t-subset index, the points in it? 
    # Actually, for the greedy step, we need to know for a partial block B, 
    # which t-subsets are formed by adding x to B.
    
    # Let's precompute a matrix or list structure for fast gain calculation.
    # For a given set of points in the block, we want to count uncovered t-subsets 
    # that can be completed by adding a specific point x.
    
    # Strategy:
    # 1. Maintain a boolean array `uncovered` of length num_tsubsets.
    # 2. For a current block `B` (list of points), and a candidate point `x`:
    #    The new t-subsets covered are those formed by `x` and `t-1` points from `B`.
    #    We can precompute for every pair of points? No, t-1 is 2.
    #    So for each x, and each pair (a,b) in B, the t-subset is {x,a,b}.
    #    We need to check if idx({x,a,b}) is uncovered.
    
    # To speed this up, we can precompute the indices of t-subsets.
    # Let's create a 3D lookup? No, too big.
    # Let's create a dictionary for pairs? 
    # C(24, 2) = 276 pairs.
    # For each pair (a,b), we can store a list of (x, idx) where {x,a,b} is a t-subset.
    # Then for a block B, for each candidate x, we iterate over all pairs in B, 
    # look up the list for that pair, filter by x, and check `uncovered`.
    
    # Actually, simpler:
    # Precompute `ts_idx_by_pair_and_point`?
    # Let's just precompute the indices for all t-subsets in a flat array structure if possible, 
    # or stick to the dictionary but optimize the loop.
    
    # Better optimization:
    # Precompute `subsets_by_points`?
    
    # Let's try a different vectorization approach.
    # For each t-subset index i, we have the tuple of points.
    # We can create a numpy array `ts_points` of shape (num_tsubsets, t).
    ts_points = np.array(all_tsubsets, dtype=np.int8)
    
    # For a given block B (numpy array of points) and candidate x:
    # We want to find all i such that x is in ts_points[i] and the other t-1 points are in B.
    # This is still hard to vectorize directly for arbitrary B.
    
    # Let's stick to the "pair lookup" idea but make it very fast.
    # Precompute: for each pair (a,b) with a<b, a list of (x, idx) such that {x,a,b} is a t-subset.
    # Note: x can be anything not a or b.
    pair_to_ts = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        # ts is a tuple of 3 sorted points
        a, b, c = ts
        # Pairs: (a,b) with c, (a,c) with b, (b,c) with a
        pair_to_ts[(a,b)].append((c, i))
        pair_to_ts[(a,c)].append((b, i))
        pair_to_ts[(b,c)].append((a, i))
    
    def greedy_cover(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered) and time.time() < time_limit:
            # Pick a random uncovered t-subset to start
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            
            while len(block) < k:
                # Find the point x that maximizes gain
                best_x = -1
                best_gain = -1
                
                # We can restrict candidates to those that actually add value
                # But checking all v points is fast enough if the inner check is fast.
                
                # Optimization: Only consider points not in block.
                # Also, we can pre-filter points that don't share any pair with block?
                # If x doesn't form a t-subset with any pair in block, gain is 0.
                
                # Let's calculate gain for all x in range(v) not in block.
                # block_set = set(block) for O(1) lookup? 
                # For v=24, list check is fine.
                
                block_set = set(block)
                # Pairs in block
                block_pairs = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # For each pair in block, check if {x, pair} is uncovered
                    for p1, p2 in block_pairs:
                        # Get the t-subset index for {x, p1, p2}
                        # The points are x, p1, p2. Sort them to find the key.
                        # Since pair_to_ts keys are sorted pairs, we need to handle x.
                        # If x < p1: key is (x, p1), other is p2
                        # If p1 < x < p2: key is (p1, x), other is p2
                        # If x > p2: key is (p1, p2), other is x
                        
                        # Actually, pair_to_ts stores (other_point, idx) for a sorted pair key.
                        # So we just need to find the correct key and check if x is the "other" point.
                        
                        # Let's just use the dictionary lookup directly for the triple.
                        # Triple is (x, p1, p2).
                        triple = tuple(sorted((x, p1, p2)))
                        idx = tsubset_to_idx.get(triple, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x < 0:
                    # No point improves coverage? This shouldn't happen if uncovered is not empty
                    # and we started with an uncovered subset.
                    # If it happens, break to avoid infinite loop.
                    break
                
                block.append(best_x)
            
            if len(block) < k:
                # Failed to complete block?
                continue
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False
        
        if not np.any(uncovered):
            return [list(b) for b in blocks]
        return None

    def remove_redundant(blocks, time_limit):
        if not blocks:
            return []
        if time.time() > time_limit:
            return blocks
        
        # Count coverage
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
        # Precompute t-subsets for each block to avoid recomputation
        block_tsubs = []
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result_blocks = list(blocks)
        result_tsubs = list(block_tsubs)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                
                # Check if block i is redundant
                # i.e., for all ts in result_tsubs[i], cover_count[idx] > 1
                redundant = True
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    changed = True
                    break # Restart loop as indices shifted
        
        return [list(b) for b in result_blocks]

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time for greedy
        # Greedy is the bottleneck. 
        # Estimate: ~100-200 blocks. Each block involves ~4-6 steps.
        # Each step checks ~18 points. 
        # With optimized loops, this should be fast.
        attempt_time = min(remaining - 2, 10) # 10s per attempt is generous
        
        blocks = greedy_cover(seed, time.time() + attempt_time)
        
        if blocks is not None:
            # Try to remove redundant blocks
            # Allocate remaining time for this
            red_time = min(remaining - 2, 10)
            blocks = remove_redundant(blocks, time.time() + red_time)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # If we hit the known lower bound or close, we can stop or keep trying
                if best_count <= 76:
                    # Good result, but keep trying for a bit more?
                    # Or just break if we are very confident?
                    # Let's keep trying to see if we can go lower (unlikely) or just secure the win.
                    pass
        
        seed += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        while np.any(uncovered):
            start_idx = np.where(uncovered)[0][0]
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in block_pairs:
                        triple = tuple(sorted((x, p1, p2)))
                        idx = tsubset_to_idx.get(triple, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False
        
        return blocks

    return best_blocks
