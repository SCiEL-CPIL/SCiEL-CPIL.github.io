import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for fast gain calculation using bitsets
    # For each pair (a, b), store a bitset of indices of t-subsets containing that pair.
    # Since num_tsubsets can be large (2024 for v=24, t=3), we use Python integers as bitsets.
    # Actually, 2024 bits is ~253 bytes. Operations on these are fast in Python big ints.
    
    # But we need to check if a specific index is uncovered.
    # Let's maintain `uncovered` as a numpy boolean array for easy `np.any` and indexing.
    # For gain calculation:
    # Gain(x, B) = number of uncovered t-subsets containing x and t-1 points from B.
    # For t=3: Gain(x, B) = sum_{p in B} count_uncovered(x, p)
    # Where count_uncovered(x, p) is 1 if the t-subset {x, p, other} is uncovered?
    # No, for t=3, the t-subset is determined by x and pair (p1, p2).
    # So Gain(x, B) = sum_{p1 < p2 in B} 1[uncovered(idx(x, p1, p2))]
    
    # Let's precompute idx(x, p1, p2) for all x, p1, p2?
    # v=24, so 24^3 = 13824 entries. Small enough to store in a 3D array or dict.
    # Let's use a 3D array `triple_idx[x][p1][p2]`.
    # If x, p1, p2 are not distinct or not sorted, handle carefully.
    # Actually, we can just store the index for sorted triples.
    
    triple_idx = np.full((v, v, v), -1, dtype=np.int32)
    for i, ts in enumerate(all_tsubsets):
        a, b, c = ts
        # Store in all permutations? No, just sorted access.
        # But in gain calc, we have x, p1, p2 which might not be sorted.
        # We can sort them and look up.
        # Or precompute a lookup for any order.
        # Let's just use the dictionary `tsubset_to_idx` but optimize the access.
        # Or, since v is small, we can precompute a 2D array for pairs?
        
        # Let's stick to a fast lookup.
        # For t=3, we can precompute `pair_to_third_point_idx`?
        # No, the third point is the variable x.
        
        # Let's precompute `idx_lookup[x][p1][p2]` where x, p1, p2 are distinct.
        # We will fill it for all permutations of the sorted triple.
        triple_idx[a, b, c] = i
        triple_idx[a, c, b] = i
        triple_idx[b, a, c] = i
        triple_idx[b, c, a] = i
        triple_idx[c, a, b] = i
        triple_idx[c, b, a] = i

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
                best_x = -1
                best_gain = -1
                block_set = set(block)
                
                # For t=3, gain is sum over pairs in block
                # block_pairs = list(itertools.combinations(block, 2))
                
                # Optimization: Precompute gains for all x?
                # For each x not in block:
                #   gain = 0
                #   for p1, p2 in block_pairs:
                #       idx = triple_idx[x][p1][p2]
                #       if idx != -1 and uncovered[idx]: gain += 1
                
                # We can vectorize this?
                # For a fixed x, we want to count uncovered[idx] for idx in {triple_idx[x][p1][p2] for p1,p2 in block_pairs}
                
                # Let's just loop, it's fast enough with numpy array access.
                # triple_idx is a numpy array, so triple_idx[x][p1][p2] is fast.
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Unroll for t=3?
                    # For p1 in block:
                    #   for p2 in block, p2 > p1:
                    #       idx = triple_idx[x, p1, p2]
                    #       if idx >= 0 and uncovered[idx]: gain += 1
                    
                    # We can use numpy sum if we gather indices.
                    # indices = []
                    # for p1, p2 in itertools.combinations(block, 2):
                    #     idx = triple_idx[x, p1, p2]
                    #     if idx >= 0: indices.append(idx)
                    # if indices:
                    #     gain = np.sum(uncovered[indices])
                    
                    # This might be faster than Python loop if block is large.
                    # But itertools.combinations creates objects.
                    
                    # Let's try the Python loop first, it's simple.
                    for p1 in block:
                        for p2 in block:
                            if p1 == p2: continue
                            # Avoid double counting? 
                            # The pair (p1, p2) and (p2, p1) represent the same subset {x, p1, p2}.
                            # triple_idx[x, p1, p2] is the same as triple_idx[x, p2, p1].
                            # So we would count it twice if we iterate all p1, p2.
                            # We must iterate unique pairs.
                            # So we need combinations.
                            
                    # Back to combinations.
                    for p1, p2 in itertools.combinations(block, 2):
                        idx = triple_idx[x, p1, p2]
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x < 0:
                    break
                
                block.append(best_x)
            
            if len(block) < k:
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
        
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
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
                
                redundant = True
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in result_blocks]

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        attempt_time = min(remaining - 2, 10)
        
        blocks = greedy_cover(seed, time.time() + attempt_time)
        
        if blocks is not None:
            red_time = min(remaining - 2, 10)
            blocks = remove_redundant(blocks, time.time() + red_time)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        seed += 1
    
    if best_blocks is None:
        # Fallback
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
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in itertools.combinations(block, 2):
                        idx = triple_idx[x, p1, p2]
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
