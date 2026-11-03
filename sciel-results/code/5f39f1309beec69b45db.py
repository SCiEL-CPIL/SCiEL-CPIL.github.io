"""Improved covering design solver using k-center diversity with local search."""
import time
import random
import itertools
from collections import defaultdict


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
    
    # Precompute for each t-subset its bitmask representation (for fast containment checks)
    point_mask = [1 << p for p in range(v)]
    
    # For each block (as a bitmask), precompute which t-subsets it covers
    tsub_mask = []
    for ts in all_tsubsets:
        m = 0
        for p in ts:
            m |= point_mask[p]
        tsub_mask.append(m)
    
    # Precompute pairwise point distance: number of t-subsets NOT shared by both points
    # Two points are "far" if they co-occur in few t-subsets
    # dist[p][q] = number of t-subsets containing p but not q (or vice versa)
    # Actually, use: dist[p][q] = |tsubs(p) \ tsubs(q)| + |tsubs(q) \ tsubs(p)|
    # Simpler: dist[p][q] = num_tsubsets - 2*|tsubs(p) ∩ tsubs(q)| + |tsubs(p)| + |tsubs(q)| - 2*|intersection|
    # Even simpler: just count how many t-subsets contain BOTH p and q
    # Then distance = (len(tsubs(p)) + len(tsubs(q)) - 2*co_occur)
    # Higher distance = more diverse
    
    # Precompute co-occurrence: for each pair (p,q), how many t-subsets contain both
    co_occur = [[0] * v for _ in range(v)]
    for ts in all_tsubsets:
        for i in range(t):
            for j in range(i+1, t):
                p, q = ts[i], ts[j]
                co_occur[p][q] += 1
                co_occur[q][p] += 1
    
    # Point degree: number of t-subsets containing each point
    point_deg = [0] * v
    for p in range(v):
        point_deg[p] = len(point_tsubs[p])
    
    # Distance: higher = more diverse (less overlap in t-subset coverage)
    # dist[p][q] = point_deg[p] + point_deg[q] - 2 * co_occur[p][q]
    # This is the size of the symmetric difference of their t-subset sets
    point_dist = [[0] * v for _ in range(v)]
    for p in range(v):
        for q in range(v):
            if p != q:
                point_dist[p][q] = point_deg[p] + point_deg[q] - 2 * co_occur[p][q]
    
    # Greedy with k-center diversity
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
            # Use urgency: prefer t-subsets that are harder to cover
            # (fewer points that could still be added to cover them)
            # Simple approach: random uncovered t-subset
            idx = rng.randrange(num_tsubsets)
            while uncovered[idx] == 0:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            
            # K-center heuristic: greedily add the point that maximizes
            # minimum distance to already-selected points (with coverage tie-breaking)
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                block_set = set(block)
                block_mask = 0
                for p in block:
                    block_mask |= point_mask[p]
                
                best_score = -1
                best_point = -1
                best_min_dist = -1
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Compute minimum distance to selected points
                    min_dist = float('inf')
                    for p in block:
                        d = point_dist[x][p]
                        if d < min_dist:
                            min_dist = d
                    
                    # Compute coverage gain as tie-breaker
                    gain = 0
                    for ts_idx in point_tsubs[x]:
                        if uncovered[ts_idx] == 0:
                            continue
                        ts_m = tsub_mask[ts_idx]
                        if (ts_m & ~point_mask[x]) & ~block_mask == 0:
                            gain += 1
                    
                    # Score: primarily diversity (min_dist), secondarily gain
                    # Normalize to combine: use min_dist as primary, gain as secondary
                    # We want to maximize min_dist, and among ties maximize gain
                    if min_dist > best_min_dist or (min_dist == best_min_dist and gain > best_gain):
                        best_min_dist = min_dist
                        best_gain = gain
                        best_point = x
                        best_score = (min_dist, gain)
                    elif min_dist == best_min_dist and gain == best_gain:
                        # Random tie-breaking
                        if rng.random() < 0.5:
                            best_point = x
                            best_score = (min_dist, gain)
                
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            block_mask = 0
            for p in block:
                block_mask |= point_mask[p]
            for ts_idx in range(num_tsubsets):
                if uncovered[ts_idx] == 0:
                    continue
                if tsub_mask[ts_idx] & ~block_mask == 0:
                    uncovered[ts_idx] = 0
                    remaining -= 1
        
        return blocks
    
    def remove_redundant(blocks, time_limit=None):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if time_limit is None:
            time_limit = deadline
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = [0] * num_tsubsets
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        for bm in block_masks:
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    cover_count[ts_idx] += 1
        
        # A block is redundant if every t-subset it covers is also covered by another block
        keep = [True] * n
        for bi in range(n):
            if not keep[bi]:
                continue
            if time.time() > time_limit:
                break
            bm = block_masks[bi]
            redundant = True
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
            if redundant:
                keep[bi] = False
        
        if all(keep):
            return blocks
        
        new_blocks = [b for i, b in enumerate(blocks) if keep[i]]
        return new_blocks
    
    def local_search(blocks, seed, time_limit=None):
        """Try to remove blocks one at a time if the covering is still valid."""
        if time_limit is None:
            time_limit = deadline
        rng = random.Random(seed)
        n = len(blocks)
        
        # Count how many blocks cover each t-subset
        cover_count = [0] * num_tsubsets
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        for bm in block_masks:
            for ts_idx in range(num_tsubsets):
                if tsub_mask[ts_idx] & ~bm == 0:
                    cover_count[ts_idx] += 1
        
        # Try removing blocks in random order, multiple passes
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
                bm = block_masks[bi]
                can_remove = True
                for ts_idx in range(num_tsubsets):
                    if tsub_mask[ts_idx] & ~bm == 0:
                        if cover_count[ts_idx] == 1:
                            can_remove = False
                            break
                if can_remove:
                    kept[bi] = False
                    improved = True
                    for ts_idx in range(num_tsubsets):
                        if tsub_mask[ts_idx] & ~bm == 0:
                            cover_count[ts_idx] -= 1
        
        new_blocks = [b for i, b in enumerate(blocks) if kept[i]]
        return new_blocks
    
    # Try multiple random seeds and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        blocks = remove_redundant(blocks)
        blocks = local_search(blocks, 0)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 15:
        blocks = greedy_cover(seed)
        if blocks is not None:
            blocks = remove_redundant(blocks, time_limit=deadline - 10)
            blocks = local_search(blocks, seed, time_limit=deadline - 5)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        return []
    
    return best_blocks
