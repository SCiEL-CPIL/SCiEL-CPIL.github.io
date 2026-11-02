"""Improved covering design C(24,7,3) using hypergraph peeling + local search refinement."""
import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (blocks) and their coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute which t-subsets it covers
    block_triples = []
    for b in all_blocks:
        triples = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_triples.append(triples)

    # Precompute: for each t-subset, which blocks cover it
    # triple_to_blocks[ts_idx] = list of block indices
    triple_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, triples in enumerate(block_triples):
        for ti in triples:
            triple_to_blocks[ti].append(bi)

    # Precompute: for each point, which t-subsets contain it
    point_triples = [[] for _ in range(v)]
    for ts_idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_triples[p].append(ts_idx)

    def greedy_peeling(seed=None, deadline=None):
        """Hypergraph peeling: iteratively remove min-degree point, add best block covering its incident triples."""
        rng = random.Random(seed)
        
        # Uncovered triples as a set for O(1) lookup
        uncovered = set(range(n_tsubs))
        blocks_used = []
        used_blocks = set()
        
        # Degree of each point: number of uncovered triples containing it
        point_degree = [0] * v
        for p in range(v):
            for ti in point_triples[p]:
                if ti in uncovered:
                    point_degree[p] += 1
        
        while uncovered:
            if deadline and time.time() > deadline:
                break
            
            # Find point with minimum positive degree
            min_deg = float('inf')
            min_points = []
            for p in range(v):
                if point_degree[p] > 0:
                    if point_degree[p] < min_deg:
                        min_deg = point_degree[p]
                        min_points = [p]
                    elif point_degree[p] == min_deg:
                        min_points.append(p)
            
            if not min_points:
                break
            
            # Choose a point (randomized if seed provided)
            if seed is not None and len(min_points) > 1:
                chosen_point = rng.choice(min_points)
            else:
                chosen_point = min_points[0]
            
            # Find all blocks that cover at least one uncovered triple incident to chosen_point
            # A block covers an uncovered triple incident to chosen_point if:
            # 1. The block contains chosen_point
            # 2. The block contains at least one uncovered triple that contains chosen_point
            
            best_block = None
            best_gain = -1
            candidates = []
            
            for bi in range(n_blocks):
                if bi in used_blocks:
                    continue
                b = all_blocks[bi]
                if chosen_point not in b:
                    continue
                
                # Count how many uncovered triples this block covers
                gain = 0
                for ti in block_triples[bi]:
                    if ti in uncovered:
                        gain += 1
                
                if gain > 0:
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [bi]
                    elif gain == best_gain:
                        candidates.append(bi)
            
            if not candidates:
                # Fallback: pick any block containing chosen_point with max uncovered coverage
                for bi in range(n_blocks):
                    if bi in used_blocks:
                        continue
                    b = all_blocks[bi]
                    if chosen_point not in b:
                        continue
                    gain = sum(1 for ti in block_triples[bi] if ti in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [bi]
                    elif gain == best_gain and gain > 0:
                        candidates.append(bi)
            
            if not candidates:
                break
            
            if len(candidates) == 1:
                chosen_block_idx = candidates[0]
            else:
                chosen_block_idx = rng.choice(candidates) if seed is not None else candidates[0]
            
            chosen_block = all_blocks[chosen_block_idx]
            blocks_used.append(list(chosen_block))
            used_blocks.add(chosen_block_idx)
            
            # Update uncovered set and point degrees
            for ti in block_triples[chosen_block_idx]:
                if ti in uncovered:
                    uncovered.discard(ti)
                    # Update point degrees for all points in this triple
                    for p in all_tsubs[ti]:
                        point_degree[p] -= 1
        
        return blocks_used

    def local_search(blocks, seed=None, deadline=None):
        """Try to improve by replacing blocks to reduce count."""
        rng = random.Random(seed)
        
        # Verify current covering
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(tsub_to_idx[ts])
        
        if len(covered) < n_tsubs:
            return blocks  # Invalid, return as-is
        
        # Try removing redundant blocks
        n_blocks = len(blocks)
        block_triples_list = []
        for b in blocks:
            triples = set(tsub_to_idx[ts] for ts in itertools.combinations(b, t))
            block_triples_list.append(triples)
        
        # Greedy removal: try to remove each block if its coverage is subset of others
        removed = [False] * n_blocks
        for _ in range(5):  # multiple passes
            if deadline and time.time() > deadline:
                break
            for i in range(n_blocks):
                if removed[i]:
                    continue
                # Check if block i's coverage is fully covered by non-removed blocks
                covered_by_others = set()
                for j in range(n_blocks):
                    if j == i or removed[j]:
                        continue
                    covered_by_others |= block_triples_list[j]
                if block_triples_list[i] <= covered_by_others:
                    removed[i] = True
        
        new_blocks = [blocks[i] for i in range(n_blocks) if not removed[i]]
        
        # Verify after removal
        covered = set()
        for b in new_blocks:
            for ts in itertools.combinations(b, t):
                covered.add(tsub_to_idx[ts])
        
        if len(covered) == n_tsubs:
            return new_blocks
        else:
            return blocks  # Removal broke coverage, revert

    def verify_covering(blocks):
        """Check if all t-subsets are covered."""
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(tsub_to_idx[ts])
        return len(covered) == n_tsubs

    # Strategy: multiple randomized peeling runs, keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Deterministic run first
    blocks = greedy_peeling(seed=None, deadline=deadline)
    if blocks and verify_covering(blocks):
        blocks = local_search(blocks, seed=None, deadline=deadline)
        if verify_covering(blocks):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                sys.stderr.write(f"Deterministic: {best_len} blocks\n")
    
    # Randomized restarts
    num_tries = 0
    while time.time() < deadline - 10:
        seed = 1000 + num_tries
        blocks = greedy_peeling(seed=seed, deadline=deadline)
        if blocks:
            if verify_covering(blocks):
                blocks = local_search(blocks, seed=seed, deadline=deadline)
                if verify_covering(blocks):
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = blocks
                        sys.stderr.write(f"Randomized {num_tries}: {best_len} blocks\n")
        num_tries += 1
        if num_tries > 300:  # safety limit
            break
    
    if best_blocks is None:
        # Emergency fallback: pure greedy
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            # Pick first uncovered triple
            first_triple = min(uncovered)
            block = list(all_tsubs[first_triple])
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_x = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for ts in itertools.combinations(block, t - 1):
                        ts_with_x = tuple(sorted(ts + (x,)))
                        if ts_with_x in tsub_to_idx and tsub_to_idx[ts_with_x] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) < k:
                break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered.discard(idx)
        
        return blocks
    
    return best_blocks
