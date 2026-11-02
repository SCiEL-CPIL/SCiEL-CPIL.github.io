"""Covering design C(19,8,5): optimized greedy with local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter


def construct(v, k, t):
    deadline = time.time() + 130
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        uncovered = set(range(n_tsubs))
        blocks = []
        order = list(range(n_tsubs))
        rng.shuffle(order)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            seed_idx = next(idx for idx in order if idx in uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)
        
        if time.time() > deadline or uncovered:
            continue
        
        blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx, deadline)
        
        if time.time() < deadline - 3:
            blocks = _local_search(blocks, v, k, t, tsub_to_idx, deadline, rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 335:
                break
    
    if best_blocks is None:
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            seed_idx = min(uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)
        best_blocks = blocks
    
    return best_blocks


def _remove_redundant(blocks, v, k, t, tsub_to_idx, deadline):
    if not blocks:
        return blocks
    
    cover_count = Counter()
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                tsubs.append(idx)
                cover_count[idx] += 1
        block_tsubs.append(tsubs)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            if not any(cover_count[idx] == 1 for idx in block_tsubs[i]):
                for idx in block_tsubs[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                changed = True
                break
    
    return blocks


def _local_search(blocks, v, k, t, tsub_to_idx, deadline, rng):
    """Simple local search: try swapping points to enable redundancy removal."""
    if not blocks:
        return blocks
    
    cover_count = Counter()
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                tsubs.append(idx)
                cover_count[idx] += 1
        block_tsubs.append(tsubs)
    
    improved = True
    max_passes = 3
    for _ in range(max_passes):
        if not improved or time.time() > deadline - 2:
            break
        improved = False
        
        n_blocks = len(blocks)
        for i in range(n_blocks):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, min(i + 10, n_blocks)):
                if time.time() > deadline - 2:
                    break
                for pi in range(k):
                    if time.time() > deadline - 2:
                        break
                    for pj in range(k):
                        if time.time() > deadline - 2:
                            break
                        if blocks[i][pi] == blocks[j][pj]:
                            continue
                        
                        old_i, old_j = blocks[i][pi], blocks[j][pj]
                        blocks[i][pi], blocks[j][pj] = old_j, old_i
                        blocks[i].sort()
                        blocks[j].sort()
                        
                        # Quick validity check
                        if len(set(blocks[i])) != k or len(set(blocks[j])) != k:
                            blocks[i][pi], blocks[j][pj] = old_i, old_j
                            blocks[i].sort()
                            blocks[j].sort()
                            continue
                        
                        # Try redundancy removal
                        new_blocks = _remove_redundant([list(b) for b in blocks], v, k, t, tsub_to_idx, deadline)
                        if len(new_blocks) < len(blocks):
                            blocks = new_blocks
                            # Rebuild coverage
                            cover_count = Counter()
                            block_tsubs = []
                            for block in blocks:
                                tsubs = []
                                for ts in itertools.combinations(block, t):
                                    idx = tsub_to_idx.get(tuple(ts))
                                    if idx is not None:
                                        tsubs.append(idx)
                                        cover_count[idx] += 1
                                block_tsubs.append(tsubs)
                            improved = True
                            break
                        else:
                            blocks[i][pi], blocks[j][pj] = old_i, old_j
                            blocks[i].sort()
                            blocks[j].sort()
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return blocks
# EVOLVE-BLOCK-END
