"""Covering design C(19,8,5) - efficient greedy with multiple restarts."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 140
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute k-subsets and their covered t-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    ksub_cover = []
    for ksub in all_ksubs:
        covered = [tsub_index[tsub] for tsub in itertools.combinations(ksub, t)]
        ksub_cover.append(covered)
    
    # Reverse mapping: t-subset -> list of k-subset indices
    tsub_to_ksubs = defaultdict(list)
    for ksub_idx, covered in enumerate(ksub_cover):
        for tsub_idx in covered:
            tsub_to_ksubs[tsub_idx].append(ksub_idx)
    
    def greedy_build(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_tsubs))
        gain = [len(c) for c in ksub_cover]
        used = [False] * len(all_ksubs)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            
            # Find bottleneck t-subset
            if len(uncovered) > 50:
                candidates = list(uncovered)[:50]
            else:
                candidates = list(uncovered)
            
            best_tsub = min(candidates, key=lambda ti: sum(
                1 for ki in tsub_to_ksubs[ti] if not used[ki] and gain[ki] > 0
            ))
            
            # Pick best k-subset covering bottleneck
            best_ki = -1
            best_g = -1
            for ki in tsub_to_ksubs[best_tsub]:
                if not used[ki] and gain[ki] > 0 and gain[ki] > best_g:
                    best_g = gain[ki]
                    best_ki = ki
            
            if best_ki < 0:
                break
            
            used[best_ki] = True
            blocks.append(list(all_ksubs[best_ki]))
            
            for ti in ksub_cover[best_ki]:
                if ti in uncovered:
                    uncovered.remove(ti)
                    for other_ki in tsub_to_ksubs[ti]:
                        if not used[other_ki]:
                            gain[other_ki] -= 1
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    for seed in range(200):
        if time.time() > deadline - 15:
            break
        blocks = greedy_build(seed)
        if blocks and len(blocks) < best_count:
            # Quick validity check
            covered = set()
            for b in blocks:
                covered.update(itertools.combinations(b, t))
            if len(covered) == n_tsubs:
                best_count = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        best_blocks = greedy_build(0)
    
    return best_blocks
