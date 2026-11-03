"""Covering design C(19,8,5) - minimize number of 8-blocks covering all 5-subsets of 19 points."""

import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    ts_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each point, list of t-subset indices containing it
    point_indices = defaultdict(list)
    for i, ts in enumerate(all_t_subsets):
        for p in ts:
            point_indices[p].append(i)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = bytearray([1]) * total_t  # 1 = uncovered
        blocks = []
        
        while any(uncovered):
            # Find first uncovered
            first_idx = uncovered.index(1)
            seed_points = set(all_t_subsets[first_idx])
            block = set(seed_points)
            remaining = k - len(block)
            
            while remaining > 0:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = ts_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    best_point = rng.choice([p for p in range(v) if p not in block])
                block.add(best_point)
                remaining -= 1
            
            block_list = sorted(block)
            blocks.append(block_list)
            
            # Mark covered
            for ts in itertools.combinations(block_list, t):
                idx = ts_to_idx[ts]
                uncovered[idx] = 0
        
        return blocks
    
    def check_valid(blocks):
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        return len(covered) == total_t
    
    def local_search_remove(blocks):
        """Remove redundant blocks."""
        blocks = [list(b) for b in blocks]
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                trial = blocks[:i] + blocks[i+1:]
                if check_valid(trial):
                    blocks = trial
                    changed = True
                    break
        return blocks
    
    # Run greedy with different seeds
    for seed in [0, 42, 123, 7, 17]:
        if time.time() > deadline:
            break
        try:
            blocks = greedy_construct(seed)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        except Exception:
            continue
    
    # Local search on best
    if best_blocks is not None and time.time() < deadline - 5:
        improved = local_search_remove(best_blocks)
        if len(improved) < best_count:
            best_blocks = improved
            best_count = len(improved)
    
    if best_blocks is None:
        best_blocks = []
    
    return best_blocks
