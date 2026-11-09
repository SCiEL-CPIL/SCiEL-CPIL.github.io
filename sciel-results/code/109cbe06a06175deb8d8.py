"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        for p in ts:
            point_tsubs[p].append(tsub_to_idx[ts])
    
    def compute_gain(block_set, point, coverage):
        """Compute how many new t-subsets adding 'point' to block_set would cover."""
        gain = 0
        for tsub_idx in point_tsubs[point]:
            if coverage[tsub_idx] == 0:
                ts = all_tsubs[tsub_idx]
                others = [x for x in ts if x != point]
                if all(o in block_set for o in others):
                    gain += 1
        return gain
    
    def greedy_cover():
        """Randomized greedy set cover with reverse indexing."""
        coverage = [0] * n_tsubs
        blocks = []
        
        while True:
            if time.time() > deadline:
                return None  # timeout
            
            # Find an uncovered t-subset
            uncovered_idx = None
            for i in range(n_tsubs):
                if coverage[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx is None:
                break
            
            # Build a block that covers this t-subset and maximizes new coverage
            seed_tsub = all_tsubs[uncovered_idx]
            block_set = set(seed_tsub)
            
            # Greedily add points
            for _ in range(k - t):
                if time.time() > deadline:
                    return None
                
                best_point = None
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                # Shuffle for randomization
                rng.shuffle(candidates)
                
                for p in candidates:
                    gain = compute_gain(block_set, p, coverage)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    if gain == n_tsubs:  # max possible
                        break
                
                if best_point is not None:
                    block_set.add(best_point)
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1
        
        return blocks
    
    def local_search_remove(blocks):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks
        
        # coverage_count[i] = number of blocks covering t-subset i
        coverage_count = [0] * n_tsubs
        block_tsubs = []  # for each block, list of t-subset indices it covers
        
        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)
        
        # Try to remove each block if it doesn't break coverage
        keep = [True] * len(blocks)
        
        for i in range(len(blocks)):
            if not keep[i]:
                continue
            # Check if all t-subsets in this block are covered by other blocks
            can_remove = True
            for tsub_idx in block_tsubs[i]:
                if coverage_count[tsub_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                keep[i] = False
                # Update coverage
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1
        
        return [blocks[i] for i in range(len(blocks)) if keep[i]]
    
    def pairwise_merge_polish(blocks):
        """Try to merge pairs of blocks if their union is still a valid k-set."""
        if not blocks or len(blocks) < 2:
            return blocks
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_blocks = []
            used = [False] * len(blocks)
            
            for i in range(len(blocks)):
                if used[i]:
                    continue
                merged = False
                for j in range(i + 1, len(blocks)):
                    if used[j]:
                        continue
                    # Check if union of block i and j has size <= k
                    union = set(blocks[i]) | set(blocks[j])
                    if len(union) <= k:
                        # Check if this merged block covers all t-subsets that i and j covered
                        # and doesn't break any coverage
                        merged_block = sorted(union)
                        # Verify coverage: every t-subset in merged_block must be coverable
                        # Actually, we just need to verify that removing i and j and adding merged
                        # still covers everything. Since merged contains both i and j, all t-subsets
                        # in i and j are in merged. So it's valid.
                        new_blocks.append(merged_block)
                        used[i] = True
                        used[j] = True
                        merged = True
                        improved = True
                        break
                if not merged:
                    new_blocks.append(blocks[i])
                    used[i] = True
            
            blocks = new_blocks
        
        return blocks
    
    # Run greedy with randomization and local search
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover()
        if blocks is None:
            continue
        
        # Local search to remove redundant blocks
        blocks = local_search_remove(blocks)
        
        # Pairwise merge polish
        blocks = pairwise_merge_polish(blocks)
        
        # Final local search after merges
        blocks = local_search_remove(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
    
    if best_blocks is not None:
        return best_blocks
    
    # Fallback: simple greedy without local search
    blocks = greedy_cover()
    if blocks is not None:
        return blocks
    
    # Last resort: original greedy
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
