"""Covering design C(19,7,4) via dynamic weighted greedy with local search."""
import time
import numpy as np
from itertools import combinations
from collections import defaultdict

def _precompute(v, k, t):
    """Precompute block-to-subset incidence structure."""
    all_subsets = list(combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute which t-subsets it covers
    block_subsets = []
    for block in all_blocks:
        covered = []
        for sub in combinations(block, t):
            covered.append(subset_to_idx[sub])
        block_subsets.append(np.array(covered, dtype=np.int32))
    
    # Reverse mapping: for each subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for bi, subs in enumerate(block_subsets):
        for si in subs:
            subset_to_blocks[si].append(bi)
    
    return all_blocks, block_subsets, n_subsets, subset_to_idx, subset_to_blocks


def _dynamic_greedy(v, k, t, all_blocks, block_subsets, n_subsets,
                    subset_to_blocks, rng, deadline):
    """
    Dynamic greedy: maintain weights for uncovered subsets.
    At each step, pick a random uncovered subset, then among blocks covering it,
    pick the one with maximum weighted gain.
    """
    uncovered = np.ones(n_subsets, dtype=np.float64)
    weights = np.ones(n_subsets, dtype=np.float64)
    
    blocks_chosen = []
    used_blocks = set()
    total_uncovered = n_subsets
    
    while total_uncovered > 0:
        if time.time() > deadline:
            break
        
        unc_mask = uncovered > 0
        unc_idx = np.where(unc_mask)[0]
        if len(unc_idx) == 0:
            break
        
        # Pick a random uncovered subset as focus
        target = unc_idx[rng.integers(len(unc_idx))]
        
        # Get all blocks covering this subset
        candidate_blocks = subset_to_blocks[target]
        
        # Score each candidate: sum of weights of uncovered subsets it covers
        best_block = None
        best_gain = -1.0
        
        for bi in candidate_blocks:
            if bi in used_blocks:
                continue
            subsets_in_block = block_subsets[bi]
            gain = weights[subsets_in_block].sum()
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        
        if best_block is None or best_gain <= 0:
            break
        
        blocks_chosen.append(all_blocks[best_block])
        used_blocks.add(best_block)
        
        # Update uncovered
        subsets_in_block = block_subsets[best_block]
        for si in subsets_in_block:
            if uncovered[si] > 0:
                uncovered[si] = 0
                total_uncovered -= 1
        
        # Dynamic weight update: boost weights of still-uncovered subsets
        if total_uncovered > 0:
            unc_idx_now = np.where(uncovered > 0)[0]
            if len(unc_idx_now) > 0:
                weights[unc_idx_now] *= 1.005
    
    return blocks_chosen


def _local_search_reduce(blocks, v, k, t, deadline):
    """Remove redundant blocks via iterative deletion."""
    if not blocks:
        return blocks
    
    all_subsets = list(combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Build coverage: for each subset, which blocks cover it
    subset_covering = defaultdict(list)
    block_subsets_list = []
    for i, b in enumerate(blocks):
        subs = [subset_to_idx[s] for s in combinations(b, t)]
        block_subsets_list.append(subs)
        for si in subs:
            subset_covering[si].append(i)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            redundant = True
            for si in block_subsets_list[i]:
                if len(subset_covering[si]) == 1:
                    redundant = False
                    break
            if redundant:
                for si in block_subsets_list[i]:
                    subset_covering[si].remove(i)
                blocks.pop(i)
                block_subsets_list.pop(i)
                # Reindex
                for j in range(len(block_subsets_list)):
                    for si in block_subsets_list[j]:
                        subset_covering[si] = [x if x < i else x - 1 for x in subset_covering[si]]
                changed = True
                break
    
    return blocks


def _verify_coverage(blocks, v, k, t):
    covered = set()
    for b in blocks:
        for sub in combinations(b, t):
            covered.add(sub)
    return len(covered) == len(list(combinations(range(v), t)))


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    all_blocks, block_subsets, n_subsets, subset_to_idx, subset_to_blocks = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    for restart in range(25):
        if time.time() > deadline - 35.0:
            break
        
        rng = np.random.default_rng(seed=42 + restart)
        blocks = _dynamic_greedy(v, k, t, all_blocks, block_subsets, n_subsets,
                                  subset_to_blocks, rng, deadline - 25.0)
        
        if not blocks:
            continue
        
        if time.time() < deadline - 10.0:
            blocks = _local_search_reduce(blocks, v, k, t, deadline - 10.0)
        
        if _verify_coverage(blocks, v, k, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        if best_count <= 151:
            break
    
    if best_blocks is None:
        rng = np.random.default_rng(seed=0)
        best_blocks = _dynamic_greedy(v, k, t, all_blocks, block_subsets, n_subsets,
                                       subset_to_blocks, rng, deadline)
    
    return best_blocks
