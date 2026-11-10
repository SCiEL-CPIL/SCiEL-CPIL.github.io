"""Covering design C(19,7,4) – sparse matrix greedy with local search refinement."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict

_DEADLINE_S = 145.0  # Safety margin from 150s limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _build_incidence_matrix(v, k, t):
    """
    Build sparse incidence structure for covering design.
    Returns:
      - tsub_list: list of all t-subsets as tuples
      - block_list: list of all k-subsets as tuples  
      - tsub_to_idx: dict mapping t-subset tuple to its index
      - block_to_tsubs: list where block_to_tsubs[b] = set of t-subset indices in block b
      - tsub_in_blocks: list where tsub_in_blocks[ts] = set of block indices containing tsub ts
    """
    tsub_list = list(itertools.combinations(range(v), t))
    block_list = list(itertools.combinations(range(v), k))
    
    n_tsubs = len(tsub_list)
    n_blocks = len(block_list)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsub_list)}
    
    # For each block, compute which t-subsets it contains
    block_to_tsubs = [None] * n_blocks
    tsub_in_blocks = [set() for _ in range(n_tsubs)]
    
    for b_idx, block in enumerate(block_list):
        tsubs_in_block = set()
        for tsub in itertools.combinations(block, t):
            ts_idx = tsub_to_idx[tsub]
            tsubs_in_block.add(ts_idx)
            tsub_in_blocks[ts_idx].add(b_idx)
        block_to_tsubs[b_idx] = tsubs_in_block
    
    return tsub_list, block_list, tsub_to_idx, block_to_tsubs, tsub_in_blocks


def _greedy_construct(v, k, t):
    """
    Construct a covering using sparse matrix greedy strategy.
    Uses numpy for fast dot product computations to select blocks maximizing uncovered coverage.
    """
    tsub_list, block_list, tsub_to_idx, block_to_tsubs, tsub_in_blocks = _build_incidence_matrix(v, k, t)
    
    n_tsubs = len(tsub_list)
    n_blocks = len(block_list)
    
    # Track uncovered t-subsets as a boolean array
    uncovered = np.ones(n_tsubs, dtype=np.int32)
    selected_blocks = []
    selected_block_set = set()
    
    # Point frequency for tie-breaking
    point_freq = np.zeros(v, dtype=np.int32)
    
    while np.any(uncovered):
        if _time_left() < 2.0:
            break
        
        # For each candidate block, compute how many currently uncovered t-subsets it covers
        # This is the dot product of block's tsub mask with uncovered mask
        
        best_block_idx = -1
        best_gain = 0
        best_point_freq_sum = float('inf')
        
        # Iterate through blocks not yet selected
        for b_idx in range(n_blocks):
            if b_idx in selected_block_set:
                continue
            
            tsubs_in_block = block_to_tsubs[b_idx]
            if not tsubs_in_block:
                continue
            
            # Count uncovered t-subsets in this block
            gain = 0
            for ts_idx in tsubs_in_block:
                if uncovered[ts_idx]:
                    gain += 1
            
            if gain == 0:
                continue
            
            # Tie-break by point frequency (prefer blocks with less-used points)
            block = block_list[b_idx]
            pf_sum = sum(point_freq[p] for p in block)
            
            if gain > best_gain or (gain == best_gain and pf_sum < best_point_freq_sum):
                best_gain = gain
                best_block_idx = b_idx
                best_point_freq_sum = pf_sum
        
        if best_block_idx == -1:
            break
        
        # Select this block
        selected_blocks.append(block_list[best_block_idx])
        selected_block_set.add(best_block_idx)
        
        # Update uncovered mask
        for ts_idx in block_to_tsubs[best_block_idx]:
            uncovered[ts_idx] = 0
        
        # Update point frequencies
        for p in block_list[best_block_idx]:
            point_freq[p] += 1
    
    return [list(b) for b in selected_blocks]


def _remove_redundant_blocks(blocks, v, k, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Build tsub -> set of block indices mapping
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)
    
    # Remove redundant blocks iteratively
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            # Check if block i is redundant
            redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                if len(tsub_to_blocks[tsub]) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for tsub in itertools.combinations(blocks[i], t):
                    tsub_to_blocks[tsub].discard(i)
                blocks.pop(i)
                # Update indices in tsub_to_blocks
                for tsub, bset in tsub_to_blocks.items():
                    new_bset = set()
                    for j in bset:
                        if j < i:
                            new_bset.add(j)
                        elif j > i:
                            new_bset.add(j - 1)
                    tsub_to_blocks[tsub] = new_bset
                changed = True
                break
    
    return blocks


def _local_search_replace(blocks, v, k, t, max_iters=200):
    """
    Try to replace blocks with better ones that cover their unique t-subsets
    and potentially more, while maintaining validity.
    """
    if not blocks:
        return blocks
    
    # Build tsub -> set of block indices mapping
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)
    
    # Point frequency for heuristic
    point_freq = [0] * v
    for block in blocks:
        for p in block:
            point_freq[p] += 1
    
    for _ in range(max_iters):
        if _time_left() < 3.0:
            break
        
        improved = False
        # Shuffle block order for diversity
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if _time_left() < 3.0:
                break
            
            # Find unique t-subsets for block i
            unique_tsubs = set()
            for tsub in itertools.combinations(blocks[i], t):
                if len(tsub_to_blocks[tsub]) == 1:
                    unique_tsubs.add(tsub)
            
            if not unique_tsubs:
                continue
            
            # Try to find a replacement block that covers all unique_tsubs
            # The new block must contain all points that appear in unique_tsubs
            # Actually, it must cover all unique_tsubs, meaning for each tsub in unique_tsubs,
            # the new block must contain that tsub.
            
            # Collect all points that must be in the new block
            must_include = set()
            for tsub in unique_tsubs:
                must_include.update(tsub)
            
            if len(must_include) > k:
                continue  # Can't fit all required points
            
            # Find candidate blocks containing all must_include points
            remaining_points = [p for p in range(v) if p not in must_include]
            need_more = k - len(must_include)
            
            if need_more < 0:
                continue
            
            best_new_block = None
            best_gain = -1
            
            for comb in itertools.combinations(remaining_points, need_more):
                new_block = tuple(sorted(list(must_include) + list(comb)))
                # Check if this new block covers all unique_tsubs
                valid = True
                for tsub in unique_tsubs:
                    if not set(tsub).issubset(set(new_block)):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count gain: how many t-subsets does this block cover that aren't already
                # covered by other blocks?
                gain = 0
                for tsub in itertools.combinations(new_block, t):
                    # This tsub is covered if it's in unique_tsubs (currently only by block i)
                    # or if it's covered by another block
                    if tsub in unique_tsubs:
                        gain += 1
                    else:
                        # Check if covered by other blocks
                        other_blocks = tsub_to_blocks.get(tsub, set())
                        if i not in other_blocks or len(other_blocks) > 1:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = new_block
            
            if best_new_block is not None and best_new_block != tuple(blocks[i]):
                # Replace block i with best_new_block
                old_block = blocks[i]
                blocks[i] = list(best_new_block)
                
                # Update tsub_to_blocks
                for tsub in itertools.combinations(old_block, t):
                    tsub_to_blocks[tsub].discard(i)
                for tsub in itertools.combinations(best_new_block, t):
                    tsub_to_blocks[tsub].add(i)
                
                # Update point_freq
                for p in old_block:
                    point_freq[p] -= 1
                for p in best_new_block:
                    point_freq[p] += 1
                
                improved = True
        
        if not improved:
            break
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Phase 1: Greedy construction using sparse matrix approach
    blocks = _greedy_construct(v, k, t)
    
    if _time_left() < 5.0:
        return blocks
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t)
    
    if _time_left() < 5.0:
        return blocks
    
    # Phase 3: Local search with block replacement
    blocks = _local_search_replace(blocks, v, k, t, max_iters=100)
    
    # Phase 4: Final redundant block removal
    blocks = _remove_redundant_blocks(blocks, v, k, t)
    
    return blocks
