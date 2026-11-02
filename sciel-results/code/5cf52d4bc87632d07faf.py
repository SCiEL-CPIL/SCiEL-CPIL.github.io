"""Covering design C(19,8,5) - improved greedy with local search."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict

# Precomputation for v=19, k=8, t=5
_V = 19
_K = 8
_T = 5

def _precompute():
    """Precompute t-subset indices and block-to-tsubsets mapping."""
    all_tsubs = list(itertools.combinations(range(_V), _T))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each possible block (8-subset), precompute which t-subsets it covers
    # This is expensive but done once. Use a dict: block_tuple -> set of tsub indices
    block_tsubs = {}
    for block in itertools.combinations(range(_V), _K):
        covered = set()
        for tsub in itertools.combinations(block, _T):
            covered.add(tsub_to_idx[tsub])
        block_tsubs[block] = covered
    
    return all_tsubs, tsub_to_idx, n_tsubs, block_tsubs

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    # Precompute for the specific v,k,t
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each k-block, compute the set of t-subset indices it covers
    # Use a list of frozensets or just lists for speed
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_tsubs = {}
    for block in all_blocks:
        covered = set()
        for tsub in itertools.combinations(block, t):
            covered.add(tsub_to_idx[tsub])
        block_to_tsubs[block] = covered
    
    # Count how many t-subsets each block covers (for heuristic)
    block_sizes = {b: len(ts) for b, ts in block_to_tsubs.items()}
    
    def _greedy_cover(uncovered_mask, blocks_remaining, deadline):
        """Greedy set cover: pick block covering most uncovered t-subsets."""
        blocks = []
        # uncovered_mask is a set of t-subset indices
        uncovered = set(uncovered_mask)
        
        # Pre-sort blocks by coverage size (descending) for initial ordering
        # But we'll re-evaluate each step
        
        while uncovered and time.time() < deadline:
            best_block = None
            best_gain = -1
            best_covered = None
            
            # Sample a subset of blocks if too many
            candidates = list(blocks_remaining)
            if len(candidates) > 2000:
                random.shuffle(candidates)
                candidates = candidates[:2000]
            
            for b in candidates:
                if time.time() > deadline:
                    break
                covered = block_to_tsubs[b]
                gain = len(covered & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    best_covered = covered
                    if gain == len(uncovered):
                        break
            
            if best_block is None or best_gain <= 0:
                break
            
            blocks.append(list(best_block))
            uncovered -= best_covered
            blocks_remaining.discard(best_block)
        
        return blocks, uncovered
    
    def _greedy_with_block_building(uncovered, deadline):
        """Build blocks point-by-point to maximize coverage."""
        blocks = []
        uncovered_set = set(uncovered)
        
        while uncovered_set and time.time() < deadline:
            # Start with the first uncovered t-subset
            first = min(uncovered_set)
            ts = all_tsubs[first]
            block = list(ts[:k-1])  # start with k-1 points from first t-subset
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                block_set = set(block)
                
                # For each candidate point not in block
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered
                    # A t-subset {a1,...,a5} is covered by block if all 5 in block
                    # Adding x to block: count t-subsets where x is in tsub, 
                    # and tsub \\ {x} subset of block
                    gain = 0
                    for tsub in uncovered_set:
                        if x in tsub:
                            rest = tuple(sorted(set(tsub) - {x}))
                            if all(p in block_set for p in rest):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            # Remove covered t-subsets
            for tsub in itertools.combinations(block, t):
                uncovered_set.discard(tsub_to_idx[tsub])
        
        return blocks, uncovered_set
    
    def _local_search(blocks, deadline):
        """Try to remove or replace blocks to reduce count."""
        blocks = [list(b) for b in blocks]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each block
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Check if block i is redundant
                remaining = blocks[:i] + blocks[i+1:]
                # Check if all t-subsets are still covered
                covered = set()
                for b in remaining:
                    for tsub in itertools.combinations(b, t):
                        covered.add(tsub_to_idx[tsub])
                if len(covered) == n_tsubs:
                    blocks.pop(i)
                    improved = True
                    break
                # Try replacing block i with a better one
                # Find t-subsets only covered by block i
                only_i = set()
                for tsub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[tsub]
                    # Check if covered by any other block
                    for j, b in enumerate(remaining):
                        if j >= i:
                            j -= 1
                        if tsub in set(itertools.combinations(b, t)):
                            break
                    else:
                        only_i.add(idx)
                
                if not only_i:
                    continue
                
                # Try to find a replacement block covering only_i
                best_rep = None
                best_extra = -1
                # Sample some blocks
                for b in random.sample(all_blocks, min(500, len(all_blocks))):
                    covered_b = block_to_tsubs[b]
                    covers_only = covered_b & only_i
                    if len(covers_only) == len(only_i):
                        # This block covers all of only_i
                        extra = len(covered_b & set(tsub_to_idx[ts] for ts in all_tsubs)) 
                        if extra > best_extra:
                            best_extra = extra
                            best_rep = b
                
                if best_rep is not None:
                    blocks[i] = list(best_rep)
                    improved = True
                    break
        
        return blocks
    
    # Strategy 1: Pure greedy set cover
    best_blocks = None
    best_count = float('inf')
    
    # Multiple random restarts of greedy
    random.seed(42)
    for attempt in range(5):
        if time.time() > DEADLINE - 30:
            break
        
        uncovered = set(range(n_tsubs))
        blocks_remaining = set(all_blocks)
        
        # Shuffle for diversity
        if attempt > 0:
            # Use a different greedy strategy
            pass
        
        blocks, remaining_uncovered = _greedy_cover(uncovered, blocks_remaining, DEADLINE - 10)
        
        if not remaining_uncovered:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
    
    # Strategy 2: Block-building greedy
    if time.time() < DEADLINE - 20:
        uncovered = set(range(n_tsubs))
        blocks, remaining = _greedy_with_block_building(uncovered, DEADLINE - 10)
        if not remaining and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on best solution
    if best_blocks is not None and time.time() < DEADLINE - 5:
        best_blocks = _local_search(best_blocks, DEADLINE - 2)
    
    if best_blocks is None:
        # Fallback: return greedy result even if incomplete (shouldn't happen)
        uncovered = set(range(n_tsubs))
        blocks, _ = _greedy_cover(uncovered, set(all_blocks), DEADLINE)
        best_blocks = blocks
    
    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(key))
    
    return unique_blocks

# Precompute at module level for speed on repeated calls
try:
    _all_tsubs, _tsub_to_idx, _n_tsubs, _block_to_tsubs = _precompute()
except:
    pass
