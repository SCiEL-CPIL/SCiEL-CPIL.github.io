"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq
import random
from collections import defaultdict


def _elapsed(start):
    return time.time() - start


def _remaining(deadline, margin=10.0):
    return max(0.0, deadline - time.time() - margin)


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Coverage count for each t-subset
    cov = [0] * n_tsubs

    def _count_tsubs(block):
        """Count how many t-subsets are in this block."""
        return sum(1 for ts in itertools.combinations(block, t) if ts in tsub_to_idx)

    # Phase 1: Dynamic greedy with priority queue
    # For each point x, maintain a heap of (gain, x) where gain = number of uncovered
    # t-subsets that x would cover given the current partial block.
    
    # We need: for a partial block P (sorted), gain(x) = number of (t-1)-subsets S of P
    # such that S ∪ {x} is an uncovered t-subset.
    
    # Strategy: maintain for each point x a counter of how many uncovered t-subsets
    # it "participates in" with the current partial block.
    
    # Actually, let's use a simpler but effective approach:
    # For each iteration, compute gains for all candidate points using the current block.
    # Use numpy for speed if available, otherwise pure Python with optimizations.
    
    blocks = []
    
    # Track which t-subsets are still uncovered
    uncovered_set = set(range(n_tsubs))
    
    while uncovered_set and _remaining(deadline) > 1.0:
        # Pick the lexicographically smallest uncovered t-subset to anchor
        min_idx = min(uncovered_set)
        anchor = all_tsubs[min_idx]
        
        # Build block greedily
        block = sorted(anchor)
        block_set = set(block)
        
        while len(block) < k and _remaining(deadline) > 1.0:
            # Compute gain for each candidate point not in block
            best_x = None
            best_gain = -1
            
            # For efficiency, precompute which (t-1)-subsets of block are uncovered when extended
            # gain(x) = |{ S ⊂ block, |S|=t-1 : S ∪ {x} is uncovered }|
            
            t_minus_1_subs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for sub in t_minus_1_subs:
                    ts = tuple(sorted(sub + (x,)))
                    idx = tsub_to_idx.get(ts, -1)
                    if idx >= 0 and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                # Early termination: if we've found a point covering all remaining
                # (unlikely but possible)
                if gain == len(uncovered_set):
                    break
            
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        
        # Add block and update coverage
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None and idx in uncovered_set:
                uncovered_set.discard(idx)
                cov[idx] += 1
    
    # Phase 2: Deletion - try to remove blocks that don't uniquely cover any t-subset
    # Rebuild coverage counts from scratch for the current blocks
    cov = [0] * n_tsubs
    for blk in blocks:
        for ts in itertools.combinations(blk, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                cov[idx] += 1
    
    # A block can be removed if no t-subset it covers is covered only by it
    # i.e., for all t-subsets in the block, cov[ts] >= 2
    changed = True
    while changed and _remaining(deadline) > 1.0:
        changed = False
        new_blocks = []
        for blk in blocks:
            # Check if this block is redundant
            redundant = True
            for ts in itertools.combinations(blk, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None and cov[idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block and update coverage
                for ts in itertools.combinations(blk, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        cov[idx] -= 1
                changed = True
            else:
                new_blocks.append(blk)
        blocks = new_blocks
    
    return blocks
# EVOLVE-BLOCK-END
