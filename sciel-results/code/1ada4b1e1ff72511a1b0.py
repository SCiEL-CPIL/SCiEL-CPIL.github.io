"""Covering design C(v,k,t) via randomized greedy with multiple strategies and deadline guard."""
import itertools
import random
import time
from collections import defaultdict


def _all_tsubsets(v, t):
    """Generate all t-subsets of range(v) as tuples."""
    return list(itertools.combinations(range(v), t))


def _greedy_block(uncovered, block, k, v, t, all_tsets, rng=None):
    """
    Greedily extend a partial block to size k.
    uncovered: set of tuples (t-subsets not yet covered)
    block: list of points already in the block
    Returns the completed block as a sorted list.
    """
    while len(block) < k:
        best_gain = -1
        best_point = -1
        for x in range(v):
            if x in block:
                continue
            # Count how many uncovered t-subsets would be covered by adding x
            gain = 0
            # A t-subset containing x is covered if the other t-1 elements are in block
            # So we need: for each (t-1)-subset S of block, check if S ∪ {x} is in uncovered
            for rest in itertools.combinations(block, t - 1):
                cand = tuple(sorted(rest + (x,)))
                if cand in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_point = x
        if best_point == -1:
            # Fallback: pick any point not in block
            for x in range(v):
                if x not in block:
                    best_point = x
                    break
        block.append(best_point)
    return sorted(block)


def _greedy_cover(v, k, t, seed=0, strategy="first_uncovered", deadline=None):
    """
    Greedy covering algorithm.
    strategy: 'first_uncovered' - start with the lexicographically first uncovered t-subset
              'random' - start with a random uncovered t-subset
              'min_degree' - start with an uncovered t-subset whose points have fewest other uncovered t-subsets
    """
    rng = random.Random(seed)
    all_tsubs = _all_tsubsets(v, t)
    uncovered = set(all_tsubs)
    blocks = []
    
    # Precompute for each point, the set of t-subsets containing it (for potential optimization)
    # But for now, use direct membership check
    
    while uncovered:
        if deadline is not None and time.time() > deadline:
            return None  # Signal timeout
        
        if strategy == "first_uncovered":
            # Pick the lexicographically smallest uncovered t-subset
            start_tuple = min(uncovered)
        elif strategy == "random":
            start_tuple = rng.choice(list(uncovered))
        elif strategy == "min_degree":
            # For each uncovered t-subset, compute the "degree" = number of uncovered t-subsets
            # that share at least t-1 elements with it (i.e., are "close" to it)
            # Pick the one with minimum degree (most constrained)
            best_sub = None
            best_deg = float('inf')
            # Sample a few if too many
            sample_size = min(len(uncovered), 50)
            sample = rng.sample(list(uncovered), sample_size)
            for sub in sample:
                deg = 0
                # Count uncovered t-subsets that differ from sub in exactly 1 element
                for i in range(t):
                    for x in range(v):
                        if x in sub:
                            continue
                        new_sub = tuple(sorted(sub[:i] + (x,) + sub[i+1:]))
                        if new_sub in uncovered:
                            deg += 1
                if deg < best_deg:
                    best_deg = deg
                    best_sub = sub
            start_tuple = best_sub
        else:
            start_tuple = min(uncovered)
        
        block = list(start_tuple)
        block = _greedy_block(uncovered, block, k, v, t, all_tsubs, rng)
        blocks.append(block)
        
        # Remove all t-subsets contained in this block
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return blocks


def _local_search_improve(blocks, v, k, t, deadline=None, max_iters=100):
    """
    Try to improve the covering by removing redundant blocks.
    A block is redundant if removing it still leaves all t-subsets covered.
    """
    all_tsubs = _all_tsubsets(v, t)
    n = len(blocks)
    
    # Count how many times each t-subset is covered
    cover_count = defaultdict(int)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if cover_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for ts in itertools.combinations(blocks[i], t):
                    cover_count[ts] -= 1
                blocks.pop(i)
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s kill
    
    # Try multiple strategies and keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # Strategy 1: Deterministic greedy (first uncovered)
    blocks = _greedy_cover(v, k, t, seed=0, strategy="first_uncovered", deadline=deadline)
    if blocks is not None:
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Strategy 2: Randomized greedy with different seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        blocks = _greedy_cover(v, k, t, seed=seed, strategy="random", deadline=deadline)
        if blocks is not None:
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
    
    # Strategy 3: Min-degree heuristic
    for seed in [0, 1, 2]:
        if time.time() > deadline:
            break
        blocks = _greedy_cover(v, k, t, seed=seed, strategy="min_degree", deadline=deadline)
        if blocks is not None:
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
    
    # Try local search improvement on the best result
    if best_blocks is not None and time.time() < deadline:
        improved = _local_search_improve(best_blocks, v, k, t, deadline=deadline)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(best_blocks)
    
    if best_blocks is None:
        # Fallback: should not happen, but just in case
        # Use the original simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    for x in range(v):
                        if x not in block:
                            best_point = x
                            break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    return best_blocks
