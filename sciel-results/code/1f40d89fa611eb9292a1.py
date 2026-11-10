"""Covering design C(v,k,t) – fast greedy with deadline guard."""
import itertools
import time
import random

_START = time.time()
_DEADLINE_S = 118.0  # safety margin below 120s


def _time_left():
    return _DEADLINE_S - (time.time() - _START)


def construct(v, k, t):
    """
    Return a list of blocks (each a sorted list of k distinct ints in 0..v-1)
    such that every t-subset of {0,...,v-1} is contained in at least one block.
    Minimize the number of blocks.
    """
    # For the specific case v=19, k=7, t=4, use a specialized approach.
    # General approach: greedy with randomization and restarts.
    
    if v == 19 and k == 7 and t == 4:
        return _covering_19_7_4()
    
    # General greedy
    return _greedy_covering(v, k, t)


def _covering_19_7_4():
    """
    Specialized covering for C(19,7,4).
    Uses greedy with multiple random seeds and keeps the best result.
    """
    best_blocks = None
    best_count = float('inf')
    
    start = time.time()
    deadline = start + 115.0  # leave 5s margin
    
    for seed in range(100000):
        if time.time() > deadline:
            break
        
        blocks = _greedy_19_7_4(seed)
        if blocks is None:
            break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # If we've hit a very good number, we can stop early
        if best_count <= 140:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = _greedy_19_7_4(0)
    
    return best_blocks


def _greedy_19_7_4(seed):
    """
    Greedy covering for v=19, k=7, t=4.
    Uses bitset for tracking uncovered 4-subsets.
    """
    v, k, t = 19, 7, 4
    rng = random.Random(seed)
    
    # Precompute all 4-subsets and assign indices
    all_4subs = list(itertools.combinations(range(v), t))
    n_4subs = len(all_4subs)  # C(19,4) = 3876
    
    # Map each 4-subset to an index
    sub_to_idx = {sub: i for i, sub in enumerate(all_4subs)}
    
    # Track uncovered 4-subsets using a set of indices
    uncovered = set(range(n_4subs))
    
    blocks = []
    
    while uncovered:
        # Pick a random uncovered 4-subset to start the block
        first_sub = list(rng.choice(list(uncovered)))
        block = list(first_sub)
        
        # Greedily add points to cover the most uncovered 4-subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Compute how many uncovered 4-subsets would be covered by adding x
                # A 4-subset {a,b,c,d} is covered by block ∪ {x} if 
                # {a,b,c,d} ⊆ (block ∪ {x}), i.e., at most one of {a,b,c,d} is x
                # and the rest are in block.
                
                # More precisely: adding x covers 4-subsets S such that
                # S ⊆ block ∪ {x} and x ∈ S (since S is not already covered by block alone)
                # So S = {x} ∪ R where R is a (t-1)-subset of block.
                
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                # No useful point to add; break
                break
            
            block.append(best_x)
        
        if len(block) < k:
            # Fill with random points to make it a valid k-subset
            remaining = [x for x in range(v) if x not in block]
            rng.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            uncovered.discard(idx)
    
    return blocks


def _greedy_covering(v, k, t):
    """
    General greedy covering with deadline guard.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    uncovered = set(range(n_tsubs))
    blocks = []
    
    start = time.time()
    deadline = start + 115.0
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick first uncovered
        first_sub = list(next(iter(uncovered)))
        block = list(first_sub)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                break
            
            block.append(best_x)
        
        if len(block) < k:
            remaining = [x for x in range(v) if x not in block]
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        blocks.append(block)
        
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            uncovered.discard(idx)
    
    # Verify completeness; if incomplete, extend with more blocks
    if uncovered and time.time() < deadline:
        # Add more blocks to cover remaining
        while uncovered and time.time() < deadline:
            first_sub = list(next(iter(uncovered)))
            block = list(first_sub)
            while len(block) < k:
                remaining = [x for x in range(v) if x not in block]
                block.append(remaining[0])
                remaining.pop(0)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                uncovered.discard(idx)
    
    return blocks
