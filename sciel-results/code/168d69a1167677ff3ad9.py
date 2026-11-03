"""Covering design C(19,7,4) using greedy set cover with randomized restarts."""
import random
import itertools
from collections import defaultdict


def _cover_greedy(v, k, t, randomize=False, seed=None):
    """
    Greedy set cover for covering all t-subsets with k-subsets.
    Each 'set' is a k-subset (block) which covers C(k,t) t-subsets.
    We want to cover all C(v,t) t-subsets with as few k-subsets as possible.
    
    If randomize is True, we add some randomness to tie-breaking to explore different solutions.
    """
    if seed is not None:
        random.seed(seed)
    
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to its index
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # For each point, precompute which t-subsets contain it
    # Actually, let's think differently. We need to efficiently find for a candidate block,
    # how many uncovered t-subsets it covers.
    
    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_t_subs))
    
    # For efficiency, maintain for each t-subset index, which points are in it
    t_sub_points = [set(ts) for ts in all_t_subs]
    
    blocks = []
    
    while uncovered:
        # Choose the best k-subset to add
        # We can't enumerate all C(19,7)=50388 subsets and check each one naively 
        # because that might be slow if we do it many times. But 50388 is manageable.
        
        # Actually, let's use a smarter approach:
        # Pick a random uncovered t-subset, then try to extend it to a k-subset
        # that covers the most uncovered t-subsets.
        
        # Strategy: 
        # 1. Pick a random uncovered t-subset (or the one that appears in fewest potential blocks)
        # 2. Extend it to a k-subset greedily, choosing points that maximize coverage
        
        # For randomization, pick a random uncovered t-subset as the seed
        if randomize:
            seed_t = random.choice(list(uncovered))
        else:
            seed_t = min(uncovered)
        
        seed_points = set(all_t_subs[seed_t])
        
        # Now extend seed_points to a k-subset
        block = list(seed_points)
        remaining_points = [x for x in range(v) if x not in seed_points]
        
        while len(block) < k:
            best_gain = -1
            best_point = None
            shuffled_remaining = remaining_points[:]
            if randomize:
                random.shuffle(shuffled_remaining)
            
            for x in shuffled_remaining:
                # Calculate how many uncovered t-subsets would be covered by adding x
                # A t-subset is covered by block+{x} if it's a subset of block+{x}
                # and not a subset of block
                # i.e., it contains x and is otherwise a subset of block
                # So it's a t-subset of (block + [x]) that includes x
                # That means it's a (t-1)-subset of block, plus x
                
                # Number of such t-subsets that are currently uncovered:
                # For each (t-1)-subset of block, check if (that subset + x) is in uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in t_sub_index:
                        idx = t_sub_index[ts]
                        if idx in uncovered:
                            gain += 1
                
                if gain > best_gain or (randomize and gain == best_gain and best_point is not None and random.random() < 0.1):
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                best_point = shuffled_remaining[0]
            
            block.append(best_point)
            remaining_points.remove(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove all covered t-subsets from uncovered
        for ts in itertools.combinations(block, t):
            idx = t_sub_index[ts]
            if idx in uncovered:
                uncovered.remove(idx)
    
    return [list(b) for b in blocks]


def construct(v, k, t):
    """
    Find a covering design with as few blocks as possible.
    Uses greedy set cover with multiple random restarts to find a good solution.
    """
    best_blocks = None
    best_len = float('inf')
    
    # First, try a deterministic greedy
    blocks = _cover_greedy(v, k, t, randomize=False)
    best_blocks = blocks
    best_len = len(blocks)
    
    # Then try several random restarts
    for seed in range(50):
        blocks = _cover_greedy(v, k, t, randomize=True, seed=seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return best_blocks
