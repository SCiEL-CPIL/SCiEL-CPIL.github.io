"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys


def _count_covers(block, t):
    """Count how many t-subsets are covered by a block."""
    return len(list(itertools.combinations(block, t)))


def _greedy_cover(v, k, t, start_subset, deadline, rng):
    """
    Greedy covering algorithm.
    - start_subset: the initial t-subset to cover
    - deadline: absolute time (seconds) by which we must stop
    - rng: random.Random instance for tie-breaking
    Returns (blocks, uncovered_set) where uncovered_set is the set of t-subsets not yet covered.
    """
    # Precompute all t-subsets and map to indices for fast lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    
    # uncovered: set of indices of t-subsets not yet covered
    uncovered = set(range(n_tsubsets))
    blocks = []
    
    # Precompute for each point x, the list of t-subset indices that contain x
    # This is a big speedup: instead of checking each t-subset, we just look up
    point_to_subsets = [[] for _ in range(v)]
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_to_subsets[x].append(idx)
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to start the block
        # Using min for determinism, but we can randomize for diversity
        start_idx = rng.choice(list(uncovered))
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many uncovered t-subsets would be covered if we add x
                # A t-subset is covered if it contains all t elements of the block plus x
                # Actually, we need t-subsets that contain (block + x) - but block has len(block) elements
                # A t-subset S is covered by block+x if S is a subset of block+x and |S|=t
                # So S must contain t elements from block+x, and S must be in uncovered
                # More precisely: S is covered if S ⊆ block ∪ {x} and S is a t-subset
                # The t-subsets covered by adding x are those S where S ⊆ block ∪ {x}, |S|=t, and S is uncovered
                # These are t-subsets that contain x and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in sub_to_idx:
                        idx = sub_to_idx[sub]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            if not best_points:
                break
            
            # Pick randomly among best points for diversity
            block.append(rng.choice(best_points))
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets from uncovered
        for sub in itertools.combinations(block, t):
            if sub in sub_to_idx:
                uncovered.discard(sub_to_idx[sub])
    
    return blocks, uncovered


def _remove_redundant_blocks(blocks, v, t, deadline):
    """
    Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
    Returns a new list of blocks with redundant ones removed.
    """
    if not blocks:
        return blocks
    
    # Build coverage map: for each t-subset, which blocks cover it
    sub_to_idx = {}
    idx = 0
    for sub in itertools.combinations(range(v), t):
        sub_to_idx[sub] = idx
        idx += 1
    
    n_tsubsets = len(sub_to_idx)
    # coverage[i] = set of block indices that cover t-subset i
    coverage = [set() for _ in range(n_tsubsets)]
    
    for bidx, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            if sub in sub_to_idx:
                coverage[sub_to_idx[sub]].add(bidx)
    
    # A block is redundant if for every t-subset it covers, there's another block covering it
    non_redundant = []
    for bidx, block in enumerate(blocks):
        if time.time() > deadline:
            break
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub in sub_to_idx:
                idx = sub_to_idx[sub]
                if len(coverage[idx]) == 1:
                    # Only this block covers this t-subset
                    redundant = False
                    break
        if not redundant:
            non_redundant.append(block)
    
    return non_redundant


def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin: stop at 110s to leave buffer for cleanup
    deadline = start_time + 110
    
    # Try multiple random restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    
    # First, do a deterministic greedy run
    rng = random.Random(42)
    blocks, uncovered = _greedy_cover(v, k, t, None, deadline, rng)
    if len(uncovered) == 0:
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try a few more random restarts if time permits
    if time.time() < deadline - 20:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            blocks, uncovered = _greedy_cover(v, k, t, None, deadline, rng)
            if len(uncovered) == 0 and len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
    
    # Remove redundant blocks
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _remove_redundant_blocks(best_blocks, v, t, deadline)
    
    if best_blocks is None:
        # Fallback: return an empty list or a trivial solution
        return []
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
