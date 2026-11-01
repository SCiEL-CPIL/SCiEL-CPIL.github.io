import time
import random
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Generate all t-subsets as bitmasks
    all_subsets = []
    for comb in combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    uncovered = set(all_subsets)
    
    # For fast gain computation, we maintain for each point x a dict:
    # rem_mask -> count of uncovered subsets S containing x where S without x = rem_mask
    # When we add x to block with mask B, gain = sum of counts for rem_mask that is subset of B
    # We'll use a different approach: for each x, maintain a list of (rem_mask, s_mask) pairs
    # and filter based on current block mask.
    
    # Actually, let's use a smarter approach:
    # For each point x, maintain a dict: rem_mask -> count
    # When checking gain for candidate x with block_mask B:
    #   gain = sum of counts[rm] for rm in counts if (rm & ~B) == 0
    
    # But iterating all rm keys is still expensive. Let's think differently.
    
    # Alternative: Precompute all (t-1)-subsets. For each point x, 
    # rem_mask is a (t-1)-subset not containing x.
    # Number of (t-1)-subsets = C(19,4) = 3876. For each x, at most 3876 entries.
    # Checking if rm is subset of B: (rm & ~B) == 0. This is one bitwise op.
    # So gain computation for one x: iterate up to 3876 entries, each one bitwise AND+compare.
    # 19 candidates * 3876 = ~73k ops per block position. 8 positions * 300 blocks = 17.5M ops.
    # This is much better than the original ~510M ops.
    
    # Initialize: for each x, count[rem_mask] = number of uncovered S containing x with S\x = rem_mask
    point_counts = [dict() for _ in range(v)]
    for s_mask in uncovered:
        # For each point in s_mask, compute rem_mask = s_mask without that point
        for x in range(v):
            if s_mask & (1 << x):
                rem = s_mask & ~(1 << x)
                point_counts[x][rem] = point_counts[x].get(rem, 0) + 1
    
    def decrement_point_counts(s_mask):
        """Remove one occurrence of s_mask from point_counts."""
        for x in range(v):
            if s_mask & (1 << x):
                rem = s_mask & ~(1 << x)
                cnt = point_counts[x].get(rem, 0)
                if cnt <= 1:
                    del point_counts[x][rem]
                else:
                    point_counts[x][rem] = cnt - 1
    
    blocks = []
    
    # Greedy Loop
    while uncovered:
        if time.time() > deadline:
            break
        
        # Choose the uncovered subset with the fewest "compatible" extensions
        # to start the block. Use scarcity: pick subset whose points have lowest total counts
        start_mask = None
        best_scarcity = float('inf')
        # Sample some uncovered subsets for speed
        sample = list(uncovered)
        random.shuffle(sample)
        for s_mask in sample[:min(50, len(sample))]:
            scarcity = sum(len(point_counts[x]) for x in range(v) if s_mask & (1 << x))
            if scarcity < best_scarcity:
                best_scarcity = scarcity
                start_mask = s_mask
        if start_mask is None:
            start_mask = min(uncovered)
        
        # Initial block contains the points of start_mask
        block_points = [i for i in range(v) if start_mask & (1 << i)]
        block_mask = start_mask
        
        # Extend the block
        while len(block_points) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_candidate = None
            best_candidates = []
            
            for x in range(v):
                if block_mask & (1 << x):
                    continue
                
                # Calculate gain: number of uncovered subsets S containing x
                # such that S\x is a subset of block_mask
                gain = 0
                counts_x = point_counts[x]
                for rem, cnt in counts_x.items():
                    if (rem & ~block_mask) == 0:
                        gain += cnt
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = x
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
                    
            if best_candidate is None:
                candidates = [x for x in range(v) if not (block_mask & (1 << x))]
                if not candidates:
                    break
                best_candidate = random.choice(candidates)
            else:
                best_candidate = random.choice(best_candidates)
                
            block_mask |= (1 << best_candidate)
            block_points.append(best_candidate)
        
        # Block is complete (size k)
        # Find all t-subsets of this block and mark as covered
        to_remove = []
        for s_mask in list(uncovered):
            if (s_mask & ~block_mask) == 0:
                to_remove.append(s_mask)
        
        for s_mask in to_remove:
            uncovered.discard(s_mask)
            decrement_point_counts(s_mask)
        
        blocks.append(sorted(block_points))
    
    # Local search: try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    if time.time() < deadline - 10:
        blocks = _local_search_reduce(blocks, v, k, t, deadline)
    
    return blocks

def _local_search_reduce(blocks, v, k, t, deadline):
    """Try to remove blocks that are fully covered by other blocks."""
    n = len(blocks)
    if n < 2:
        return blocks
    
    # Precompute: for each block, the set of t-subsets it covers
    block_subsets = []
    for b in blocks:
        bmask = 0
        for x in b:
            bmask |= (1 << x)
        subs = set()
        for comb in combinations(b, t):
            smask = 0
            for x in comb:
                smask |= (1 << x)
            subs.add(smask)
        block_subsets.append(subs)
    
    # Count how many times each t-subset is covered
    subset_count = {}
    for subs in block_subsets:
        for s in subs:
            subset_count[s] = subset_count.get(s, 0) + 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing each block if all its subsets are covered by others
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            redundant = True
            for s in block_subsets[i]:
                if subset_count[s] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in block_subsets[i]:
                    subset_count[s] -= 1
                del block_subsets[i]
                del blocks[i]
                changed = True
                break  # Restart from beginning
    
    return blocks

def _combinations(iterable, r):
    # Standard combinations generator
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i:] = range(indices[i] + 1, indices[i] + 1 + r - i)
        yield tuple(pool[i] for i in indices)
