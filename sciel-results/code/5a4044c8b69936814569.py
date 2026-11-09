"""Covering design C(v,k,t) via multi-start randomized greedy with time guard."""
import itertools
import random
import time
from math import comb


def _all_tsubs(v, t):
    """Generate all t-subsets as frozensets."""
    return set(itertools.combinations(range(v), t))


def _greedy_construct(v, k, t, rng, time_limit):
    """
    Single greedy construction pass.
    Returns list of blocks (each a sorted list of k points).
    """
    start_time = time.time()
    all_tsubs = _all_tsubs(v, t)
    uncovered = set(all_tsubs)
    blocks = []
    
    # Precompute for speed: for a given partial block, what t-subsets would be covered
    # by adding point x? We need C(partial, t-1) combos, and check if (combo + x) is uncovered.
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random uncovered t-subset to start the block
        seed_tsub = rng.choice(list(uncovered))
        block = list(seed_tsub)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() - start_time > time_limit:
                break
            
            best_x = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in block_set]
            
            # Shuffle candidates for tie-breaking diversity
            rng.shuffle(candidates)
            
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point helps, pick random
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks


def _is_complete(blocks, v, k, t):
    """Check if the blocks form a valid covering."""
    covered = set()
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            covered.add(tsub)
    return len(covered) == comb(v, t)


def construct(v, k, t):
    """
    Multi-start randomized greedy covering design constructor.
    Returns list of blocks; each block is a sorted list of k distinct ints in 0..v-1.
    """
    total_deadline = time.time() + 110  # safety margin before 120s kill
    rng = random.Random(42)
    
    # For small instances, do a few quick runs
    # For the target (19,8,5), do randomized restarts with time budget each
    
    best_blocks = None
    best_len = float('inf')
    
    # Estimate how many restarts we can afford
    # Each greedy pass for C(19,8,5) takes roughly 0.5-2 seconds
    # We have ~110s, so aim for ~20-40 restarts
    
    elapsed = 0
    restart_num = 0
    per_restart_budget = 2.5  # seconds per restart attempt
    
    while time.time() < total_deadline:
        restart_num += 1
        # Vary the random seed each restart
        rng = random.Random(restart_num * 12345 + 67890)
        
        blocks = _greedy_construct(v, k, t, rng, per_restart_budget)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've found a very good solution, we could try to improve further
        # but for now just keep restarting
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        rng = random.Random(0)
        best_blocks = _greedy_construct(v, k, t, rng, 30)
    
    return best_blocks
