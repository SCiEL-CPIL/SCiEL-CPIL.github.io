"""Covering design C(v,k,t) via randomized greedy construction + iterative reduction, multi-start."""
import itertools
import random
import time
from math import comb


def _all_tsubs(v, t):
    """Return list of all t-subsets as tuples (sorted order)."""
    return list(itertools.combinations(range(v), t))


def _tsub_index_map(tsubs):
    """Map each t-subset tuple to a dense integer index."""
    return {tsub: i for i, tsub in enumerate(tsubs)}, len(tsubs)


def _block_tsub_indices(block, t, idx_map):
    """Return list of indices of t-subsets contained in block."""
    return [idx_map[ts] for ts in itertools.combinations(block, t)]


def _greedy_construct(v, k, t, tsubs, idx_map, n_tsubs, rng, time_limit, start_time):
    """
    Single greedy construction pass.
    Returns list of blocks (each a sorted list of k points).
    """
    uncovered = set(range(n_tsubs))
    blocks = []
    blocks_tsubs = []  # parallel list: tsub indices covered by each block
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random uncovered t-subset to start the block
        seed_idx = rng.choice(list(uncovered))
        seed_tsub = tsubs[seed_idx]
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
                    try:
                        idx = idx_map[tsub]
                        if idx in uncovered:
                            gain += 1
                    except KeyError:
                        pass
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
        
        # Compute tsub indices for this block
        bts = _block_tsub_indices(block, t, idx_map)
        blocks_tsubs.append(bts)
        
        # Remove all t-subsets covered by this block
        uncovered -= set(bts)
    
    return blocks, blocks_tsubs


def _try_reduce(blocks, blocks_tsubs, tsubs, n_tsubs, rng, time_limit, start_time):
    """
    Iteratively remove redundant blocks.
    A block is redundant if all its tsub indices are covered by other blocks.
    Returns reduced (blocks, blocks_tsubs).
    """
    if not blocks:
        return blocks, blocks_tsubs
    
    # Build coverage count: for each tsub index, how many blocks cover it
    cover_count = [0] * n_tsubs
    for bts in blocks_tsubs:
        for idx in bts:
            cover_count[idx] += 1
    
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        
        changed = False
        # Try to remove each block (in random order for diversity)
        order = list(range(len(blocks)))
        rng.shuffle(order)
        
        for bi in order:
            if time.time() - start_time > time_limit:
                break
            
            bts = blocks_tsubs[bi]
            # Check if all tsubs in this block are covered by others (count > 1)
            redundant = all(cover_count[idx] > 1 for idx in bts)
            
            if redundant:
                # Remove this block
                for idx in bts:
                    cover_count[idx] -= 1
                blocks.pop(bi)
                blocks_tsubs.pop(bi)
                changed = True
                # Adjust indices in order
                for j in range(bi, len(order)):
                    order[j] -= 1
                break  # Restart after removal since indices shifted
    
    return blocks, blocks_tsubs


def _evaluate(blocks, v, k, t):
    """Check if blocks form a valid covering. Return number of covered tsubs."""
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return len(covered)


def construct(v, k, t):
    """
    Multi-start randomized greedy covering design constructor with reduction.
    Returns list of blocks; each block is a sorted list of k distinct ints in 0..v-1.
    """
    total_deadline = time.time() + 115  # safety margin before 120s kill
    
    # Precompute all tsubsets
    tsubs = _all_tsubs(v, t)
    idx_map, n_tsubs = _tsub_index_map(tsubs)
    
    best_blocks = None
    best_len = float('inf')
    
    # Strategy: multiple restarts with varying time budgets
    # Early restarts are quick to find a baseline, later ones try harder
    elapsed = 0
    restart_num = 0
    base_budget = 2.0
    
    while time.time() < total_deadline:
        restart_num += 1
        # Vary the random seed each restart
        rng = random.Random(restart_num * 12345 + 67890)
        
        # Allocate time: leave room for reduction
        # Use up to 70% of remaining time for construction, rest for reduction
        remaining = total_deadline - time.time()
        if remaining < 1.0:
            break
        
        # For early restarts, use shorter construction time
        if restart_num <= 3:
            cons_budget = min(base_budget, remaining * 0.5)
        else:
            cons_budget = min(remaining * 0.6, 8.0)
        
        start_time = time.time()
        blocks, blocks_tsubs = _greedy_construct(v, k, t, tsubs, idx_map, n_tsubs, rng, cons_budget, start_time)
        
        # Try to reduce
        if time.time() < total_deadline:
            red_budget = min(remaining * 0.3, 10.0)
            blocks, blocks_tsubs = _try_reduce(blocks, blocks_tsubs, tsubs, n_tsubs, rng, red_budget, time.time())
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we have a very good solution, we can try more aggressive reduction
        if best_len < 350 and time.time() < total_deadline - 5:
            # Do another round of reduction on the best solution
            rng2 = random.Random(restart_num * 99999 + 1)
            blocks2, blocks_tsubs2 = _try_reduce(list(best_blocks), [_block_tsub_indices(b, t, idx_map) for b in best_blocks], tsubs, n_tsubs, rng2, 5.0, time.time())
            if len(blocks2) < best_len:
                best_len = len(blocks2)
                best_blocks = blocks2
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        rng = random.Random(0)
        start_time = time.time()
        best_blocks, _ = _greedy_construct(v, k, t, tsubs, idx_map, n_tsubs, rng, 30, start_time)
    
    return best_blocks
