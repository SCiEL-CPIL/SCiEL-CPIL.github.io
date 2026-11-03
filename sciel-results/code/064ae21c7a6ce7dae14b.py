"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

# Precompute all t-subsets as bitmasks for fast set operations
_ALL_T_SUBSETS = None
_V = None
_K = None
_T = None


def _init_masks(v, k, t):
    global _ALL_T_SUBSETS, _V, _K, _T
    if _V == v and _K == k and _T == t and _ALL_T_SUBSETS is not None:
        return
    _V, _K, _T = v, k, t
    _ALL_T_SUBSETS = set()
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        _ALL_T_SUBSETS.add(mask)


def _mask_of_block(block):
    m = 0
    for x in block:
        m |= (1 << x)
    return m


def _t_subsets_of_block(block, t):
    """Yield bitmask for each t-subset of block."""
    for combo in itertools.combinations(block, t):
        m = 0
        for x in combo:
            m |= (1 << x)
        yield m


def _count_uncovered_covered(block, uncovered_set):
    """Count how many t-subsets of block are in uncovered_set."""
    cnt = 0
    for m in _t_subsets_of_block(block, _T):
        if m in uncovered_set:
            cnt += 1
    return cnt


def _greedy_construct(v, k, t, uncovered, rng, time_limit):
    """Greedy: repeatedly pick a seed point, then greedily add k-1 more points
    maximizing newly-covered t-subsets. Returns list of blocks."""
    blocks = []
    start = time.time()
    while uncovered:
        if time.time() - start > time_limit:
            break
        # Pick the uncovered t-subset with the highest 'pressure' (randomized)
        # We just pick a random one for diversity
        seed_subset = next(iter(uncovered))
        # Convert bitmask to list
        block = []
        for i in range(v):
            if seed_subset & (1 << i):
                block.append(i)
        
        # Greedily add points
        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets that would be covered
                # by adding x to block
                gain = 0
                new_block = block + [x]
                for m in _t_subsets_of_block(new_block, t):
                    if m in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove covered t-subsets
        for m in _t_subsets_of_block(block, t):
            uncovered.discard(m)
    
    return blocks


def _swap_local_search(blocks, v, k, t, uncovered, rng, time_limit):
    """Try 1-for-1 swaps: remove one point from a block, add another, if it
    doesn't uncover any previously-covered t-subset and covers new ones."""
    start = time.time()
    improved = True
    while improved:
        if time.time() - start > time_limit:
            break
        improved = False
        # Build coverage count: for each t-subset, how many blocks cover it
        coverage = {}
        for block in blocks:
            for m in _t_subsets_of_block(block, t):
                coverage[m] = coverage.get(m, 0) + 1
        
        for bi, block in enumerate(blocks):
            if time.time() - start > time_limit:
                break
            block_set = set(block)
            for old_x in block:
                new_block = [x for x in block if x != old_x]
                # Try adding each candidate
                best_gain = 0
                best_new_x = -1
                for new_x in range(v):
                    if new_x in block_set:
                        continue
                    trial_block = new_block + [new_x]
                    # Check: does removing old_x uncover any t-subset?
                    # A t-subset is uncovered if it was only covered by this block
                    # and contains old_x
                    ok = True
                    gain = 0
                    for m in _t_subsets_of_block(trial_block, t):
                        if m in coverage:
                            if coverage[m] == 1 and (1 << old_x) & m:
                                # This subset would be uncovered
                                ok = False
                                break
                        else:
                            gain += 1
                    if not ok:
                        continue
                    if gain > best_gain:
                        best_gain = gain
                        best_new_x = new_x
                if best_new_x != -1 and best_gain > 0:
                    # Perform swap
                    blocks[bi] = sorted(new_block + [best_new_x])
                    # Update coverage
                    for m in _t_subsets_of_block(block, t):
                        coverage[m] -= 1
                        if coverage[m] == 0:
                            del coverage[m]
                    for m in _t_subsets_of_block(blocks[bi], t):
                        coverage[m] = coverage.get(m, 0) + 1
                    improved = True
    
    return blocks


def _merge_duplicate_blocks(blocks):
    """Remove duplicate blocks."""
    seen = set()
    unique = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def construct(v, k, t):
    global _ALL_T_SUBSETS, _V, _K, _T
    _init_masks(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave safety margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple random restarts
    rng = random.Random(42)
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Need time for local search
            break
        
        # Initialize uncovered
        uncovered = set(_ALL_T_SUBSETS)
        
        # Greedy construction
        blocks = _greedy_construct(v, k, t, uncovered, rng, 20)
        
        if not uncovered:
            # Full cover achieved
            blocks = _merge_duplicate_blocks(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        # Local search: try to reduce block count via swaps
        if time.time() < deadline - 5:
            # Rebuild uncovered for local search
            covered = set()
            for block in blocks:
                for m in _t_subsets_of_block(block, t):
                    covered.add(m)
            uncovered_ls = _ALL_T_SUBSETS - covered
            
            blocks = _swap_local_search(blocks, v, k, t, uncovered_ls, rng, 
                                        min(15, max(1, deadline - time.time() - 5)))
            blocks = _merge_duplicate_blocks(blocks)
            
            # Verify
            covered = set()
            for block in blocks:
                for m in _t_subsets_of_block(block, t):
                    covered.add(m)
            if covered == _ALL_T_SUBSETS:
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
    
    # If we didn't find a valid cover, fall back to a simple greedy
    if best_blocks is None:
        uncovered = set(_ALL_T_SUBSETS)
        blocks = _greedy_construct(v, k, t, uncovered, rng, 60)
        if uncovered:
            # Emergency: keep adding blocks until covered
            while uncovered:
                seed = next(iter(uncovered))
                block = [i for i in range(v) if seed & (1 << i)]
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    block_set = set(block)
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = sum(1 for m in _t_subsets_of_block(block + [x], t) if m in uncovered)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    block.append(best_x)
                block.sort()
                blocks.append(block)
                for m in _t_subsets_of_block(block, t):
                    uncovered.discard(m)
        best_blocks = _merge_duplicate_blocks(blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
