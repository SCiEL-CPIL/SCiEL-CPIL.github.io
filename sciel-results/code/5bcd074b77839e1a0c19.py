"""Simulated annealing for covering design C(24,7,3) with numpy-accelerated coverage tracking."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0

    # Precompute all t-subsets as indices for numpy tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(s): i for i, s in enumerate(all_tsubs)}
    
    # Precompute for each point pair, which t-subsets they participate in (for k=7,t=3)
    # For t=3: a t-subset {a,b,c} is covered by block B iff {a,b,c} ⊆ B
    
    # Greedy initial cover
    blocks = _greedy_cover(v, k, t, rng)
    
    # Build coverage count array
    def build_counts(block_list):
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for b in block_list:
            for c in itertools.combinations(b, t):
                idx = tsub_to_idx[frozenset(c)]
                counts[idx] += 1
        return counts
    
    counts = build_counts(blocks)
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    best_uncovered = np.sum(counts == 0)
    
    # SA parameters
    T_init = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    def get_uncovered():
        return int(np.sum(counts == 0))
    
    def get_single():
        return int(np.sum(counts == 1))
    
    uncovered = get_uncovered()
    single = get_single()
    
    iteration = 0
    T = T_init
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Try removals first (free improvement)
        removed = False
        if len(blocks) > 0:
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for i in indices:
                b = blocks[i]
                redundant = True
                for fs in itertools.combinations(b, t):
                    idx = tsub_to_idx[frozenset(fs)]
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for fs in itertools.combinations(b, t):
                        idx = tsub_to_idx[frozenset(fs)]
                        old_c = counts[idx]
                        counts[idx] = old_c - 1
                        if old_c == 1:
                            single -= 1
                        if old_c - 1 == 0:
                            uncovered += 1
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    removed = True
                    break
        
        if removed:
            T = max(T * cooling_rate, T_min)
            continue
        
        # If no removal possible, try swaps
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Find critical subsets (count == 1) in old_block
            critical = []
            for fs in itertools.combinations(old_block, t):
                idx = tsub_to_idx[frozenset(fs)]
                if counts[idx] == 1:
                    critical.append(fs)
            
            if not critical:
                # Shouldn't happen if removal failed, but skip
                T = max(T * cooling_rate, T_min)
                continue
            
            points_needed = set()
            for fs in critical:
                points_needed.update(fs)
            
            if len(points_needed) > k:
                T = max(T * cooling_rate, T_min)
                continue
            
            # Build new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            for x in cands:
                if len(new_block_set) >= k:
                    break
                # Gain: how many subsets with count 0 or 1 would be covered
                gain = 0
                for rest in itertools.combinations(new_block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = tsub_to_idx[fs]
                    c = counts[idx]
                    if c == 0:
                        gain += 10
                    elif c == 1:
                        gain += 1
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)
            
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                if new_block == old_block:
                    T = max(T * cooling_rate, T_min)
                    continue
                
                # Simulate swap
                # Subsets affected: those in old_block or new_block
                affected = set()
                for fs in itertools.combinations(old_block, t):
                    affected.add(frozenset(fs))
                for fs in itertools.combinations(new_block, t):
                    affected.add(frozenset(fs))
                
                # Save old counts
                old_state = {}
                for fs in affected:
                    old_state[fs] = counts[tsub_to_idx[fs]]
                
                # Apply removal of old_block
                for fs in itertools.combinations(old_block, t):
                    idx = tsub_to_idx[frozenset(fs)]
                    counts[idx] -= 1
                
                # Apply addition of new_block
                for fs in itertools.combinations(new_block, t):
                    idx = tsub_to_idx[frozenset(fs)]
                    counts[idx] += 1
                
                # Check validity: no count should be 0
                new_uncovered = 0
                new_single = 0
                for fs in affected:
                    idx = tsub_to_idx[fs]
                    c = counts[idx]
                    if c == 0:
                        new_uncovered += 1
                    elif c == 1:
                        new_single += 1
                
                # Compare with current
                delta_uncovered = new_uncovered - uncovered
                delta_single = new_single - single
                
                # Energy: prefer fewer uncovered, then fewer single
                # For SA, we accept if it doesn't increase uncovered, or with probability
                if delta_uncovered <= 0 and delta_single <= 0:
                    # Accept (improvement or neutral)
                    blocks[i] = new_block
                    uncovered = new_uncovered
                    single = new_single
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                else:
                    # SA: accept with probability exp(-delta_E / T)
                    # Energy delta: weight uncovered heavily
                    delta_E = delta_uncovered * 100 + delta_single
                    if delta_E > 0 and rng.random() < np.exp(-delta_E / max(T, 1e-10)):
                        blocks[i] = new_block
                        uncovered = new_uncovered
                        single = new_single
                    else:
                        # Revert
                        for fs in affected:
                            idx = tsub_to_idx[fs]
                            counts[idx] = old_state[fs]
        
        # Perturbation every 200 iterations
        if iteration % 200 == 0:
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = build_counts(blocks)
                uncovered = get_uncovered()
                single = get_single()
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
        
        T = max(T * cooling_rate, T_min)
    
    return best_blocks
# EVOLVE-BLOCK-END
