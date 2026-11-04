"""Covering design C(v,k,t) via greedy set-cover with efficient bitset tracking."""
import itertools
import random
import sys

def construct(v, k, t):
    # For small parameters, we can use a smarter approach.
    # Strategy: Greedy set cover where each candidate block is evaluated by
    # how many currently uncovered t-subsets it covers.
    # To be faster, we use a bitset-based greedy which is significantly faster.
    
    if v <= 20 and k <= 8:
        # Use bitset greedy
        return _greedy_cover_bitset(v, k, t, seed=42)
    else:
        # Fallback to basic greedy for larger instances
        return _basic_greedy(v, k, t)


def _basic_greedy(v, k, t):
    """Original greedy: pick uncovered t-subset, extend to k-subset maximizing coverage."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick the first uncovered t-subset as seed
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets of this block from uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks


def _greedy_cover_bitset(v, k, t, seed=42):
    """
    Greedy set cover using bitsets for speed.
    1. Enumerate all t-subsets and assign an index to each.
    2. Enumerate all k-subsets.
    3. For each k-subset, compute a bitmask of the t-subsets it covers.
    4. Greedily select the k-subset that covers the most *currently uncovered* t-subsets.
    """
    # Step 1: Map t-subsets to indices
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # Step 2 & 3: Precompute masks for all k-subsets
    all_ksubsets = list(itertools.combinations(range(v), k))
    masks = []
    for ksub in all_ksubsets:
        mask = 0
        for tsub in itertools.combinations(ksub, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        masks.append(mask)
    
    # Step 4: Greedy selection
    # Initial uncovered mask: all bits set
    # Python ints have arbitrary precision, so we can set n_tsubs bits
    uncovered_mask = (1 << n_tsubs) - 1
    blocks = []
    
    # To speed up, we can sort candidates initially or just scan all each time.
    # Scanning 50k candidates * 150 iterations = 7.5M operations.
    # Each operation is a bitwise AND and bit_count on ~4000-bit integers.
    # This should be well within 120s.
    
    while uncovered_mask:
        best_idx = -1
        best_count = 0
        
        for i, mask in enumerate(masks):
            # Count how many bits are set in (mask & uncovered_mask)
            # This represents the number of new t-subsets covered
            new_covers = mask & uncovered_mask
            count = new_covers.bit_count()
            
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1 or best_count == 0:
            break
            
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered_mask &= ~masks[best_idx]
        
    return blocks


def _greedy_cover_optimized(v, k, t, num_restarts=3, seed=42):
    """
    Optimized greedy with numpy for speed.
    """
    import numpy as np
    
    rng = random.Random(seed)
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    n_ksubs = len(all_ksubsets)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # Build incidence matrix: n_ksubs x n_tsubs
    # incidence[i, j] = 1 if k-subset i contains t-subset j
    incidence = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for i, ksub in enumerate(all_ksubsets):
        for tsub in itertools.combinations(ksub, t):
            j = tsub_to_idx[tsub]
            incidence[i, j] = 1
    
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks = []
    
    while uncovered.any():
        # For each k-subset, count how many uncovered t-subsets it covers
        covered_counts = incidence @ uncovered
        best_idx = int(np.argmax(covered_counts))
        if covered_counts[best_idx] == 0:
            break
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered[incidence[best_idx] == 1] = 0
    
    return blocks
