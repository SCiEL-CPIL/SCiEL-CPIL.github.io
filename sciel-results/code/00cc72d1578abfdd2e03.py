"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 18
NUM_BOOLS = 1 << N  # 262144
NUM_WORDS = NUM_BOOLS // 64  # 4096

# Precompute initial masks efficiently using pattern generation
def _make_initial_masks(n):
    """Create initial bitmask array for n wires.
    masks[i, w] has bit b set iff input k = 64*w + b has wire i = 1,
    i.e., (k >> i) & 1 == 1."""
    masks = np.zeros((n, NUM_WORDS), dtype=np.uint64)
    for i in range(n):
        period = 1 << (i + 1)
        half = 1 << i
        for w in range(NUM_WORDS):
            base = 64 * w
            # Determine pattern offset
            # Within each period of 2^(i+1), the lower half has bit 0, upper half has bit 1
            # We need to figure out where base falls in the period
            # For efficiency, compute the pattern for 64 consecutive k values
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            masks[i, w] = word
    return masks

# Precompute once
_INITIAL_MASKS = _make_initial_masks(N)

def _verify_fast(comps, n=N):
    """Fast verification using precomputed masks. Returns True if correct."""
    # Copy initial masks
    masks = _INITIAL_MASKS.copy()
    
    # Apply comparators
    for (a, b) in comps:
        ma = masks[a]
        mb = masks[b]
        masks[a] = ma & mb
        masks[b] = ma | mb
    
    # Check: after sorting, wire i should have exactly {k : popcount(k) >= n - i}
    # Check monotonicity: mask[i] ⊆ mask[i+1]
    for i in range(n - 1):
        if np.any(masks[i] & ~masks[i + 1]):
            return False
    
    # Check wire 0: exactly 1 bit set (the all-ones input)
    bit_count_0 = int(np.sum([bin(w).count('1') for w in masks[0]]))
    if bit_count_0 != 1:
        return False
    
    # Check wire n-1: all bits except bit 0 (all-zeros input)
    bit_count_last = int(np.sum([bin(w).count('1') for w in masks[n - 1]]))
    if bit_count_last != NUM_BOOLS - 1:
        return False
    
    return True

def _build_ooe_network(n):
    """Odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = N
    comps = _build_ooe_network(n)
    
    # Verify initial network
    if not _verify_fast(comps, n):
        return comps
    
    best = list(comps)
    best_len = len(best)
    
    # Phase 1: Greedy removal - repeatedly try to remove each comparator
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, n):
                comps = trial
                improved = True
                best = list(comps)
                best_len = len(comps)
                break  # restart scan after removal
    
    # Phase 2: Try all possible single-comparator replacements
    # For each position, try replacing with a different comparator
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for _round in range(20):
        if time.time() > deadline:
            break
        found = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            current = comps[idx]
            for new_pair in all_pairs:
                if new_pair == current:
                    continue
                trial = comps[:idx] + [new_pair] + comps[idx+1:]
                if _verify_fast(trial, n):
                    # Now try to remove something
                    for j in range(len(trial)):
                        if time.time() > deadline:
                            break
                        trial2 = trial[:j] + trial[j+1:]
                        if _verify_fast(trial2, n):
                            comps = trial2
                            best = list(comps)
                            best_len = len(comps)
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        if not found:
            break
    
    # Phase 3: Continue greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, n):
                comps = trial
                improved = True
                best = list(comps)
                best_len = len(comps)
                break
    
    return best
# EVOLVE-BLOCK-END
