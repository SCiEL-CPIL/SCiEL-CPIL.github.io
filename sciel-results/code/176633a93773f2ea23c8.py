"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
# Precompute initial wire masks for fast verification
# Wire i outputs 1 for input pattern k iff bit i of k is 1
# Represent each wire's "is-1 set" as a Python int (bitset of 65536 bits)

def _build_initial_masks():
    """Build initial bitset masks for 16 wires."""
    masks = []
    for i in range(16):
        # Bit i is 1 for k where (k >> i) & 1 == 1
        # Pattern: 2^i zeros, 2^i ones, repeating
        block = 1 << i
        # Build the mask using a loop (done once at startup)
        mask = 0
        # For efficiency, build using repetition
        # The pattern has period 2^(i+1), with the upper half being 1s
        period = 1 << (i + 1)
        ones_block = ((1 << block) - 1)  # block consecutive 1s
        # Repeat this pattern 65536 / period times
        num_periods = 65536 // period
        # Build the full mask
        m = 0
        for p in range(num_periods):
            # In each period, the ones are in positions [p*period + block, p*period + 2*block)
            offset = p * period + block
            m |= (ones_block << offset)
        masks.append(m)
    return masks

_INITIAL_MASKS = _build_initial_masks()

def _verify_network(comps):
    """Verify sorting network using zero-one principle with bitset representation."""
    if not comps:
        return True
    # Copy initial masks
    wires = list(_INITIAL_MASKS)
    for (i, j) in comps:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b  # min: 1 only if both are 1
        wires[j] = a | b  # max: 1 if either is 1
    
    # Check sorted: wire[i]_1set ⊆ wire[i+1]_1set for all i
    for i in range(15):
        if (wires[i] & ~wires[i+1]) != 0:
            return False
    return True

def _batcher_oe_merge_sort(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Recursively merge even-indexed and odd-indexed subsequences
        # Even indices: 0, 2, 4, ... -> map to 0, 1, 2, ...
        # Odd indices: 1, 3, 5, ... -> map to 0, 1, 2, ...
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        # Apply even merge on even positions
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        # Apply odd merge on odd positions
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            # Handle odd number: sort n-1, then insert the last element
            base = sort_network(n - 1)
            # Insert element n-1 into sorted network of n-1 elements
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        # Even: split into two halves
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        # Odd-even merge the two sorted halves
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _local_search(comps, max_iters=200000, deadline=None):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    best_count = len(best)
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    import random
    random.seed(42)
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random removal of a comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > 20000:
            break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    # Safety margin: assume we have ~85s left
    deadline = time.time() + 80.0
    best = _local_search(comps, max_iters=500000, deadline=deadline)
    
    return best
# EVOLVE-BLOCK-END
