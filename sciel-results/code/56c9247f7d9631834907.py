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

def _local_search_aggressive(comps, deadline=None):
    """Aggressive local search: try to remove as many comparators as possible.
    Uses multiple random restarts within the time budget."""
    import random
    best = list(comps)
    best_count = len(best)
    
    start_time = time.time()
    
    # Multiple restarts with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        if deadline is not None and time.time() > deadline:
            break
        
        random.seed(seed)
        current = list(best)
        no_improve = 0
        max_no_improve = 5000
        
        while no_improve < max_no_improve:
            if deadline is not None and time.time() > deadline:
                break
            
            if len(current) <= 1:
                break
            
            # Try random removal
            idx = random.randint(0, len(current) - 1)
            trial = current[:idx] + current[idx+1:]
            
            if _verify_network(trial):
                current = trial
                no_improve = 0
                # Update best if improved
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(best)
            else:
                no_improve += 1
        
        # If we've exhausted this seed, try another
        if len(best) <= 60:
            break
    
    return best

def _local_search_targeted(comps, deadline=None):
    """Targeted local search: focus on removing specific problematic comparators.
    Tries to identify which comparators are most likely to be redundant."""
    import random
    best = list(comps)
    best_count = len(best)
    
    start_time = time.time()
    random.seed(2024)
    
    # First, try to identify and remove clearly redundant comparators
    # A comparator is redundant if removing it doesn't break sorting
    # We'll use a heuristic: try removing comparators that appear later in the network
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        # Try removing from the end first (often more redundant)
        for idx in range(len(best) - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                improved = True
                break
        
        # If no improvement from end, try random positions
        if not improved and len(best) > 1:
            for _ in range(1000):
                if deadline is not None and time.time() > deadline:
                    break
                
                idx = random.randint(0, len(best) - 1)
                trial = best[:idx] + best[idx+1:]
                if _verify_network(trial):
                    best = trial
                    best_count = len(best)
                    improved = True
                    break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Two-phase local search
    # Safety margin: assume we have ~85s left, use 75s for search
    deadline = time.time() + 75.0
    
    # Phase 1: Aggressive removal with multiple restarts
    best = _local_search_aggressive(comps, deadline=deadline)
    
    # Phase 2: Targeted refinement if time remains
    if time.time() < deadline - 5:
        best = _local_search_targeted(best, deadline=deadline)
    
    return best
# EVOLVE-BLOCK-END
