import time
import numpy as np
import random

N = 18
SEED = 153

def _verify(net):
    """Verify network sorts all 2^N Boolean inputs using 0-1 principle."""
    if not net:
        return N <= 1
    masks = np.arange(1 << N, dtype=np.uint32)
    for i, j in net:
        bi = N - 1 - i
        bj = N - 1 - j
        bit_i = (masks >> bi) & 1
        bit_j = (masks >> bj) & 1
        # swap where bit_i=1 and bit_j=0
        swap_mask = (bit_i & ~bit_j) * ((1 << bi) | (1 << bj))
        masks ^= swap_mask
    # Check sorted: all 0s then all 1s
    # popcount via byte lookup
    table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    pc = (table[masks & 0xFF] + table[(masks >> 8) & 0xFF] +
          table[(masks >> 16) & 0xFF] + table[(masks >> 24) & 0xFF])
    expected = (1 << pc) - 1
    return bool(np.all(masks == expected))

def _batcher_odd_merge(comps, lo, hi):
    """Batcher odd-even merge for ascending sort on wires lo..hi-1."""
    length = hi - lo
    if length <= 1:
        return
    half = length // 2
    _batcher_odd_merge(comps, lo, lo + half)
    _batcher_odd_merge(comps, lo + half, hi)
    # Interleave: compare lo+j with lo+j+half for j=0..half-1
    for j in range(half):
        a = lo + j
        b = lo + j + half
        if a < b:
            comps.append((a, b))
        else:
            comps.append((b, a))
    # Recursively merge the two interleaved halves
    # After interleave, we have two sequences to merge
    # Actually Batcher odd-even merge:
    # 1. Sort first half, sort second half
    # 2. Compare lo+j with lo+j+half
    # 3. Recursively merge the even and odd subsequences
    # Let me redo this properly.
    
    # Standard Batcher odd-even merge:
    # Sort A[0..n/2-1] and A[n/2..n-1]
    # Compare A[i] with A[i+n/2] for i=0..n/2-1
    # Then recursively merge A[0,2,4,...] and A[1,3,5,...]
    
    # This is getting complex. Let me use a simpler correct approach.
    # Actually, let me just use a known-good construction.
    pass

def _odd_even_merge(comps, lo, hi):
    """Odd-even merge sort (ascending)."""
    length = hi - lo
    if length <= 1:
        return
    half = length // 2
    _odd_even_merge(comps, lo, lo + half)
    _odd_even_merge(comps, lo + half, hi)
    # Interleave comparison
    for j in range(half):
        a = lo + j
        b = lo + j + half
        if a < b:
            comps.append((a, b))
        else:
            comps.append((b, a))
    # Recursively merge even and odd positions
    # Even positions: lo, lo+2, lo+4, ...
    # Odd positions: lo+1, lo+3, lo+5, ...
    # This requires a different recursive structure.
    # For simplicity, let me just do a simple merge-like comparison.
    # Actually, let me use a different strategy: just build a good network
    # using a known template.
    pass

def _odd_even_transposition(n):
    """Standard odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps

def _bitonic_sort(n):
    """Bitonic sort for power-of-2 sizes; pads if needed."""
    m = 1
    while m < n:
        m *= 2
    if m != n:
        return _odd_even_transposition(n)
    comps = []
    def build(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        build(lo, lo + half, 1)
        build(lo + half, hi, -1)
        for i in range(1, half + 1):
            for j in range(half):
                a = lo + j
                b = lo + j + i
                if direction == 1:
                    if a < b:
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
                else:
                    # For descending, we want max at smaller index
                    # But the comparator always puts min at first, max at second
                    # So to get descending, we compare (b, a) where b > a in index?
                    # No. The interface is: compare(i, j) with i<j puts min at i, max at j.
                    # For descending merge, we want the larger value at the smaller index.
                    # So we need to effectively do the opposite.
                    # In bitonic sort, the merge phase for descending uses
                    # comparators that put max at lower index.
                    # Since our comparator always puts min at lower index,
                    # we can't directly do descending with the same comparator.
                    # The standard bitonic sort uses comparators in both directions.
                    # Let me just do ascending for now and see if it works.
                    # Actually, bitonic sort requires bidirectional comparators.
                    # Let me skip this and use OET.
                    pass
    build(0, n, 1)
    return comps

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    # Start with OET (known correct, 153 comparators)
    net = _odd_even_transposition(N)
    
    # Verify initial network
    if not _verify(net):
        return net
    
    best_net = list(net)
    best_size = len(net)
    
    # Greedy deletion: try removing each comparator
    # Multiple passes, randomized order for diversity
    for pass_num in range(5):
        if time.time() > deadline - 15.0:
            break
        
        indices = list(range(len(net)))
        random.shuffle(indices)
        
        progress = False
        for idx in indices:
            if time.time() > deadline - 15.0:
                break
            if idx >= len(net):
                continue
            
            candidate = net[:idx] + net[idx+1:]
            if _verify(candidate):
                net = candidate
                progress = True
                if len(net) < best_size:
                    best_size = len(net)
                    best_net = list(net)
        
        if not progress:
            break
    
    # Local search: try replacing comparators
    # Simulated annealing style
    current_net = list(net)
    current_size = len(current_net)
    T = 1.0
    cooling = 0.995
    
    for step in range(2000):
        if time.time() > deadline - 3.0:
            break
        
        # Random move: either delete, add, or replace a comparator
        move_type = random.random()
        
        if move_type < 0.3 and len(current_net) > 10:
            # Delete a random comparator
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            delta = -1
        elif move_type < 0.5:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = current_net + [(i, j)]
            delta = 1
        else:
            # Replace a random comparator
            idx = random.randint(0, len(current_net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = current_net[:idx] + [(i, j)] + current_net[idx+1:]
            delta = 0
        
        if not _verify(candidate):
            # Reject invalid moves
            T *= cooling
            continue
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_net = candidate
            current_size = len(current_net)
            if current_size < best_size:
                best_size = current_size
                best_net = list(current_net)
        
        T *= cooling
    
    return best_net
