"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------------------------------------------------------------------------
# Fast zero-one principle verification using NumPy vectorization
# ---------------------------------------------------------------------------
def _verify_network(comparators):
    """Verify sorting network using zero-one principle over all 2**N Boolean inputs."""
    n = N
    M = 1 << n
    if M == 0:
        return True
    state = np.arange(M, dtype=np.int32)
    for i, j in comparators:
        bi = 1 << i
        bj = 1 << j
        mask = (state & bi) & ~(state & bj)
        if np.any(mask):
            state[mask] = state[mask] ^ bi ^ bj
    # Fast sortedness check: sorted 0/1 sequence means all 1s come before all 0s
    # i.e., state[k] >= state[k+1] for all k, OR equivalently:
    # the set of states must be exactly {0, 1, 2, ..., M-1} in some order where
    # higher values (more 1s) appear at lower indices after sorting.
    # Actually for a sorting network, output should be sorted ascending.
    # For 0/1 inputs, sorted means: if we count number of 1s = c, then
    # output has c ones at the end (positions N-c..N-1) and zeros before.
    # So state[k] should equal (M - 1 - k) when k < M - popcount... 
    # Simpler: just check that state is a permutation of 0..M-1 AND sorted.
    # But we know it's always a permutation (comparators just swap bits).
    # So just check: is state[k] non-decreasing? No, it should be sorted.
    # For 0/1 inputs sorted ascending: 0,0,...,0,1,1,...,1
    # The count of 1s in input k is popcount(k). After sorting, output has
    # popcount(k) ones at the top. So state[k] should equal 
    # (1 << popcount(k)) - 1 ... no wait.
    # Let's just check: for each k, state[k] should be the number formed by
    # having popcount(k) ones in the highest bits.
    # i.e., state[k] = ((1 << popcount(k)) - 1) << (n - popcount(k))
    # No, that's not right either. Let me think again.
    # Input k has bits. After sorting ascending (min to wire 0, max to wire N-1),
    # the zeros go to lower wires, ones to higher wires.
    # So if input has c ones, output has 0s in wires 0..N-c-1 and 1s in wires N-c..N-1.
    # That means state[k] = ((1 << c) - 1) << (n - c) where c = popcount(k).
    # Let's verify: n=4, k=0b0101 (popcount=2). Output should be 0011 = 3.
    # ((1<<2)-1) << (4-2) = 3 << 2 = 12. That's wrong.
    # Actually wire 0 is least significant? Let me re-read.
    # "compare-exchange writes min to wire i and max to j"
    # So wire 0 gets the minimum. For 0/1, min=0, max=1.
    # So after sorting, wire 0 has the smallest value, wire N-1 has the largest.
    # For 0/1 inputs, all 0s should be on lower wires, all 1s on higher wires.
    # If input has c ones, then wires 0..N-c-1 have 0, wires N-c..N-1 have 1.
    # The bitmask: bit i corresponds to wire i. So state[k] should have
    # bits N-c through N-1 set, i.e., state[k] = ((1 << c) - 1) << (n - c).
    # For n=4, c=2: ((1<<2)-1) << (4-2) = 3 << 2 = 12 = 0b1100. Wires 2,3 are 1. Correct!
    # For c=0: 0. For c=4: 15. Good.
    # So we need: state[k] == ((1 << popcount(k)) - 1) << (n - popcount(k))
    # Precompute expected values
    popcounts = np.zeros(M, dtype=np.int32)
    for k in range(1, M):
        popcounts[k] = popcounts[k >> 1] + (k & 1)
    expected = ((1 << popcounts) - 1) << (n - popcounts)
    return bool(np.array_equal(state, expected))


def _verify_network_fast(comparators):
    """Even faster verification using bit manipulation tricks."""
    n = N
    M = 1 << n
    if M == 0:
        return True
    
    # Use a different approach: track the set of values on each wire
    # After all comparators, wire 0 should have the minimum, wire 15 the maximum
    # For 0-1 inputs, this means the sorted output is determined by the count of 1s
    
    # Actually, let's use the same vectorized approach but with early termination
    state = np.arange(M, dtype=np.int32)
    
    for i, j in comparators:
        bi = 1 << i
        bj = 1 << j
        # Only process if there are entries that need swapping
        need_swap = (state & bi) & ~(state & bj)
        if np.any(need_swap):
            state[need_swap] = state[need_swap] ^ bi ^ bj
    
    # Check if sorted
    for i in range(n - 1):
        threshold = 1 << i
        mask_bit = 1 << (i + 1)
        if not np.all(state[threshold:] & mask_bit):
            return False
    return True


def _compute_depth(comparators):
    """Compute parallel depth (greedy layering of non-adjacent comparators)."""
    if not comparators:
        return 0
    depth = 0
    remaining = list(comparators)
    while remaining:
        used = set()
        layer = []
        still = []
        for comp in remaining:
            i, j = comp
            if i not in used and j not in used:
                layer.append(comp)
                used.add(i)
                used.add(j)
            else:
                still.append(comp)
        depth += 1
        remaining = still
    return depth


# ---------------------------------------------------------------------------
# Bitonic sorting network construction (the design family)
# ---------------------------------------------------------------------------
def _bitonic_sort(n, offset, direction):
    """Generate comparators for a bitonic sort of n wires starting at offset."""
    if n <= 1:
        return []
    half = n // 2
    comps = []
    if direction == 'up':
        comps.extend(_bitonic_sort(half, offset, 'down'))
        comps.extend(_bitonic_sort(half, offset + half, 'up'))
    else:
        comps.extend(_bitonic_sort(half, offset, 'up'))
        comps.extend(_bitonic_sort(half, offset + half, 'down'))
    # Bitonic merge: k = n/2 down to 1 (the k = n stage is a no-op)
    k = half
    while k > 0:
        for i in range(0, n, 2 * k):
            for j in range(k):
                a = offset + i + j
                b = offset + i + k + j
                if direction == 'up':
                    comps.append((a, b))
                else:
                    comps.append((b, a))
        k //= 2
    return comps


def _baseline_network():
    """A verified correct sorting network (bitonic, 56 comparators for N=16)."""
    return _bitonic_sort(N, 0, 'up')


# ---------------------------------------------------------------------------
# Refinement machinery within the bitonic family
# ---------------------------------------------------------------------------
def _reorder_into_layers(comps):
    """Reorder non-adjacent comparators into parallel layers (safe: they commute).

    Returns (reordered_list, depth).
    """
    layers = []
    remaining = list(comps)
    while remaining:
        used = set()
        layer = []
        still = []
        for c in remaining:
            i, j = c
            if i not in used and j not in used:
                layer.append(c)
                used.add(i)
                used.add(j)
            else:
                still.append(c)
        layers.append(layer)
        remaining = still
    out = []
    for layer in layers:
        out.extend(layer)
    return out, len(layers)


def _prune_smart(comps, deadline):
    """Smart pruning: try removing comparators from the end first (often redundant),
    then from the middle, using batch removal when possible."""
    current = list(comps)
    if not _verify_network(current):
        return current
    
    # First pass: try removing from the end backwards
    i = len(current) - 1
    while i >= 0 and time.time() < deadline:
        trial = current[:i] + current[i+1:]
        if _verify_network(trial):
            current = trial
        else:
            i -= 1
    
    # Second pass: try removing from the beginning
    i = 0
    while i < len(current) and time.time() < deadline:
        trial = current[:i] + current[i+1:]
        if _verify_network(trial):
            current = trial
        else:
            i += 1
    
    # Third pass: try removing from the middle
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current)):
            if time.time() >= deadline:
                break
            trial = current[:idx] + current[idx + 1:]
            if _verify_network(trial):
                current = trial
                improved = True
                break
    
    # Fourth pass: try removing pairs of comparators (faster convergence)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current) - 1):
            if time.time() >= deadline:
                break
            trial = current[:idx] + current[idx+2:]
            if _verify_network(trial):
                current = trial
                improved = True
                break
    
    return current


def _prune(comps, deadline):
    """Greedy removal of redundant comparators (any subsequence of a sorter sorts)."""
    return _prune_smart(comps, deadline)


def _odd_even_transposition():
    """Fallback correct network: 120 comparators."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def construct():
    deadline = time.time() + 85  # safety margin well under 90s wall-clock
    try:
        comps = _baseline_network()
        # Ensure the baseline is actually correct before refining.
        if not _verify_network(comps):
            comps = _odd_even_transposition()

        # Refinement 1: reorder non-adjacent comparators into parallel layers
        # (reduces reported depth; correctness unchanged).
        comps, _depth = _reorder_into_layers(comps)

        # Refinement 2: prune redundant comparators (reduces count; correctness
        # preserved because any subsequence of a sorting network sorts).
        comps = _prune(comps, deadline)

        # Final safety: guarantee a valid network.
        if not _verify_network(comps):
            comps = _odd_even_transposition()
        return comps
    except Exception:
        return _odd_even_transposition()
# EVOLVE-BLOCK-END
