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
    
    # For sorted 0/1 output: wire 0 has min, wire N-1 has max.
    # If input has c ones, output has 0s in wires 0..N-c-1, 1s in wires N-c..N-1.
    # state[k] should have bits N-c..N-1 set where c = popcount(k).
    # Precompute expected values using vectorized popcount
    k = np.arange(M, dtype=np.int32)
    # Vectorized popcount
    popcounts = np.zeros(M, dtype=np.int32)
    for shift in range(n):
        popcounts += (k >> shift) & 1
    expected = ((1 << popcounts) - 1) << (n - popcounts)
    return bool(np.array_equal(state, expected))


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
# Bitonic sorting network construction
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
    # Bitonic merge: k = n/2 down to 1
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
    """A correct sorting network (bitonic)."""
    return _bitonic_sort(N, 0, 'up')


# ---------------------------------------------------------------------------
# Refinement machinery
# ---------------------------------------------------------------------------
def _reorder_into_layers(comps):
    """Reorder non-adjacent comparators into parallel layers (safe: they commute)."""
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
    """Smart pruning: iterative removal of redundant comparators."""
    current = list(comps)
    if not _verify_network(current):
        return current
    
    # Repeated passes: try removing each comparator
    for _ in range(5):  # max 5 full passes
        if time.time() >= deadline:
            break
        improved = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() >= deadline:
                break
            trial = current[:idx] + current[idx + 1:]
            if _verify_network(trial):
                current = trial
                improved = True
        if not improved:
            break
    
    return current


def _odd_even_transposition():
    """Fallback correct network: 120 comparators."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def construct():
    deadline = time.time() + 85
    try:
        comps = _baseline_network()
        if not _verify_network(comps):
            comps = _odd_even_transposition()

        # Reorder into layers for better depth
        comps, _ = _reorder_into_layers(comps)

        # Prune redundant comparators
        comps = _prune_smart(comps, deadline)

        # Final safety check
        if not _verify_network(comps):
            comps = _odd_even_transposition()
        return comps
    except Exception:
        return _odd_even_transposition()
# EVOLVE-BLOCK-END
