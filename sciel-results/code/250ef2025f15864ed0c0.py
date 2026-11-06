"""Size-efficient sorting networks on 18 wires – search-based construction."""
import time
import numpy as np
import random
import heapq

N = 18
NUM_INPUTS = 1 << N
DEADLINE = 85.0  # seconds; safety margin before 90s kill
START = time.time()

def _remaining():
    return max(0.0, DEADLINE - (time.time() - START))

# ---------------------------------------------------------------------------
# Boolean simulation helpers
# ---------------------------------------------------------------------------

def _apply_comparator(wires, i, j):
    """In-place compare-exchange on a (NUM_INPUTS, N) int array."""
    a = wires[:, i]
    b = wires[:, j]
    # min -> wire i, max -> wire j
    wires[:, i] = np.minimum(a, b)
    wires[:, j] = np.maximum(a, b)

def _is_sorting(comparators, max_check=NUM_INPUTS):
    """Return True if the comparator list sorts every Boolean input."""
    if not comparators:
        return N <= 1
    # Build all 2^N inputs
    inputs = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for k in range(NUM_INPUTS):
        v = k
        for i in range(N):
            inputs[k, i] = (v >> i) & 1
            v >>= 1
    # Actually a faster way: use broadcasting
    idx = np.arange(NUM_INPUTS)
    for i in range(N):
        inputs[:, i] = ((idx >> i) & 1).astype(np.uint8)

    wires = inputs.copy()
    for (i, j) in comparators:
        _apply_comparator(wires, i, j)

    # Check: for each row, all 1s must be at the end (right side)
    # i.e., the row must be sorted in non-decreasing order.
    # Since values are 0/1, sorted means some 0s then some 1s.
    # Equivalently: for each row, the number of 1s in the suffix of length c equals c
    # where c is the total count of 1s.
    # Simpler: check that wire[i] <= wire[i+1] for all i, for all rows.
    for i in range(N - 1):
        if np.any(wires[:, i] > wires[:, i + 1]):
            return False
    return True

# ---------------------------------------------------------------------------
# Depth computation
# ---------------------------------------------------------------------------

def _depth(comparators):
    """Compute parallel depth (length of critical path)."""
    if not comparators:
        return 0
    # For each wire, track the latest step at which it was last compared
    last_step = [0] * N
    step = 0
    for (i, j) in comparators:
        step = max(last_step[i], last_step[j]) + 1
        last_step[i] = step
        last_step[j] = step
    return step

# ---------------------------------------------------------------------------
# Search: local search / greedy removal
# ---------------------------------------------------------------------------

def _greedy_removal(comparators, max_iters=200):
    """Try to remove comparators one at a time while maintaining correctness."""
    comps = list(comparators)
    improved = True
    iters = 0
    while improved and iters < max_iters and _remaining() > 5.0:
        improved = False
        iters += 1
        for idx in range(len(comps) - 1, -1, -1):
            if _remaining() < 5.0:
                break
            candidate = comps[:idx] + comps[idx + 1:]
            if _is_sorting(candidate):
                comps = candidate
                improved = True
                break
    return comps

def _local_search_replacement(comparators, max_iters=50):
    """Try replacing a comparator with a different pair to enable removal elsewhere."""
    comps = list(comparators)
    improved = True
    iters = 0
    while improved and iters < max_iters and _remaining() > 5.0:
        improved = False
        iters += 1
        for idx in range(len(comps)):
            if _remaining() < 5.0:
                break
            i, j = comps[idx]
            # Try all other pairs
            for ni in range(N):
                for nj in range(ni + 1, N):
                    if (ni, nj) == (i, j):
                        continue
                    candidate = comps[:idx] + [(ni, nj)] + comps[idx + 1:]
                    if _is_sorting(candidate):
                        # After replacement, try greedy removal
                        candidate = _greedy_removal(candidate, max_iters=10)
                        if len(candidate) < len(comps):
                            comps = candidate
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return comps

# ---------------------------------------------------------------------------
# Seed: odd-even transposition
# ---------------------------------------------------------------------------

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# ---------------------------------------------------------------------------
# Alternative seed: AKS-inspired / batcher odd-even merge sort network
# ---------------------------------------------------------------------------

def _batcher_oom(N):
    """Batcher's odd-even merge sort network."""
    if N <= 1:
        return []
    if N == 2:
        return [(0, 1)]

    # For powers of two, use recursive construction
    # For N=18, we'll build on N=16 and extend, or just use a known good network
    # Let's use a simpler approach: build for the next power of 2 and restrict
    # Actually, let's just use odd-even transposition as seed and search from there.
    return _odd_even_transposition()

# ---------------------------------------------------------------------------
# Heuristic: try to build a smaller network using known structures
# ---------------------------------------------------------------------------

def _known_small_network():
    """
    Return a known compact sorting network for 18 wires.
    We'll start with a structure inspired by Batcher and refine.
    """
    # For now, return odd-even transposition; search will improve it.
    return _odd_even_transposition()

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    """Return an ordered list of integer pairs (i, j) that sorts all 18-input sequences."""
    t_start = time.time()
    deadline = t_start + DEADLINE

    # Start with seed
    best = _odd_even_transposition()
    best_len = len(best)

    # Verify seed
    if not _is_sorting(best):
        # Fallback: should not happen, but just in case
        best = [(i, i + 1) for i in range(N - 1)] * N
        best_len = len(best)

    # Phase 1: Greedy removal
    best = _greedy_removal(best, max_iters=100)
    best_len = len(best)

    # Phase 2: Local search with replacements
    if _remaining() > 10.0:
        best = _local_search_replacement(best, max_iters=30)
        best_len = len(best)

    # Phase 3: More aggressive removal
    if _remaining() > 5.0:
        best = _greedy_removal(best, max_iters=200)
        best_len = len(best)

    # Final verification
    if not _is_sorting(best):
        # Return the seed if search broke correctness (shouldn't happen)
        best = _odd_even_transposition()

    return list(best)
