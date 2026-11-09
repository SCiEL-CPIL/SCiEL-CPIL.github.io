"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 16

# EVOLVE-BLOCK-START
# Pre-compute all binary inputs once for fast verification
_all_inputs = None

def _get_all_inputs():
    """Get or create the pre-computed all binary inputs array."""
    global _all_inputs
    if _all_inputs is None:
        _all_inputs = np.zeros((65536, 16), dtype=np.int8)
        idx = np.arange(65536, dtype=np.int32)
        for i in range(16):
            _all_inputs[:, i] = (idx >> i) & 1
    return _all_inputs

def _apply_comparators(x, comps):
    """Apply a list of comparators to array x, returning modified copy."""
    x = x.copy()
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    return x

def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    if not comps:
        return True
    x = _apply_comparators(_get_all_inputs(), comps)
    for k in range(15):
        if not np.all(x[:, k] <= x[:, k + 1]):
            return False
    return True

def _batcher_oe_merge_sort(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        comps = [(i, n + i) for i in range(n)]
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
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
            base = sort_network(n - 1)
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _compute_wire_states(comps):
    """Compute the state of each wire after each comparator step.
    Returns a list where states[k] is the 65536x16 array after k comparators."""
    x = _get_all_inputs().copy()
    states = [x]
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
        states.append(x)
    return states

def _is_comparator_redundant(comps, idx, states):
    """Check if comparator at index idx is redundant by checking if removing it
    still produces a sorted output. Uses precomputed states for efficiency."""
    if idx == 0:
        # Removing first comparator: apply all except first
        trial = _apply_comparators(_get_all_inputs(), comps[1:])
    elif idx == len(comps) - 1:
        # Removing last comparator: state before last comparator
        trial = states[idx]
    else:
        # Need to recompute from scratch for middle removals
        trial = _apply_comparators(_get_all_inputs(), comps[:idx] + comps[idx+1:])
    
    for k in range(15):
        if not np.all(trial[:, k] <= trial[:, k + 1]):
            return False
    return True

def _redundancy_analysis(comps, deadline=None):
    """Systematically find and remove redundant comparators."""
    best = list(comps)
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        states = _compute_wire_states(best)
        # Try removing from end first (most likely to be redundant)
        for idx in range(len(best) - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            if _is_comparator_redundant(best, idx, states):
                best = best[:idx] + best[idx+1:]
                changed = True
                break  # Restart scan since states changed
    return best

def _local_search(comps, max_iters=50000, deadline=None):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    
    # Phase 1: Systematic redundancy removal
    best = _redundancy_analysis(best, deadline)
    
    # Phase 2: Random removals for further optimization
    random.seed(42)
    iters = 0
    no_improve = 0
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > 3000:
            break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Safety margin: use 80 seconds out of 90 second limit
    deadline = time.time() + 80.0
    best = _local_search(comps, max_iters=200000, deadline=deadline)
    
    return best
# EVOLVE-BLOCK-END
