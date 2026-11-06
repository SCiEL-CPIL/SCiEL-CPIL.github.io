"""
Size-efficient sorting network on 17 wires.
Strategy: Start with odd-even transposition (known valid), then aggressively
remove redundant comparators using fast numpy-based simulation.
"""
import time
import random
import numpy as np

N = 17
ZERO_ONE_COUNT = 1 << N  # 131072
DEADLINE = 85.0


def _build_transposition():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _simulate_fast(comparators, inputs):
    """
    Fast simulation. inputs: (M, N) uint8 array of 0/1.
    Returns bool array (M,) - is output sorted?
    """
    data = inputs.copy()
    for (i, j) in comparators:
        a = data[:, i]
        b = data[:, j]
        mn = np.minimum(a, b)
        mx = a + b - mn
        data[:, i] = mn
        data[:, j] = mx
    diffs = data[:, 1:] - data[:, :-1]
    return np.all(diffs >= 0, axis=1)


def _validate_full(comparators):
    """Full zero-one validation over all 2^17 inputs."""
    all_inputs = np.zeros((ZERO_ONE_COUNT, N), dtype=np.uint8)
    for k in range(N):
        all_inputs[:, k] = (np.arange(ZERO_ONE_COUNT) >> k) & 1
    return _simulate_fast(comparators, all_inputs).all()


def construct():
    start_time = time.time()
    
    # Initial valid network
    comparators = _build_transposition()
    best = list(comparators)
    best_count = len(best)
    
    # Verify
    if not _validate_full(best):
        # Should not happen
        return best
    
    rng = random.Random(42)
    
    # Greedy removal: repeatedly try to remove a random comparator
    # and check if the network still sorts all inputs.
    # To speed up, first try a smaller sample, then full check.
    
    def is_valid(comps, full=True):
        if full:
            return _validate_full(comps)
        else:
            # Quick check with random samples
            rng2 = np.random.default_rng(0)
            samples = rng2.integers(0, 2, size=(2000, N), dtype=np.uint8)
            return _simulate_fast(comps, samples).all()
    
    # Main loop: try removals
    while time.time() < DEADLINE - 5:
        if len(comparators) <= 2:
            break
        idx = rng.randrange(len(comparators))
        candidate = comparators[:idx] + comparators[idx + 1:]
        
        # Quick check first
        if is_valid(candidate, full=False):
            # Full check
            if _validate_full(candidate):
                comparators = candidate
                if len(comparators) < best_count:
                    best = list(comparators)
                    best_count = len(best)
    
    # Final safety check
    if time.time() < DEADLINE - 2:
        if not _validate_full(best):
            best = _build_transposition()
    
    return best
