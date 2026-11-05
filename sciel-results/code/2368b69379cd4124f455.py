import time
import random

def _incremental_valid(values, sums, x):
    """Check if x can be appended to values. Returns (valid, new_sums_to_add).
    values is a list, sums is a set of all existing consecutive block sums.
    We only need to check suffix sums ending at the new position.
    """
    tails = []
    total = x
    tails.append(total)
    for a in reversed(values):
        total += a
        tails.append(total)
    # Check collision
    for t in tails:
        if t in sums:
            return False, None
    return True, tails

def _greedy_build_fast(order):
    """Build a valid sequence using incremental validity check."""
    values = []
    sums = set()
    for x in order:
        valid, new_sums = _incremental_valid(values, sums, x)
        if valid:
            values.append(x)
            sums.update(new_sums)
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    best_seq = []

    # 1. Deterministic Baselines
    # Ascending
    seq = _greedy_build_fast(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq

    # Descending
    seq = _greedy_build_fast(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq

    # 2. Random Restarts
    rng = random.Random(42)
    base_range = list(range(1, 257))

    while time.time() < deadline:
        rng.shuffle(base_range)
        seq = _greedy_build_fast(base_range)
        if len(seq) > len(best_seq):
            best_seq = seq

    return best_seq
