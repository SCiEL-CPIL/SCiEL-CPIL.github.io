"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _try_order(order, limit=256):
    """Try to build a sequence by considering candidates in given order.
    Returns the longest valid strictly-increasing sequence found."""
    values = []
    sums = set()
    for x in order:
        if values and x <= values[-1]:
            continue
        # Compute all new consecutive-block sums ending at x
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if sums.intersection(tails):
            continue
        values.append(x)
        sums.update(tails)
    return values

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best = []
    
    # Strategy 1: ascending greedy (baseline)
    asc = list(range(1, 257))
    result = _try_order(asc)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: descending greedy
    desc = list(range(256, 0, -1))
    result = _try_order(desc)
    if len(result) > len(best):
        best = result
    
    # Strategy 3: random restarts with different orderings
    # Use a seeded RNG for reproducibility within a run
    rng = random.Random(42)
    candidates = list(range(1, 257))
    
    # Try shuffled orders
    while time.time() < deadline:
        order = candidates[:]
        rng.shuffle(order)
        result = _try_order(order)
        if len(result) > len(best):
            best = result
            # If we've found a very long sequence, we can be more conservative
            if len(best) >= 20:
                break
    
    # Strategy 4: biased random orders (favor small numbers early, or large)
    while time.time() < deadline and len(best) < 20:
        # Create a biased order: mix of small and large
        order = []
        remaining = set(range(1, 257))
        while remaining:
            # Pick with bias toward extremes
            if rng.random() < 0.5:
                x = min(remaining)
            else:
                x = max(remaining)
            order.append(x)
            remaining.discard(x)
        result = _try_order(order)
        if len(result) > len(best):
            best = result
    
    return best
# EVOLVE-BLOCK-END
