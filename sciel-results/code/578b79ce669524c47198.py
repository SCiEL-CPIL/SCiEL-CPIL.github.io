"""Size-efficient sorting networks on 17 wires – search-based construction."""

import time
import random
import numpy as np

N = 17

# Deadline guard: kill search well before 90s wall-clock limit
_START_TIME = time.time()
_DEADLINE = 75.0  # seconds; leaves 15s safety margin


def _now():
    return time.time()


def _time_exceeded():
    return (_now() - _START_TIME) >= _DEADLINE


def _verify_booleans_numpy(network):
    """
    Vectorized verification using NumPy.
    Checks all 2^N Boolean inputs.
    """
    total = 1 << N
    # Create indices 0 to total-1
    indices = np.arange(total, dtype=np.uint64)
    
    # Initialize wires: shape (total, N)
    # wire[k] for input m is bit k of m
    wires = np.zeros((total, N), dtype=np.uint8)
    for k in range(N):
        wires[:, k] = (indices >> k) & 1
        
    # Apply comparators
    for (i, j) in network:
        a = wires[:, i]
        b = wires[:, j]
        # min and max
        min_ab = np.minimum(a, b)
        max_ab = np.maximum(a, b)
        wires[:, i] = min_ab
        wires[:, j] = max_ab
        
    # Check sorted
    # For each row, check if wire[k] <= wire[k+1] for all k
    # This is equivalent to checking if any wire[k] > wire[k+1]
    # diff = np.diff(wires, axis=1)
    # return not np.any(diff < 0)
    
    # More efficient: check if all adjacent pairs are sorted
    # wires[:, :-1] <= wires[:, 1:]
    return np.all(wires[:, :-1] <= wires[:, 1:])


def _odd_even_transposition(n=N):
    """Standard odd-even transposition sorting network."""
    comp = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comp.append((i, i + 1))
    return comp


def construct():
    """
    Construct a sorting network on N=17 wires.
    Uses search: try known constructions, then random local search within time budget.
    Returns ordered list of (i, j) pairs.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with odd-even transposition (136 comparators for 17 wires)
    best = _odd_even_transposition(N)
    
    # Verify base (should be true)
    if not _verify_booleans_numpy(best):
        return best
        
    # Greedy removal loop
    improved = True
    while improved and not _time_exceeded():
        improved = False
        # Try removing each comparator
        # We iterate through indices. If we remove one, we restart the loop
        # because indices shift.
        for idx in range(len(best)):
            if _time_exceeded():
                return best
                
            candidate = best[:idx] + best[idx+1:]
            
            # Verify
            if _verify_booleans_numpy(candidate):
                best = candidate
                improved = True
                break # Restart loop with new length
                
        # If no improvement in a full pass, we might be stuck.
        # We could try random swaps or different orderings, but for now,
        # simple greedy removal is the primary driver.
        # To avoid infinite loops on "stuck" states where no single removal works
        # but a combination does, we rely on the fact that sorting networks
        # often have many redundant comparators.
        
    return best
