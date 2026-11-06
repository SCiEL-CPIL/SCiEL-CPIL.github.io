import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 80  # safety margin before 90s kill

def _count_comparators(net):
    return len(net)

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    # Each comparator at position i depends on the max of wires i and j from the previous stage
    # We process in topological order: a comparator (i,j) can be placed in layer = max(layer[i], layer[j]) + 1
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle: a network sorts all inputs iff it sorts all boolean inputs.
    For efficiency, we simulate all 2^N boolean inputs through the network.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    # Simulate all 2^N inputs
    for mask in range(1 << N):
        # Initialize wires
        wires = [(mask >> i) & 1 for i in range(N)]
        # Apply comparators
        for i, j in net:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if sorted: wires should be non-decreasing
        for k in range(N - 1):
            if wires[k] > wires[k + 1]:
                return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _remove_redundant_comparator(net, idx):
    """Try to remove comparator at index idx and check if network still sorts."""
    if idx < 0 or idx >= len(net):
        return None
    new_net = net[:idx] + net[idx+1:]
    if _is_sorted_network(new_net):
        return new_net
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Try to greedily remove comparators
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        # Try removing from the end first (often less critical)
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            new_net = _remove_redundant_comparator(net, idx)
            if new_net is not None:
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        # If we made a change, restart the loop to try more removals
    
    # If we still have time, try local search: swap or modify comparators
    if time.time() < deadline - 10:
        # Try replacing pairs of comparators with different ones
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            # Pick a random comparator to replace
            idx = random.randint(0, len(net) - 1)
            i, j = net[idx]
            # Try a random different comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            if (ni, nj) == (i, j):
                continue
            new_net = net[:]
            new_net[idx] = (ni, nj)
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
    
    return best_net
