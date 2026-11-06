import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _count_comparators(net):
    return len(net)

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle: a network sorts all inputs iff it sorts all boolean inputs.
    Vectorized: processes all 2^N inputs simultaneously.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Generate all 2^N boolean inputs as a matrix: shape (2^N, N)
    # wires[k] = (mask >> k) & 1
    all_masks = np.arange(1 << N, dtype=np.int32)
    # Create binary representation: each column is one wire
    # wires[:, i] = (all_masks >> i) & 1
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    # Apply comparators
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        # min goes to i, max goes to j
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    
    # Check if sorted: wires should be non-decreasing along wire dimension
    # For each row, check if wires[k] <= wires[k+1] for all k
    for k in range(N - 1):
        if np.any(wires[:, k] > wires[:, k + 1]):
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

def _try_remove_multiple(net, indices_to_try):
    """
    Try removing a set of comparators simultaneously.
    Returns the reduced network if it still sorts, else None.
    """
    indices_set = set(indices_to_try)
    new_net = [c for idx, c in enumerate(net) if idx not in indices_set]
    if len(new_net) == len(net):
        return None
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
    
    # Phase 1: Greedy removal - try to remove one comparator at a time
    # Try from end to start, but also try strategic positions
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        # Try removing from various positions
        # First pass: from end
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
        # Second pass: from start (different order may find different redundancies)
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 5:
                    break
                new_net = _remove_redundant_comparator(net, idx)
                if new_net is not None:
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Try removing pairs of comparators simultaneously
    if time.time() < deadline - 10:
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            # Pick two random indices to try removing
            if len(net) < 2:
                break
            idx1, idx2 = random.sample(range(len(net)), 2)
            new_net = _try_remove_multiple(net, [idx1, idx2])
            if new_net is not None:
                net = new_net
                best_net = net[:]
                best_count = len(net)
                # Reset and continue trying
                if time.time() > deadline - 5:
                    break
    
    # Phase 3: Local search - replace comparators
    if time.time() < deadline - 10:
        # Try replacing individual comparators with better ones
        for attempt in range(500):
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
    
    # Phase 4: Try adding then removing - sometimes adding a comparator allows removing two
    if time.time() < deadline - 15:
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            # Add a random comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            new_net = net[:]
            new_net.append((ni, nj))
            
            # Now try to remove two comparators from the extended network
            if len(new_net) >= 2:
                idx1, idx2 = random.sample(range(len(new_net)), 2)
                reduced = _try_remove_multiple(new_net, [idx1, idx2])
                if reduced is not None and len(reduced) < len(net):
                    net = reduced
                    best_net = net[:]
                    best_count = len(net)
    
    return best_net
