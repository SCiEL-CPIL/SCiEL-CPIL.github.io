"""Size-efficient sorting networks on 17 wires using bitmask-based greedy deletion."""
import time

N = 17
TOTAL_INPUTS = 1 << N  # 131072
DEADLINE = 85.0  # Safety margin before 90s kill


def _build_bitmasks():
    """
    Build bitmask representations for all 2^N Boolean inputs.
    
    For each wire w (0..N-1), we build an integer where bit k is set
    iff input k has a 1 on wire w.
    
    Input k is an N-bit number. Wire w corresponds to bit w of that number.
    """
    # wire_bitmasks[w] is an integer with bit k = 1 iff the k-th input has 1 on wire w
    wire_bitmasks = [0] * N
    for k in range(TOTAL_INPUTS):
        for w in range(N):
            if (k >> w) & 1:
                wire_bitmasks[w] |= (1 << k)
    return wire_bitmasks


def _is_sorted(wire_bitmasks):
    """
    Check if the network output is sorted for all inputs.
    
    For the network to be correct, for every input k, the output must be sorted.
    The output on wire w for input k should be 1 iff the number of 1s in input k
    is greater than (N - 1 - w), i.e., the top (count_ones) wires are 1.
    
    Equivalently: wire w should be 1 for input k iff popcount(k) > N - 1 - w.
    """
    # Precompute expected output bitmasks
    # expected[w] has bit k set iff popcount(k) > N - 1 - w
    for w in range(N):
        threshold = N - 1 - w
        expected = 0
        for k in range(TOTAL_INPUTS):
            if bin(k).count('1') > threshold:
                expected |= (1 << k)
        if wire_bitmasks[w] != expected:
            return False
    return True


def _simulate(network, wire_bitmasks):
    """
    Simulate the sorting network on all inputs simultaneously using bitmasks.
    Returns the final wire_bitmasks after applying all comparators.
    """
    wb = list(wire_bitmasks)  # copy
    for (i, j) in network:
        # Compare-exchange: min goes to wire i, max goes to wire j
        # min(a, b) = a & b
        # max(a, b) = a | b
        new_i = wb[i] & wb[j]
        new_j = wb[i] | wb[j]
        wb[i] = new_i
        wb[j] = new_j
    return wb


def _odd_even_transposition():
    """Build the odd-even transposition sorting network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network


def _greedy_delete(network, wire_bitmasks):
    """
    Greedy deletion: try to remove each comparator if the network still sorts.
    Repeat until no more can be removed.
    """
    current = list(network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx + 1:]
            wb = _simulate(candidate, wire_bitmasks)
            if _is_sorted(wb):
                current = candidate
                changed = True
    return current


def construct():
    start_time = time.time()
    
    # Build initial odd-even transposition network
    network = _odd_even_transposition()
    
    # Build wire bitmasks once
    wire_bitmasks = _build_bitmasks()
    
    # Verify initial network is valid
    wb = _simulate(network, wire_bitmasks)
    if not _is_sorted(wb):
        # Should not happen, but return as-is
        return network
    
    # Greedy deletion pass 1
    network = _greedy_delete(network, wire_bitmasks)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return network
    
    # Try multiple rounds of greedy deletion with different orderings
    best_network = network
    
    # Round 2: try again (sometimes a second pass helps)
    network2 = _greedy_delete(network, wire_bitmasks)
    if len(network2) < len(best_network):
        best_network = network2
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return best_network
    
    # Try forward greedy deletion (from start to end)
    network3 = list(best_network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(network3)):
            candidate = network3[:idx] + network3[idx + 1:]
            wb = _simulate(candidate, wire_bitmasks)
            if _is_sorted(wb):
                network3 = candidate
                changed = True
                break  # restart from beginning since list changed
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 3:
            break
    
    if len(network3) < len(best_network):
        best_network = network3
    
    return best_network
