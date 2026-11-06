"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import product

def _apply_comparator(state, i, j):
    """Apply a comparator (i,j) to a Boolean state (list or array of 0/1).
    Writes min to i and max to j."""
    state[i], state[j] = min(state[i], state[j]), max(state[i], state[j])

def _is_sorted(state):
    """Check if state is sorted in non-decreasing order."""
    for k in range(len(state) - 1):
        if state[k] > state[k + 1]:
            return False
    return True

def _apply_network(state, network):
    """Apply an entire network to a state. Returns a new state."""
    s = list(state)
    for i, j in network:
        s[i], s[j] = min(s[i], s[j]), max(s[i], s[j])
    return s

def _network_sorts_all(network, n=N):
    """Check if the network sorts all 2^n Boolean inputs.
    Uses the zero-one principle: a network sorts all inputs iff it sorts all Boolean inputs."""
    for bits in range(1 << n):
        state = [(bits >> k) & 1 for k in range(n)]
        result = _apply_network(state, network)
        if not _is_sorted(result):
            return False
    return True

def _network_depth(network):
    """Compute the parallel depth of a network."""
    if not network:
        return 0
    # Use a simple longest-path method
    # For each comparator, its depth is max(depth of i, depth of j) + 1
    # But wires can be involved in multiple comparators, so we track depth per wire
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0

def _batcher_odd_even_merge(n):
    """Generate a Batcher odd-even merge network for n wires.
    Returns a list of (i, j) comparators."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comparators = []
    half = n // 2
    # Recursively sort even and odd positions
    comparators.extend(_batcher_odd_even_merge(half))
    # Shift indices: the recursive calls operate on local indices, we need to map them
    # Actually, let's implement this properly with index mapping
    
    # Better approach: implement Batcher's odd-even merge correctly
    return _batcher_oe_merge_impl(list(range(n)), n)

def _batcher_oe_merge_impl(wires, n):
    """Implement Batcher's odd-even merge on given wire labels."""
    if n == 1:
        return []
    if n == 2:
        return [(wires[0], wires[1])]
    
    comparators = []
    half = n // 2
    even_wires = wires[0::2]
    odd_wires = wires[1::2]
    
    # Sort even and odd subsequences
    comparators.extend(_batcher_oe_merge_impl(even_wires, len(even_wires)))
    comparators.extend(_batcher_oe_merge_impl(odd_wires, len(odd_wires)))
    
    # Merge phase: compare wires[i] and wires[i+half] for i in range(half)
    # Actually, for odd-even merge, after sorting even and odd, we compare:
    # for i in range(n-1, 0, -2): compare wires[i] and wires[i-1]
    # Let me look up the exact structure...
    
    # The standard odd-even merge:
    # 1. Recursively sort even-indexed elements
    # 2. Recursively sort odd-indexed elements
    # 3. For i = n-1, n-3, ..., 1: compare-exchange elements i and i-1
    # 4. For i = n-2, n-4, ..., 0: compare-exchange elements i and i+1 (if valid)
    # Actually, step 3 is: for i in range(n-1, 0, -2): compare wires[i] and wires[i-1]
    # Step 4 is: for i in range(n-2, 0, -2): compare wires[i] and wires[i-1]
    
    # Let me just use a simpler known network: Batcher's sorting network (bitonic-like)
    return comparators

def _batcher_sort(n):
    """Generate a Batcher sorting network for n wires."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Use the standard Batcher network construction
    # This is complex, so let's use a simpler approach:
    # Use a known good network like the AKS or a small hand-tuned one
    # For n=18, let's try a recursive bitonic-like approach
    
    return _bitonic_sort_network(n)

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (must be power of 2 for classic version).
    For non-power-of-2, we can pad or use a different approach."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, use a simple insertion-like network or
    # just use odd-even transposition as fallback but try to optimize
    if n & (n - 1) != 0:
        # Not a power of 2, use odd-even transposition
        return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]
    
    comparators = []
    comparators.extend(_bitonic_merge(n, 0, n, True))
    return comparators

def _bitonic_merge(n, start, length, ascending):
    """Bitonic merge on wires start to start+length."""
    if length <= 1:
        return []
    half = length // 2
    comparators = []
    # First, split into two halves and sort them in opposite directions
    comparators.extend(_bitonic_merge(n, start, half, ascending))
    comparators.extend(_bitonic_merge(n, start + half, half, not ascending))
    # Then compare-exchange across the halves
    for i in range(half):
        a = start + i
        b = start + half + i
        if not ascending:
            a, b = b, a
        comparators.append((a, b))
    return comparators

def _random_network(n, max_comparators=100):
    """Generate a random network."""
    num_comp = random.randint(50, max_comparators)
    network = []
    for _ in range(num_comp):
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        network.append((i, j))
    return network

def _mutate_network(network, n):
    """Mutate a network by adding, removing, or swapping comparators."""
    new_network = list(network)
    op = random.random()
    if op < 0.3 and len(new_network) > 10:
        # Remove a random comparator
        idx = random.randint(0, len(new_network) - 1)
        new_network.pop(idx)
    elif op < 0.6:
        # Add a random comparator
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        new_network.append((i, j))
    else:
        # Swap two comparators
        if len(new_network) >= 2:
            i, j = random.sample(range(len(new_network)), 2)
            new_network[i], new_network[j] = new_network[j], new_network[i]
    return new_network

def construct():
    """Construct a sorting network for N wires.
    
    Uses a genetic algorithm to evolve a network that sorts all Boolean inputs,
    minimizing the number of comparators."""
    n = N
    best_network = None
    best_count = float('inf')
    
    # Initial population: odd-even transposition and some random networks
    pop = []
    pop.append([(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)])
    for _ in range(10):
        pop.append(_random_network(n, 80))
    
    for gen in range(5):
        # Evaluate population
        scored_pop = []
        for net in pop:
            if _network_sorts_all(net, n):
                scored_pop.append((len(net), net))
        
        if scored_pop:
            scored_pop.sort(key=lambda x: x[0])
            best_network = scored_pop[0][1]
            best_count = scored_pop[0][0]
        
        # Select, mutate for next generation
        new_pop = []
        if scored_pop:
            # Keep the best few
            for _, net in scored_pop[:5]:
                new_pop.append(net)
            # Mutate the best
            for _ in range(5):
                _, net = scored_pop[0]
                new_pop.append(_mutate_network(net, n))
        else:
            # All invalid, just generate random
            for _ in range(10):
                new_pop.append(_random_network(n, 80))
        
        pop = new_pop
    
    if best_network is not None:
        return best_network
    else:
        # Fallback: odd-even transposition
        return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]
# EVOLVE-BLOCK-END
