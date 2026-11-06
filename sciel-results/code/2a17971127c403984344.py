"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 90.0
SAFETY_MARGIN = 5.0

def _apply_comparator(wires, i, j):
    """Apply compare-exchange: min goes to i, max to j. wires is a 2D array (num_inputs, N)."""
    a = wires[:, i]
    b = wires[:, j]
    new_a = np.minimum(a, b)
    new_b = np.maximum(a, b)
    wires[:, i] = new_a
    wires[:, j] = new_b

def _check_sorted(wires):
    """Check if all rows are sorted non-decreasingly."""
    return np.all(wires[:, :-1] <= wires[:, 1:])

def _check_all_boolean(wires):
    """Check all 2^18 boolean inputs are sorted."""
    # Use the 0/1 inputs
    # Generate all 2^18 inputs
    bits = np.arange(1 << N, dtype=np.uint16).reshape(-1, 1)
    # Extract bits: bit k of value corresponds to wire k
    # For a value v, wire i has bit (v >> i) & 1
    masks = (1 << np.arange(N, dtype=np.uint16))
    all_inputs = (bits >> np.arange(N, dtype=np.uint16)) & 1
    all_inputs = all_inputs.astype(np.uint8)
    
    # Apply the network
    for (i, j) in network:
        a = all_inputs[:, i]
        b = all_inputs[:, j]
        all_inputs[:, i] = np.minimum(a, b)
        all_inputs[:, j] = np.maximum(a, b)
    
    # Check sorted
    return np.all(all_inputs[:, :-1] <= all_inputs[:, 1:])

def _depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    # A simple upper bound: number of comparators (worst case all sequential)
    # But we can compute a tighter bound using a greedy coloring
    # For now, use the number of comparators as a rough upper bound, 
    # but for scoring we only care about count, so we'll just return len(network)
    # Actually, let's compute the true depth using a simple algorithm
    # depth[i] = 0 for all wires
    # For each comparator (i, j) in order:
    #   depth[i] = max(depth[i], depth[j])  # not quite right
    # Actually, the depth of a comparator is 1 + max(depth of last use of i, depth of last use of j)
    # Let's use a different approach: simulate layers
    # A layer is a set of comparators that can be done in parallel (disjoint wires)
    # We'll greedily assign layers
    wire_layer = [0] * N
    for (i, j) in network:
        # This comparator must be done after the last comparator on wire i and wire j
        # So its layer is max(wire_layer[i], wire_layer[j]) + 1
        layer = max(wire_layer[i], wire_layer[j]) + 1
        wire_layer[i] = layer
        wire_layer[j] = layer
    return max(wire_layer) if wire_layer else 0

def _local_search(current_network, max_iterations=5000):
    """Perform local search to improve the network."""
    current_network = list(current_network)
    current_count = len(current_network)
    
    # Precompute all 2^18 inputs
    bits = np.arange(1 << N, dtype=np.uint16).reshape(-1, 1)
    all_inputs = (bits >> np.arange(N, dtype=np.uint16)) & 1
    all_inputs = all_inputs.astype(np.uint8)
    
    def is_valid(net):
        test = all_inputs.copy()
        for (i, j) in net:
            a = test[:, i]
            b = test[:, j]
            test[:, i] = np.minimum(a, b)
            test[:, j] = np.maximum(a, b)
        return np.all(test[:, :-1] <= test[:, 1:])
    
    # Check initial validity
    if not is_valid(current_network):
        return current_network
    
    best = current_count
    for _ in range(max_iterations):
        # Try random modifications
        action = np.random.random()
        
        if action < 0.3:
            # Try removing a comparator
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network))
                new_net = current_network[:idx] + current_network[idx+1:]
                if is_valid(new_net):
                    current_network = new_net
                    best = min(best, len(current_network))
        elif action < 0.6:
            # Try moving a comparator
            if len(current_network) > 1:
                idx = np.random.randint(len(current_network))
                new_idx = np.random.randint(len(current_network))
                if idx != new_idx:
                    comp = current_network[idx]
                    new_net = current_network[:idx] + current_network[idx+1:new_idx] + [comp] + current_network[new_idx:]
                    if is_valid(new_net):
                        current_network = new_net
        else:
            # Try adding a random comparator
            if len(current_network) < MAX_COMPARATORS:
                i = np.random.randint(N)
                j = np.random.randint(i+1, N)
                pos = np.random.randint(len(current_network) + 1)
                new_net = current_network[:pos] + [(i, j)] + current_network[pos:]
                if is_valid(new_net):
                    current_network = new_net
                    best = min(best, len(current_network))
    
    return current_network

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Start with a known good network: odd-even transposition
    # For N=18, this takes N phases
    def make_odd_even_transposition(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    current_network = make_odd_even_transposition(N)
    current_count = len(current_network)
    
    # Verify initial network
    bits = np.arange(1 << N, dtype=np.uint16).reshape(-1, 1)
    all_inputs = (bits >> np.arange(N, dtype=np.uint16)) & 1
    all_inputs = all_inputs.astype(np.uint8)
    
    def is_valid(net):
        test = all_inputs.copy()
        for (i, j) in net:
            a = test[:, i]
            b = test[:, j]
            test[:, i] = np.minimum(a, b)
            test[:, j] = np.maximum(a, b)
        return np.all(test[:, :-1] <= test[:, 1:])
    
    if not is_valid(current_network):
        # Fallback: return the seed as-is
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search with time check
    while time.time() < deadline:
        current_network = _local_search(current_network, max_iterations=2000)
        if time.time() > deadline:
            break
    
    return current_network
# EVOLVE-BLOCK-END
