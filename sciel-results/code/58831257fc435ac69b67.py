"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _all_bool_inputs(n):
    """Generate all 2^n boolean inputs as a (2^n, n) array of 0s and 1s."""
    return np.array([[int((mask >> i) & 1) for i in range(n)] for mask in range(1 << n)], dtype=np.int8)

def _simulate_network(comparators, inputs):
    """Simulate a network on all inputs, return final state."""
    n = inputs.shape[1]
    state = inputs.copy()
    for (i, j) in comparators:
        col_i = state[:, i]
        col_j = state[:, j]
        new_i = np.minimum(col_i, col_j)
        new_j = np.maximum(col_i, col_j)
        state[:, i] = new_i
        state[:, j] = new_j
    return state

def _is_sorted_state(state):
    """Check if all rows in state are sorted (non-decreasing)."""
    n = state.shape[1]
    for k in range(1, n):
        if np.any(state[:, k] < state[:, k - 1]):
            return False
    return True

def _is_sorted_network(comparators, inputs):
    """Check if the network sorts all given inputs using zero-one principle."""
    state = _simulate_network(comparators, inputs)
    return _is_sorted_state(state)

def _seed_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _build_dag(comparators):
    """
    Build a DAG representation of the sorting network.
    Each comparator node has:
      - index in the network
      - wires (i, j)
      - predecessor nodes (comparators that affect wires i or j before this one)
      - successor nodes (comparators that affect wires i or j after this one)
    
    Returns:
      nodes: list of dicts with keys: idx, wires, preds, succs
      wire_to_nodes: dict mapping wire -> list of node indices that touch that wire (in order)
    """
    n_comps = len(comparators)
    nodes = []
    for idx, (i, j) in enumerate(comparators):
        nodes.append({
            'idx': idx,
            'wires': (i, j),
            'preds': set(),
            'succs': set()
        })
    
    # For each wire, track which nodes touch it in order
    wire_to_nodes = {w: [] for w in range(N)}
    for idx, (i, j) in enumerate(comparators):
        wire_to_nodes[i].append(idx)
        wire_to_nodes[j].append(idx)
    
    # Build predecessor/successor relationships
    # Two comparators are connected if they share a wire and one comes before the other
    for wire in range(N):
        nodes_on_wire = wire_to_nodes[wire]
        for pos in range(len(nodes_on_wire)):
            node_a = nodes_on_wire[pos]
            for pos2 in range(pos + 1, len(nodes_on_wire)):
                node_b = nodes_on_wire[pos2]
                nodes[node_a]['succs'].add(node_b)
                nodes[node_b]['preds'].add(node_a)
    
    return nodes, wire_to_nodes

def _compute_node_criticality(nodes, comparators, all_inputs):
    """
    For each node, determine which boolean inputs are "critical" for it.
    An input is critical for a comparator if removing that comparator would
    cause the output to be unsorted for that input.
    
    Returns: dict mapping node_idx -> set of input indices that are critical
    """
    n_comps = len(comparators)
    n_inputs = all_inputs.shape[0]
    criticality = {idx: set() for idx in range(n_comps)}
    
    # For each input, simulate the network and track when each comparator "matters"
    # A comparator matters if its output differs from its input (i.e., it actually swaps)
    # But more precisely, it matters if removing it would break sortedness.
    
    # Efficient approach: for each input, run the network and track the state
    # before and after each comparator. If the comparator changes the state,
    # it might be critical. But we need to check if it's truly critical.
    
    # For now, use a heuristic: a comparator is critical for an input if it swaps
    # (i.e., min != max before the comparison). This is a necessary but not sufficient
    # condition for criticality, but it gives us a good approximation.
    
    for input_idx in range(n_inputs):
        state = all_inputs[input_idx].copy()
        for comp_idx, (i, j) in enumerate(comparators):
            if state[i] != state[j]:
                # This comparator swaps for this input
                criticality[comp_idx].add(input_idx)
    
    return criticality

def _topological_peel(comparators, all_inputs, deadline):
    """
    Apply topological peeling to remove redundant comparators.
    
    Strategy:
    1. Build DAG
    2. Compute criticality for each node
    3. Iteratively remove nodes that are not critical for any input
       (i.e., no input depends on them for sortedness)
    4. After removal, recompute criticality and repeat
    
    Returns: reduced list of comparators
    """
    network = comparators[:]
    start_time = time.time()
    
    while time.time() < deadline:
        if len(network) == 0:
            break
        
        # Build DAG
        nodes, wire_to_nodes = _build_dag(network)
        
        # Compute criticality
        criticality = _compute_node_criticality(nodes, network, all_inputs)
        
        # Find nodes that can be removed (not critical for any input)
        removable = []
        for idx in range(len(network)):
            if len(criticality[idx]) == 0:
                removable.append(idx)
        
        if not removable:
            # No more removable nodes, break
            break
        
        # Remove the last removable node (to preserve order as much as possible)
        # Actually, let's remove all non-critical nodes in one pass
        # But we need to be careful: removing one node might make another node critical
        # So let's remove them one at a time, starting from the end
        
        removed_any = False
        for idx in reversed(removable):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            # Verify the candidate is still valid
            if _is_sorted_network(candidate, all_inputs):
                network = candidate
                removed_any = True
                # Don't break, continue removing more
            else:
                # This node was actually critical (our heuristic was wrong)
                # Mark it as critical and continue
                pass
        
        if not removed_any:
            # No nodes could be removed, break
            break
    
    return network

def _local_search_optimize(comparators, all_inputs, deadline):
    """
    Local search: try swapping, inserting, and removing comparators.
    """
    network = comparators[:]
    start_time = time.time()
    
    # Phase 1: Try removing comparators (greedy)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _is_sorted_network(candidate, all_inputs):
                network = candidate
                improved = True
                break  # Restart from beginning
    
    # Phase 2: Try swapping adjacent comparators
    if time.time() < deadline:
        max_swaps = 200
        for _ in range(max_swaps):
            if time.time() >= deadline:
                break
            if len(network) < 2:
                break
            idx = random.randint(0, len(network) - 2)
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if _is_sorted_network(candidate, all_inputs):
                network = candidate
    
    # Phase 3: Try inserting a comparator at a random position
    # This can help if we removed too many
    if time.time() < deadline:
        max_inserts = 50
        for _ in range(max_inserts):
            if time.time() >= deadline:
                break
            # Random wire pair
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Random position
            pos = random.randint(0, len(network))
            candidate = network[:pos] + [(i, j)] + network[pos:]
            if _is_sorted_network(candidate, all_inputs):
                # Only accept if it doesn't increase the count (we want to minimize)
                # Actually, we should only insert if it allows further removal
                # For now, skip insertion since we want to minimize
                pass
    
    return network

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Pre-compute all boolean inputs
    all_inputs = _all_bool_inputs(N)
    
    # Start with seed network
    network = _seed_network()
    
    # Track best
    best_network = network
    best_count = len(network)
    
    # Phase 1: Topological peeling
    peeled = _topological_peel(network, all_inputs, deadline)
    if len(peeled) < best_count:
        best_network = peeled
        best_count = len(peeled)
    
    # Phase 2: Local search optimization
    optimized = _local_search_optimize(peeled, all_inputs, deadline)
    if len(optimized) < best_count:
        best_network = optimized
        best_count = len(optimized)
    
    return best_network
# EVOLVE-BLOCK-END
