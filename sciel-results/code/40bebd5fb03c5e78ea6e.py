"""Size-efficient sorting networks on 17 wires."""
N = 17
import time
import numpy as np

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorted_fast(state):
    """Check if all rows are sorted (non-decreasing) using vectorized ops."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _find_redundant_comparators(network, all_inputs):
    """Find all redundant comparators in one pass.
    
    A comparator (i, j) is redundant if for every input, the value on wire i
    is already <= the value on wire j before the comparator executes.
    Also uses a DAG-based heuristic: if wire i's last write before this comparator
    comes from a wire that is "above" wire j in the partial order, the comparator
    may be redundant.
    """
    n_comps = len(network)
    if n_comps == 0:
        return set()
    
    # Simulate and record the two relevant wire values before each comparator
    state = all_inputs.copy()
    wire_vals_before = []
    for i, j in network:
        wire_vals_before.append((state[:, i].copy(), state[:, j].copy()))
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check which comparators are redundant
    redundant = set()
    for k, (vi, vj) in enumerate(wire_vals_before):
        if not np.any(vi > vj):
            redundant.add(k)
    
    return redundant

def _build_dag_and_prune(network, all_inputs, deadline):
    """Use DAG structure to identify potentially removable comparators.
    
    Builds a dependency graph where a comparator depends on all prior comparators
    that write to either of its wires. A comparator is a candidate for removal
    if its two input wires have a known ordering from prior comparators.
    """
    n_comps = len(network)
    if n_comps == 0:
        return network
    
    # For each wire, track the last comparator that wrote to it
    last_write = {}  # wire -> (comp_index, other_wire, is_min)
    # is_min=True means this wire got the min value, is_min=False means max
    
    # Process comparators in order, tracking wire dependencies
    for idx, (i, j) in enumerate(network):
        # Record what this comparator does to wires i and j
        last_write[i] = (idx, j, True)   # i gets min
        last_write[j] = (idx, i, False)  # j gets max
    
    # For each comparator, check if the ordering is already guaranteed
    # by the DAG of prior operations
    removable = set()
    for idx, (i, j) in enumerate(network):
        if idx in removable:
            continue
        
        # Get the last writes to wires i and j before this comparator
        # We need to look at the state just before this comparator
        # For efficiency, we use the wire_vals approach but with DAG hints
        
        # Check: if wire i was last written by a comparator that put it as min
        # and wire j was last written by a comparator that put it as max,
        # and those comparators are "before" in some sense, it might be redundant
        # This is a heuristic - we'll verify with simulation
        
        # Simple heuristic: if both wires have been written recently and
        # the last write to i was a min and to j was a max, flag for testing
        if i in last_write and j in last_write:
            last_i = last_write[i]
            last_j = last_write[j]
            # If i's last write is before this comp and made it min,
            # and j's last write is before this comp and made it max,
            # then i <= j might already hold
            if last_i[0] < idx and last_i[2] == True and last_j[0] < idx and last_j[2] == False:
                removable.add(idx)
    
    # Verify removable candidates
    if removable:
        # Try removing all candidates at once, then verify
        new_network = [c for idx, c in enumerate(network) if idx not in removable]
        if _is_sorting_network_fast(new_network, all_inputs):
            return new_network
    
    return network

def _try_remove_single(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _batch_try_removals(network, candidates, all_inputs):
    """Try removing multiple comparators at once. Returns new network if all can be removed."""
    if not candidates:
        return network
    
    new_network = [c for idx, c in enumerate(network) if idx not in candidates]
    if _is_sorting_network_fast(new_network, all_inputs):
        return new_network
    return network

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively remove redundant comparators using batch detection and targeted removal."""
    network = list(network)
    
    # Phase 1: Aggressive batch removal of provably redundant comparators
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        
        redundant = _find_redundant_comparators(network, all_inputs)
        if not redundant:
            break
        
        # Remove all redundant comparators at once
        network = [c for i, c in enumerate(network) if i not in redundant]
        improved = True
    
    # Phase 2: DAG-based pruning
    if time.time() < deadline:
        network = _build_dag_and_prune(network, all_inputs, deadline)
    
    # Phase 3: Batch removal of groups of candidates
    # Try removing pairs of adjacent comparators
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        
        n = len(network)
        found = False
        for i in range(n - 1):
            if time.time() > deadline:
                return network
            # Try removing comparator i and i+1 together
            candidates = {i, i + 1}
            new_network = _batch_try_removals(network, candidates, all_inputs)
            if len(new_network) < len(network):
                network = new_network
                improved = True
                found = True
                break
        
        # If no pair worked, try single removals from the end
        if not found:
            for i in range(len(network) - 1, -1, -1):
                if time.time() > deadline:
                    return network
                if _try_remove_single(network, i, all_inputs):
                    network.pop(i)
                    improved = True
                    break
    
    # Phase 4: Another batch pass to catch newly redundant comparators
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        
        redundant = _find_redundant_comparators(network, all_inputs)
        if not redundant:
            break
        
        network = [c for i, c in enumerate(network) if i not in redundant]
        improved = True
    
    # Phase 5: DAG-based pruning again
    if time.time() < deadline:
        network = _build_dag_and_prune(network, all_inputs, deadline)
    
    # Phase 6: Targeted single-removal search from the beginning
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                return network
            if _try_remove_single(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
    
    # Phase 7: Final batch pass
    if time.time() < deadline:
        redundant = _find_redundant_comparators(network, all_inputs)
        if redundant:
            network = [c for i, c in enumerate(network) if i not in redundant]
    
    # Phase 8: Try triple removals of consecutive comparators
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            n = len(network)
            for i in range(n - 2):
                if time.time() > deadline:
                    return network
                candidates = {i, i + 1, i + 2}
                new_network = _batch_try_removals(network, candidates, all_inputs)
                if len(new_network) < len(network):
                    network = new_network
                    improved = True
                    break
    
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    # Final verification
    if not _is_sorting_network_fast(network, all_inputs):
        # Fallback to original
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    return network
