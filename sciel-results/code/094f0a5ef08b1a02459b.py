"""Size-efficient sorting network search for N=16 wires.

Structural approach: layered removal with greedy repair.
- Start from Batcher's 60-comparator bitonic network.
- Systematically attempt removal of each comparator (layer by layer).
- If removal breaks correctness, try greedy repair by inserting
  a different comparator at a promising position.
- Also try reordering via dependency-aware shuffling.
- Use bit-parallel verification over all 65536 Boolean inputs.
"""

import time
import random
from collections import defaultdict

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def compute_states(net):
    """Returns (states, final_state) for incremental checking."""
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    for k, (i, j) in enumerate(net):
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k + 1] = next_state
        current_state = next_state
    return states, current_state

def is_sorted_state(final_state):
    """Check if the final state is sorted."""
    for i in range(N - 1):
        if final_state[i] & ~final_state[i + 1]:
            return False
    return True

def remove_redundant(net):
    """Removes all no-op comparators in one sweep."""
    if not net:
        return [], _get_initial_states()[:]
    states, final_state = compute_states(net)
    to_remove = set()
    for k, (i, j) in enumerate(net):
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
    if not to_remove:
        return net, final_state
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires."""
    if n <= 1:
        return []
    comparators = []
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    bitonic_sort(n, 0, True)
    return comparators

def identify_layers(net):
    """
    Identify parallel layers from a network.
    Two comparators are in the same layer if they don't share a wire
    and all comparators before them in the layer are independent.
    Returns list of lists, each inner list is a layer (set of parallel comparators).
    """
    if not net:
        return []
    layers = []
    used_wires = set()
    current_layer = []
    for (i, j) in net:
        if i in used_wires or j in used_wires:
            # Can't add to current layer, start new layer
            if current_layer:
                layers.append(current_layer)
            current_layer = [(i, j)]
            used_wires = {i, j}
        else:
            current_layer.append((i, j))
            used_wires.add(i)
            used_wires.add(j)
    if current_layer:
        layers.append(current_layer)
    return layers

def rebuild_from_layers(layers):
    """Rebuild a flat network from layers, preserving order within layers."""
    net = []
    for layer in layers:
        net.extend(layer)
    return net

def systematic_removal_search(net, deadline, seed):
    """
    Structured removal search:
    1. Try removing each comparator one at a time (in order).
    2. If removal keeps network valid, accept it.
    3. If removal breaks validity, try a greedy repair: add a different
       comparator at a position where it might help.
    4. Repeat until no more progress or deadline.
    """
    rng = random.Random(seed)
    best_net = list(net)
    best_count = len(net)
    current_net = list(net)
    
    # Initial cleanup
    current_net, final_state = remove_redundant(current_net)
    if is_sorted_state(final_state):
        best_net = list(current_net)
        best_count = len(current_net)
    
    # Systematic removal rounds
    max_rounds = 200
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        removed_any = False
        n = len(current_net)
        
        # Try removing each comparator (shuffled order for variety)
        indices = list(range(n))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(current_net):
                continue
            
            candidate = current_net[:idx] + current_net[idx + 1:]
            candidate, candidate_final = remove_redundant(candidate)
            
            if is_sorted_state(candidate_final):
                # Accept the removal
                current_net = candidate
                removed_any = True
                if len(current_net) < best_count:
                    best_net = list(current_net)
                    best_count = len(current_net)
                # After a removal, restart the round
                break
        
        if removed_any:
            continue
        
        # No simple removals left. Try reordering + removal.
        # Shuffle the network and try again
        if rng.random() < 0.3:
            # Try swapping random pairs
            if len(current_net) > 1:
                idx1 = rng.randint(0, len(current_net) - 2)
                idx2 = idx1 + 1
                candidate = current_net[:]
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                candidate, candidate_final = remove_redundant(candidate)
                if is_sorted_state(candidate_final):
                    current_net = candidate
                    if len(current_net) < best_count:
                        best_net = list(current_net)
                        best_count = len(current_net)
        
        # Try inserting a helpful comparator then removing another
        if len(current_net) > 0 and rng.random() < 0.5:
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final) and len(candidate) < len(current_net):
                current_net = candidate
                if len(current_net) < best_count:
                    best_net = list(current_net)
                    best_count = len(current_net)
    
    return best_net

def layer_reorder_search(net, deadline, seed):
    """
    Try different orderings within parallel layers.
    Also try merging/splitting layers.
    """
    rng = random.Random(seed)
    best_net = list(net)
    best_count = len(net)
    current_net = list(net)
    
    current_net, final_state = remove_redundant(current_net)
    if is_sorted_state(final_state):
        best_net = list(current_net)
        best_count = len(current_net)
    
    # Try shuffling within layers
    for _ in range(50):
        if time.time() > deadline:
            break
        
        layers = identify_layers(current_net)
        # Shuffle comparators within each layer
        new_layers = []
        for layer in layers:
            shuffled = list(layer)
            rng.shuffle(shuffled)
            new_layers.append(shuffled)
        
        candidate = rebuild_from_layers(new_layers)
        candidate, candidate_final = remove_redundant(candidate)
        
        if is_sorted_state(candidate_final):
            if len(candidate) < best_count:
                best_net = list(candidate)
                best_count = len(candidate)
            current_net = candidate
        
        # Try moving a comparator from one layer to another
        if len(layers) > 1 and rng.random() < 0.4:
            src_layer = rng.randint(0, len(layers) - 1)
            dst_layer = rng.randint(0, len(layers) - 1)
            if src_layer == dst_layer:
                continue
            if len(layers[src_layer]) <= 1:
                continue
            
            comp = layers[src_layer][0]
            new_src = layers[src_layer][1:]
            new_dst = layers[dst_layer] + [comp]
            
            new_layers2 = []
            for li, layer in enumerate(layers):
                if li == src_layer:
                    new_layers2.append(new_src)
                elif li == dst_layer:
                    new_layers2.append(new_dst)
                else:
                    new_layers2.append(layer)
            
            candidate2 = rebuild_from_layers(new_layers2)
            candidate2, candidate2_final = remove_redundant(candidate2)
            if is_sorted_state(candidate2_final):
                if len(candidate2) < best_count:
                    best_net = list(candidate2)
                    best_count = len(candidate2)
                current_net = candidate2
    
    return best_net

def dependency_aware_reorder(net, seed):
    """
    Reorder comparators based on wire dependency.
    Comparators that feed into other comparators should come first.
    """
    rng = random.Random(seed)
    
    # Build a simple dependency: comparator A depends on B if A reads a wire
    # that B writes, and B appears before A in the current ordering.
    # We want to preserve this topological order but shuffle independent ones.
    
    # Simple approach: group by "generation" based on wire last-write time
    last_write = [0] * N
    generations = []
    for k, (i, j) in enumerate(net):
        gen = max(last_write[i], last_write[j])
        generations.append(gen)
        last_write[i] = k
        last_write[j] = k
    
    # Group by generation
    gen_groups = defaultdict(list)
    for k, gen in enumerate(generations):
        gen_groups[gen].append(k)
    
    # Shuffle within each generation group
    new_order = []
    for gen in sorted(gen_groups.keys()):
        group = gen_groups[gen]
        rng.shuffle(group)
        new_order.extend(group)
    
    return [net[k] for k in new_order]

def construct():
    deadline = time.time() + 82  # Leave 8s margin for overhead
    
    # Start from Batcher's bitonic network (60 comparators)
    base_net = bitonic_sorter(N)
    base_net, base_final = remove_redundant(base_net)
    
    best_net = list(base_net)
    best_count = len(base_net)
    
    # Phase 1: Systematic removal search
    phase1_deadline = min(deadline, time.time() + 30)
    found1 = systematic_removal_search(base_net, phase1_deadline, seed=42)
    if len(found1) < best_count:
        best_net = found1
        best_count = len(found1)
    
    # Phase 2: Layer reordering search from the best so far
    phase2_deadline = min(deadline, time.time() + 20)
    found2 = layer_reorder_search(best_net, phase2_deadline, seed=123)
    if len(found2) < best_count:
        best_net = found2
        best_count = len(found2)
    
    # Phase 3: Dependency-aware reordering + removal
    if time.time() < deadline:
        reordered = dependency_aware_reorder(best_net, seed=777)
        reordered, reordered_final = remove_redundant(reordered)
        if is_sorted_state(reordered_final):
            # Now try removal on the reordered network
            phase3_deadline = min(deadline, time.time() + 15)
            found3 = systematic_removal_search(reordered, phase3_deadline, seed=999)
            if len(found3) < best_count:
                best_net = found3
                best_count = len(found3)
    
    # Phase 4: Multiple restarts with different seeds
    if time.time() < deadline:
        restart_deadline = deadline
        for restart_seed in [111, 222, 333]:
            if time.time() > restart_deadline:
                break
            # Slightly perturb the best network
            perturbed = list(best_net)
            rng = random.Random(restart_seed)
            # Remove a random comparator and see if it still works
            if len(perturbed) > 0:
                rem_idx = rng.randint(0, len(perturbed) - 1)
                candidate = perturbed[:rem_idx] + perturbed[rem_idx + 1:]
                candidate, candidate_final = remove_redundant(candidate)
                if is_sorted_state(candidate_final):
                    perturbed = candidate
            
            # Run a short systematic search
            short_deadline = min(restart_deadline, time.time() + 10)
            found4 = systematic_removal_search(perturbed, short_deadline, seed=restart_seed)
            if len(found4) < best_count:
                best_net = found4
                best_count = len(found4)
    
    # Final cleanup
    if best_net is not None:
        best_net, _ = remove_redundant(best_net)
    else:
        best_net = base_net
        best_net, _ = remove_redundant(best_net)
    
    # Safety: verify the final answer
    if not fast_verify(best_net):
        # Fallback to the original bitonic network
        best_net = bitonic_sorter(N)
        best_net, _ = remove_redundant(best_net)
    
    return best_net
