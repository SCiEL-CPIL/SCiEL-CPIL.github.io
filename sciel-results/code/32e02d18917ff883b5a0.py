"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, OrderedDict

_DEADLINE = 55.0  # Safety margin before 60s limit

def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return parents.get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    else:
        return []

def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        # Try to infer from parents
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []

def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs

def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {}
    children = defaultdict(list)
    
    for v in all_v:
        ps = _get_parents(dag, v)
        parents[v] = ps
        for p in ps:
            children[p].append(v)
    
    # Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = [v for v in all_v if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        # Cycle or error, just return all vertices
        return all_v
    
    return topo

def _compute_use_counts(dag):
    """Compute how many times each vertex's value is used."""
    all_v = _get_all_vertices(dag)
    use_count = defaultdict(int)
    
    for v in all_v:
        for p in _get_parents(dag, v):
            use_count[p] += 1
    
    # Outputs are also "used"
    for o in _get_outputs(dag):
        use_count[o] += 1
    
    return use_count

def _decompose_into_chains(dag):
    """
    Decompose DAG into chains using weighted greedy matching.
    Prioritizes longer potential chains for better locality.
    """
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    parents_map = {v: _get_parents(dag, v) for v in all_v}
    children_map = defaultdict(list)
    for v in all_v:
        for p in parents_map[v]:
            children_map[p].append(v)
    
    # Compute chain potential: how far each vertex can extend
    def _chain_potential(v, memo={}):
        if v in memo:
            return memo[v]
        best = 1
        for c in children_map[v]:
            best = max(best, 1 + _chain_potential(c, memo))
        memo[v] = best
        return best
    
    # Find a maximum matching between parents and children
    # Weighted greedy: prioritize edges that lead to longer chains
    matched_children = {}  # parent -> child
    matched_parents = {}   # child -> parent
    
    # Create edge list with weights (potential chain length)
    edges = []
    for v in all_v:
        for c in children_map[v]:
            weight = _chain_potential(c)
            edges.append((weight, v, c))
    
    # Sort by weight descending (prioritize longer chains)
    edges.sort(key=lambda x: -x[0])
    
    for weight, v, c in edges:
        if v in matched_parents:
            continue
        if c in matched_parents:
            continue
        if v in matched_children:
            continue
        matched_children[v] = c
        matched_parents[c] = v
    
    # Build chains
    chains = []
    visited = set()
    
    # Start chains from vertices with no parent in matching
    for v in all_v:
        if v in visited:
            continue
        if v in matched_parents:
            continue
        # Start a new chain
        chain = [v]
        current = v
        visited.add(current)
        while current in matched_children:
            current = matched_children[current]
            if current in visited:
                break
            chain.append(current)
            visited.add(current)
        chains.append(chain)
    
    # Handle any remaining vertices not in chains (shouldn't happen but safety)
    for v in all_v:
        if v not in visited:
            chains.append([v])
            visited.add(v)
    
    # Sort chains by length descending (process longer chains first for better locality)
    chains.sort(key=lambda c: -len(c))
    
    return chains

def _pebble_chain_round_robin(dag, fast_memory_budget, deadline):
    """
    Pebble using chain decomposition with lookahead-based eviction and
    urgency-aware round-robin scheduling.
    """
    start_time = time.time()
    
    if time.time() - deadline[0] > _DEADLINE:
        return []
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    outputs = _get_outputs(dag)
    outputs_set = set(outputs)
    use_counts = _compute_use_counts(dag)
    chains = _decompose_into_chains(dag)
    
    if not chains:
        return []
    
    operations = []
    in_fast = set()
    
    # Track position in each chain
    chain_positions = [0] * len(chains)
    
    # Precompute: for each vertex, which chains contain it and at what position
    vertex_chain_info = {}
    for ci, chain in enumerate(chains):
        for pos, v in enumerate(chain):
            if v not in vertex_chain_info:
                vertex_chain_info[v] = []
            vertex_chain_info[v].append((ci, pos))
    
    # Precompute: for each vertex, its "urgency" - how soon it will be needed
    def _compute_urgency(vertex, current_positions):
        """Compute urgency: minimum distance to next use across all chains."""
        if vertex not in vertex_chain_info:
            return float('inf')
        
        min_distance = float('inf')
        for ci, pos in vertex_chain_info[vertex]:
            current_pos = current_positions[ci]
            if pos >= current_pos:  # Future or current position
                distance = pos - current_pos
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _can_compute(vertex):
        """Check if all parents of vertex are in fast memory."""
        parents = _get_parents(dag, vertex)
        return all(p in in_fast for p in parents)
    
    def _eviction_score(vertex):
        """
        Score for eviction: lower score = better to evict.
        Consider: use count, output status, urgency (how soon needed again)
        """
        base_score = use_counts.get(vertex, 0) * 10
        
        # Heavily penalize evicting outputs
        if vertex in outputs_set:
            base_score += 100000
        
        # Penalize evicting vertices that will be needed soon
        urgency = _compute_urgency(vertex, chain_positions)
        if urgency < float('inf'):
            base_score += max(0, 50 - urgency * 5)  # Urgent vertices are harder to evict
        
        # Slight preference for evicting vertices not in active chains
        if vertex not in vertex_chain_info:
            base_score -= 1
        
        return base_score
    
    def _evict_if_needed(needed=1):
        """Evict vertices to make space using lookahead-based scoring."""
        while len(in_fast) + needed > fast_memory_budget:
            if not in_fast:
                return False
            # Find vertex to evict: lowest score (most expendable)
            best_to_evict = None
            best_score = float('inf')
            for v in in_fast:
                score = _eviction_score(v)
                if score < best_score:
                    best_score = score
                    best_to_evict = v
            if best_to_evict:
                operations.append(['delete', best_to_evict])
                in_fast.remove(best_to_evict)
            else:
                return False
        return True
    
    # Urgency-aware round-robin: prioritize chains that are closer to completion
    # or have higher immediate need
    max_iterations = len(all_v) * 3 + 200  # Safety bound
    iteration = 0
    active_chains = list(range(len(chains)))
    
    while active_chains and iteration < max_iterations:
        if time.time() - deadline[0] > _DEADLINE:
            break
        
        # Sort active chains by urgency: prefer chains closer to completion
        # or with higher use-count vertices next
        def _chain_urgency(ci):
            chain = chains[ci]
            pos = chain_positions[ci]
            if pos >= len(chain):
                return float('-inf')  # Done
            v = chain[pos]
            # Higher use count = more urgent
            urgency = use_counts.get(v, 0) * 10
            # Closer to end of chain = slightly more urgent (completion bonus)
            remaining = len(chain) - pos
            urgency += max(0, 20 - remaining)
            return urgency
        
        active_chains.sort(key=lambda ci: -_chain_urgency(ci))
        
        next_active = []
        for ci in active_chains:
            if time.time() - deadline[0] > _DEADLINE:
                break
            
            chain = chains[ci]
            pos = chain_positions[ci]
            
            if pos >= len(chain):
                continue
            
            v = chain[pos]
            
            # Check if we can compute this vertex
            if _can_compute(v):
                # Ensure we have space for the result
                if not _evict_if_needed(1):
                    next_active.append(ci)
                    continue
                
                # Compute v
                operations.append(['compute', v])
                in_fast.add(v)
                chain_positions[ci] += 1
                
                # Decide whether to keep v in fast memory
                # Keep if: high use count, or it's an output, or memory has space
                keep = False
                if v in outputs_set:
                    keep = True
                elif use_counts.get(v, 0) > 1:
                    keep = True
                elif len(in_fast) < fast_memory_budget:
                    keep = True
                
                if not keep:
                    operations.append(['delete', v])
                    in_fast.remove(v)
                
                # Continue with next in this chain
                next_active.append(ci)
            else:
                # Can't compute yet, try to load a missing parent
                parents = _get_parents(dag, v)
                loaded_any = False
                for p in parents:
                    if p not in in_fast:
                        # Try to load parent
                        if _evict_if_needed(1):
                            operations.append(['load', p])
                            in_fast.add(p)
                            loaded_any = True
                            break
                
                if not loaded_any:
                    # Can't make progress on this chain right now
                    pass
                
                next_active.append(ci)
        
        active_chains = next_active
        iteration += 1
    
    # Phase 2: Ensure all outputs are in fast memory and store them
    for o in outputs:
        if time.time() - deadline[0] > _DEADLINE:
            break
        if o not in in_fast:
            if _evict_if_needed(1):
                operations.append(['load', o])
                in_fast.add(o)
        
        if o in in_fast:
            operations.append(['store', o])
            in_fast.remove(o)
    
    # Clean up
    while in_fast:
        v = in_fast.pop()
        operations.append(['delete', v])
    
    return operations

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    
    # Check that all outputs end up blue (stored or in fast memory)
    outputs = _get_outputs(dag)
    for o in outputs:
        # Outputs must be stored (removed from fast memory) or we need to check
        # The requirement is "all outputs must finish blue"
        # If we store them, they're in slow memory as blue - that should be fine
        pass
    
    return True

def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _naive_schedule_transfers(dag, fast_memory_budget):
    """Estimate naive schedule transfers."""
    all_v = _get_all_vertices(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    use_counts = _compute_use_counts(dag)
    
    # Naive: For each vertex computation, load all parents that aren't cached
    # Simple estimate: each vertex's parents need to be loaded once
    # Plus loading inputs and storing outputs
    transfers = len(inputs) + len(outputs)
    
    # Each non-input vertex needs its parents loaded
    for v in all_v:
        if v not in inputs:
            parents = _get_parents(dag, v)
            transfers += len(parents)
    
    return transfers

def _pebble_layered(dag, fast_memory_budget, deadline):
    """
    Layered pebbling: process DAG layer by layer.
    For each layer, load inputs, compute all vertices in layer, 
    then decide which to keep based on future layer needs.
    """
    start_time = time.time()
    
    if time.time() - deadline[0] > _DEADLINE:
        return []
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    outputs = _get_outputs(dag)
    outputs_set = set(outputs)
    
    # Compute layers (BFS from inputs)
    parents_map = {v: _get_parents(dag, v) for v in all_v}
    children_map = defaultdict(list)
    for v in all_v:
        for p in parents_map[v]:
            children_map[p].append(v)
    
    # Find inputs (no parents)
    inputs = [v for v in all_v if not parents_map[v]]
    
    # BFS to compute layers
    from collections import deque
    layer_of = {}
    queue = deque()
    for v in inputs:
        layer_of[v] = 0
        queue.append(v)
    
    while queue:
        v = queue.popleft()
        for c in children_map[v]:
            new_layer = layer_of[v] + 1
            if c not in layer_of or new_layer > layer_of[c]:
                layer_of[c] = new_layer
                queue.append(c)
    
    # Group by layer
    layers = defaultdict(list)
    for v in all_v:
        if v in layer_of:
            layers[layer_of[v]].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # Precompute: for each vertex, which future layers use it
    future_use = defaultdict(set)
    for v in all_v:
        for c in children_map[v]:
            if c in layer_of:
                future_use[v].add(layer_of[c])
    
    operations = []
    in_fast = set()
    
    def _evict_if_needed(needed=1):
        """Evict vertices to make space."""
        while len(in_fast) + needed > fast_memory_budget:
            if not in_fast:
                return False
            # Evict vertex with lowest future use
            best_to_evict = None
            best_score = float('inf')
            for v in in_fast:
                # Count how many future layers use this vertex
                future_layers = future_use.get(v, set())
                score = len(future_layers)
                if v in outputs_set:
                    score += 1000
                if score < best_score:
                    best_score = score
                    best_to_evict = v
            if best_to_evict:
                operations.append(['delete', best_to_evict])
                in_fast.remove(best_to_evict)
            else:
                return False
        return True
    
    # Process layer by layer
    for layer_idx in range(max_layer + 1):
        if time.time() - deadline[0] > _DEADLINE:
            break
        
        layer_vertices = layers.get(layer_idx, [])
        
        # Phase 1: Load all parents needed for this layer
        needed_parents = set()
        for v in layer_vertices:
            for p in parents_map[v]:
                needed_parents.add(p)
        
        for p in needed_parents:
            if p not in in_fast:
                if _evict_if_needed(1):
                    operations.append(['load', p])
                    in_fast.add(p)
        
        # Phase 2: Compute all vertices in this layer
        for v in layer_vertices:
            if time.time() - deadline[0] > _DEADLINE:
                break
            
            parents = parents_map[v]
            if all(p in in_fast for p in parents):
                if not _evict_if_needed(1):
                    continue
                operations.append(['compute', v])
                in_fast.add(v)
        
        # Phase 3: Decide what to keep
        # Keep vertices that will be used in future layers or are outputs
        to_evict = []
        for v in list(in_fast):
            if v in outputs_set:
                continue
            future_layers = future_use.get(v, set())
            # Only keep if used in a future layer we haven't processed yet
            if future_layers and max(future_layers) > layer_idx:
                continue
            # Not needed in future, can evict
            to_evict.append(v)
        
        for v in to_evict:
            operations.append(['delete', v])
            in_fast.remove(v)
    
    # Phase 4: Store all outputs
    for o in outputs:
        if time.time() - deadline[0] > _DEADLINE:
            break
        if o not in in_fast:
            if _evict_if_needed(1):
                operations.append(['load', o])
                in_fast.add(o)
        
        if o in in_fast:
            operations.append(['store', o])
            in_fast.remove(o)
    
    # Clean up
    while in_fast:
        v = in_fast.pop()
        operations.append(['delete', v])
    
    return operations

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    best_ops = None
    best_transfers = float('inf')
    
    # Try multiple strategies and pick the best
    strategies = []
    
    # Strategy 1: Chain-based round-robin
    try:
        ops = _pebble_chain_round_robin(dag, fast_memory_budget, deadline)
        if ops:
            if _validate_operations(ops, dag, fast_memory_budget):
                if len(ops) <= 20000:
                    transfers = _count_transfers(ops)
                    if transfers < best_transfers:
                        best_transfers = transfers
                        best_ops = ops
    except Exception:
        pass
    
    # Strategy 2: Layered pebbling
    try:
        ops = _pebble_layered(dag, fast_memory_budget, deadline)
        if ops:
            if _validate_operations(ops, dag, fast_memory_budget):
                if len(ops) <= 20000:
                    transfers = _count_transfers(ops)
                    if transfers < best_transfers:
                        best_transfers = transfers
                        best_ops = ops
    except Exception:
        pass
    
    # Strategy 3: Simple topological order with LRU
    try:
        ops_simple = []
        all_v = _get_all_vertices(dag)
        topo = _topological_order(dag)
        outputs = _get_outputs(dag)
        use_counts = _compute_use_counts(dag)
        
        in_fast = set()
        
        for v in topo:
            if time.time() - deadline[0] > _DEADLINE:
                break
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Evict something (simple: remove first added)
                        evict_target = None
                        for item in in_fast:
                            if item not in outputs:
                                evict_target = item
                                break
                        if evict_target:
                            ops_simple.append(['delete', evict_target])
                            in_fast.remove(evict_target)
                        else:
                            break
                    ops_simple.append(['load', p])
                    in_fast.add(p)
            ops_simple.append(['compute', v])
            in_fast.add(v)
        
        for o in outputs:
            if o not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_target = None
                    for item in in_fast:
                        if item not in outputs:
                            evict_target = item
                            break
                    if evict_target:
                        ops_simple.append(['delete', evict_target])
                        in_fast.remove(evict_target)
                    else:
                        break
                ops_simple.append(['load', o])
                in_fast.add(o)
            ops_simple.append(['store', o])
            in_fast.remove(o)
        
        if _validate_operations(ops_simple, dag, fast_memory_budget):
            if len(ops_simple) <= 20000:
                transfers = _count_transfers(ops_simple)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops_simple
    except Exception:
        pass
    
    # Return best result found
    if best_ops is not None:
        return best_ops
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
