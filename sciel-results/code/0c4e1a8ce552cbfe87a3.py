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
    Decompose DAG into chains using greedy matching.
    Returns list of chains, each chain is a list of vertices in topological order.
    """
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    parents_map = {v: _get_parents(dag, v) for v in all_v}
    children_map = defaultdict(list)
    for v in all_v:
        for p in parents_map[v]:
            children_map[p].append(v)
    
    # Find a maximum matching between parents and children
    # Greedy: for each vertex, try to match with one child
    matched_children = {}  # parent -> child
    matched_parents = {}   # child -> parent
    
    # Sort vertices by number of children (fewer first for better matching)
    sorted_vertices = sorted(all_v, key=lambda v: len(children_map[v]))
    
    for v in sorted_vertices:
        if v in matched_parents:
            continue
        for c in children_map[v]:
            if c not in matched_parents and v not in matched_children:
                matched_children[v] = c
                matched_parents[c] = v
                break
    
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
    
    return chains

def _pebble_chain_round_robin(dag, fast_memory_budget, deadline):
    """
    Pebble using chain decomposition and round-robin scheduling.
    Each chain is processed in parallel, cycling through chains.
    """
    start_time = time.time()
    
    if time.time() - deadline[0] > _DEADLINE:
        return []
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    outputs = _get_outputs(dag)
    use_counts = _compute_use_counts(dag)
    chains = _decompose_into_chains(dag)
    
    if not chains:
        return []
    
    operations = []
    in_fast = set()
    
    # Track position in each chain
    chain_positions = [0] * len(chains)
    
    # For each chain, track which vertices are already computed and in fast memory
    chain_computed = [set() for _ in chains]
    
    # Get all vertices in topological order for dependency checking
    topo = _topological_order(dag)
    topo_pos = {v: i for i, v in enumerate(topo)}
    
    def _can_compute(vertex):
        """Check if all parents of vertex are in fast memory."""
        parents = _get_parents(dag, vertex)
        return all(p in in_fast for p in parents)
    
    def _evict_if_needed(needed=1):
        """Evict vertices to make space, preferring vertices not needed soon."""
        while len(in_fast) + needed > fast_memory_budget:
            if not in_fast:
                return False
            # Find vertex to evict: prefer vertices with low future use
            best_to_evict = None
            best_score = float('inf')
            for v in in_fast:
                # Score: lower use count = better to evict
                # Also prefer evicting vertices not in outputs
                score = use_counts.get(v, 0)
                if v in outputs:
                    score += 1000  # Penalize evicting outputs
                if score < best_score:
                    best_score = score
                    best_to_evict = v
            if best_to_evict:
                operations.append(['delete', best_to_evict])
                in_fast.remove(best_to_evict)
            else:
                return False
        return True
    
    # Round-robin through chains
    max_iterations = len(all_v) * 2 + 100  # Safety bound
    iteration = 0
    active_chains = list(range(len(chains)))
    
    while active_chains and iteration < max_iterations:
        if time.time() - deadline[0] > _DEADLINE:
            break
        
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
                chain_computed[ci].add(v)
                chain_positions[ci] += 1
                
                # Decide whether to keep v in fast memory
                # Keep if: high use count, or it's an output, or memory has space
                keep = False
                if v in outputs:
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

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    # Try chain-based round-robin approach
    try:
        ops = _pebble_chain_round_robin(dag, fast_memory_budget, deadline)
        
        if ops:
            # Validate
            if _validate_operations(ops, dag, fast_memory_budget):
                if len(ops) <= 20000:
                    return ops
        
        # Fallback to simple topological order with LRU
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
                return ops_simple
        
    except Exception:
        pass
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
