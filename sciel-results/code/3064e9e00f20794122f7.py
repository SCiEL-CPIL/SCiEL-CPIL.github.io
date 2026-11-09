"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

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

def _compute_layers(dag, topo):
    """Compute the layer (depth) of each vertex."""
    layer = {}
    for v in topo:
        parents = _get_parents(dag, v)
        if not parents:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parents) + 1
    return layer

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
    return True

def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _naive_schedule_transfers(dag, fast_memory_budget):
    """Estimate naive schedule transfers."""
    all_v = _get_all_vertices(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    
    # Naive: load all inputs, compute all, store all outputs
    # This is an estimate
    transfers = len(inputs) + len(outputs)
    return transfers

def _pebble_cost_aware(dag, fast_memory_budget, deadline):
    """
    Pebbling strategy using cost-aware eviction.
    """
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)
    
    # Compute layers
    layer = _compute_layers(dag, topo)
    
    # Pre-compute children and future usage
    children = defaultdict(list)
    for v in topo:
        parents = _get_parents(dag, v)
        for p in parents:
            children[p].append(v)
    
    # For each vertex, compute how many of its children are in higher layers (future uses)
    future_need = {}
    for v in topo:
        count = 0
        for c in children[v]:
            if layer[c] > layer[v]:
                count += 1
        future_need[v] = count
    
    operations = []
    in_fast = set()
    
    # Track which vertices have been fully processed (all children done)
    remaining_children = {v: len(children[v]) for v in topo}
    
    # Helper to estimate recompute cost of a vertex
    def get_recompute_cost(v):
        if not _get_parents(dag, v):
            return 1 # Input, cost is 1 load
        # Cost is 1 (compute) + sum of costs of parents not in memory?
        # Simplified: 1 + number of parents not in fast memory
        parents = _get_parents(dag, v)
        cost = 1
        for p in parents:
            if p not in in_fast:
                cost += 1 # Assume we have to load/recompute parent
        return cost

    def evict_one():
        if not in_fast:
            return
        candidates = list(in_fast)
        
        def sort_key(v):
            # We want to evict the vertex with the LOWEST "Value"
            # Value = Future Need * Recompute Cost
            # If Future Need is 0, Value is 0 (best to evict)
            
            fn = future_need.get(v, 0)
            
            # If remaining_children is 0, it won't be needed again in this pass?
            # Actually, future_need already accounts for "higher layers".
            # If we are in topological order, and we've passed all children, it's 0.
            
            rc = remaining_children.get(v, 0)
            
            # If rc == 0, it is definitely done. Value = 0.
            if rc == 0:
                return (0, 0, layer.get(v, 0)) # Priority 0: Done. Secondary: Layer (evict older? or newer? usually doesn't matter if done)
            
            # Estimate cost
            cost = get_recompute_cost(v)
            
            # Value = fn * cost
            # We want to minimize this.
            val = fn * cost
            
            # Ties:
            # 1. Prefer evicting Inputs? (Cost 1 vs Cost > 1). If fn is same, evict higher cost?
            #    If fn is same, and cost is same, evict the one with lower layer (older) to free up "space" for newer stuff?
            #    Actually, if fn is same, we don't care much.
            
            # If fn is 0 but rc > 0 (shouldn't happen if future_need is correct for topo order), treat as low value.
            
            return (val, cost, layer.get(v, 0))
            
        # Sort by key, evict the first one (lowest value)
        candidates.sort(key=sort_key)
        v_to_evict = candidates[0]
        operations.append(['delete', v_to_evict])
        in_fast.remove(v_to_evict)

    # Process each vertex in topological order
    for v in topo:
        if time.time() - deadline[0] > _DEADLINE:
            break
            
        parents = _get_parents(dag, v)
        
        # Ensure all parents are in fast memory
        for p in parents:
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    evict_one()
                
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        if len(in_fast) >= fast_memory_budget:
            evict_one()
        
        operations.append(['compute', v])
        in_fast.add(v)
        
        # Update remaining_children for parents
        for p in parents:
            if p in remaining_children:
                remaining_children[p] -= 1

    # Phase 2: Ensure all outputs are in fast memory
    for o in outputs:
        if time.time() - deadline[0] > _DEADLINE:
            break
        if o not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_one()
            operations.append(['load', o])
            in_fast.add(o)
        
        # Store the output
        operations.append(['store', o])
        in_fast.remove(o)
        
    return operations

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    try:
        ops = _pebble_cost_aware(dag, fast_memory_budget, deadline)
        
        if not ops:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        # Validate
        if _validate_operations(ops, dag, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # Fallback to baseline if invalid
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
                
    except Exception:
        pass
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
