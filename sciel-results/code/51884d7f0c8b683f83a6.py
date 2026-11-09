"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

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

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal and all outputs finish blue."""
    in_fast = set()
    outputs = _get_outputs(dag)
    
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
    
    # Check all outputs are in fast memory (blue) at the end
    for o in outputs:
        if o not in in_fast:
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

def _pebble_lru(dag, fast_memory_budget, deadline):
    """
    LRU-based pebbling strategy.
    Uses topological order with LRU eviction when memory is full.
    """
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = set(_get_inputs(dag))
    outputs = set(_get_outputs(dag))
    use_counts = _compute_use_counts(dag)
    
    # Build children map
    children = defaultdict(list)
    for v in topo:
        for p in _get_parents(dag, v):
            children[p].append(v)
    
    operations = []
    in_fast = set()
    
    # Track remaining uses for each vertex
    remaining_uses = {}
    for v in topo:
        remaining_uses[v] = use_counts[v]
    
    # Process in topological order
    for v in topo:
        if time.time() - start_time > _DEADLINE:
            break
        
        parents = _get_parents(dag, v)
        
        if not parents:
            # Input vertex - starts blue, no action needed unless we need to load it
            # Inputs are already available, we just need them in fast memory when computing children
            continue
        
        # Ensure all parents are in fast memory
        for p in parents:
            if p not in in_fast:
                # Need to load p
                # If memory is full, evict using LRU-like strategy
                while len(in_fast) >= fast_memory_budget:
                    # Find vertex with lowest priority to evict
                    # Priority: fewer remaining uses, lower topological position (closer to being done)
                    best_evict = None
                    best_score = float('inf')
                    for u in in_fast:
                        # Score: remaining uses (lower is better to evict)
                        # But we must keep parents of current vertex if possible
                        if u in parents:
                            continue
                        score = remaining_uses.get(u, 0)
                        if score < best_score:
                            best_score = score
                            best_evict = u
                    
                    if best_evict is None:
                        # All remaining are parents, can't evict
                        # This shouldn't happen if budget is sufficient
                        break
                    
                    operations.append(['delete', best_evict])
                    in_fast.remove(best_evict)
                
                if len(in_fast) < fast_memory_budget:
                    operations.append(['load', p])
                    in_fast.add(p)
                else:
                    # Can't load, this is a problem
                    # Should not happen with proper budget
                    pass
        
        # Compute v
        if len(in_fast) >= fast_memory_budget:
            # Need to make room
            while len(in_fast) >= fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                for u in in_fast:
                    if u in parents:
                        continue
                    score = remaining_uses.get(u, 0)
                    if score < best_score:
                        best_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                operations.append(['delete', best_evict])
                in_fast.remove(best_evict)
        
        if len(in_fast) < fast_memory_budget:
            operations.append(['compute', v])
            in_fast.add(v)
        
        # Update remaining uses for parents
        for p in parents:
            if p in remaining_uses:
                remaining_uses[p] -= 1
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            # Load output
            while len(in_fast) >= fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                for u in in_fast:
                    if u in outputs:
                        continue
                    score = remaining_uses.get(u, 0)
                    if score < best_score:
                        best_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                operations.append(['delete', best_evict])
                in_fast.remove(best_evict)
            
            if len(in_fast) < fast_memory_budget:
                operations.append(['load', o])
                in_fast.add(o)
    
    return operations

def _pebble_divide_conquer(dag, fast_memory_budget, deadline):
    """
    Recursive divide-and-conquer pebbling strategy.
    For layered DAGs, split at the middle layer.
    """
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = set(_get_inputs(dag))
    outputs = set(_get_outputs(dag))
    use_counts = _compute_use_counts(dag)
    
    # Compute layer for each vertex
    layer = {}
    for v in topo:
        parents = _get_parents(dag, v)
        if not parents:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parents) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Strategy: For each layer, decide which vertices to keep in fast memory
    # Vertices with high use counts should be kept
    # Process layer by layer
    
    operations = []
    in_fast = set()
    
    # Process vertices in topological order, but manage memory per layer
    for v in topo:
        if time.time() - start_time > _DEADLINE:
            break
        
        parents = _get_parents(dag, v)
        
        if not parents:
            continue
        
        # Ensure parents are in fast memory
        for p in parents:
            if p not in in_fast:
                # Evict if necessary
                while len(in_fast) >= fast_memory_budget:
                    # Evict vertex with lowest remaining use count
                    best_evict = None
                    best_score = float('inf')
                    for u in in_fast:
                        if u in parents:
                            continue
                        score = use_counts.get(u, 0)
                        if score < best_score:
                            best_score = score
                            best_evict = u
                    
                    if best_evict is None:
                        break
                    
                    operations.append(['delete', best_evict])
                    in_fast.remove(best_evict)
                
                if len(in_fast) < fast_memory_budget:
                    operations.append(['load', p])
                    in_fast.add(p)
        
        # Compute v
        while len(in_fast) >= fast_memory_budget:
            best_evict = None
            best_score = float('inf')
            for u in in_fast:
                if u in parents:
                    continue
                score = use_counts.get(u, 0)
                if score < best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                break
            
            operations.append(['delete', best_evict])
            in_fast.remove(best_evict)
        
        if len(in_fast) < fast_memory_budget:
            operations.append(['compute', v])
            in_fast.add(v)
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            while len(in_fast) >= fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                for u in in_fast:
                    if u in outputs:
                        continue
                    score = use_counts.get(u, 0)
                    if score < best_score:
                        best_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                operations.append(['delete', best_evict])
                in_fast.remove(best_evict)
            
            if len(in_fast) < fast_memory_budget:
                operations.append(['load', o])
                in_fast.add(o)
    
    return operations

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    # Try LRU-based approach
    try:
        ops = _pebble_lru(dag, fast_memory_budget, deadline)
        
        if ops and len(ops) <= 20000:
            if _validate_operations(ops, dag, fast_memory_budget):
                return ops
        
        # Try divide-and-conquer approach
        ops = _pebble_divide_conquer(dag, fast_memory_budget, deadline)
        
        if ops and len(ops) <= 20000:
            if _validate_operations(ops, dag, fast_memory_budget):
                return ops
        
        # Fallback to simple topological approach
        ops_simple = []
        all_v = _get_all_vertices(dag)
        topo = _topological_order(dag)
        outputs = _get_outputs(dag)
        
        in_fast = set()
        
        for v in topo:
            if time.time() - deadline[0] > _DEADLINE:
                break
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    # Evict if necessary
                    while len(in_fast) >= fast_memory_budget:
                        # Evict least useful vertex
                        best_evict = None
                        best_score = float('inf')
                        use_counts = _compute_use_counts(dag)
                        for u in in_fast:
                            if u in parents:
                                continue
                            score = use_counts.get(u, 0)
                            if score < best_score:
                                best_score = score
                                best_evict = u
                        
                        if best_evict is None:
                            break
                        
                        ops_simple.append(['delete', best_evict])
                        in_fast.remove(best_evict)
                    
                    if len(in_fast) < fast_memory_budget:
                        ops_simple.append(['load', p])
                        in_fast.add(p)
            
            # Make room for compute
            while len(in_fast) >= fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                use_counts = _compute_use_counts(dag)
                for u in in_fast:
                    if u in parents:
                        continue
                    score = use_counts.get(u, 0)
                    if score < best_score:
                        best_score = score
                        best_evict = u
                
                if best_evict is None:
                    break
                
                ops_simple.append(['delete', best_evict])
                in_fast.remove(best_evict)
            
            if len(in_fast) < fast_memory_budget:
                ops_simple.append(['compute', v])
                in_fast.add(v)
        
        # Ensure outputs are in fast memory
        for o in outputs:
            if o not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    best_evict = None
                    best_score = float('inf')
                    use_counts = _compute_use_counts(dag)
                    for u in in_fast:
                        if u in outputs:
                            continue
                        score = use_counts.get(u, 0)
                        if score < best_score:
                            best_score = score
                            best_evict = u
                    
                    if best_evict is None:
                        break
                    
                    ops_simple.append(['delete', best_evict])
                    in_fast.remove(best_evict)
                
                if len(in_fast) < fast_memory_budget:
                    ops_simple.append(['load', o])
                    in_fast.add(o)
        
        if ops_simple and len(ops_simple) <= 20000:
            if _validate_operations(ops_simple, dag, fast_memory_budget):
                return ops_simple
        
    except Exception:
        pass
    
    # Final fallback: minimal valid schedule
    try:
        all_v = _get_all_vertices(dag)
        topo = _topological_order(dag)
        outputs = _get_outputs(dag)
        
        ops_final = []
        in_fast = set()
        
        for v in topo:
            parents = _get_parents(dag, v)
            if not parents:
                continue
            
            for p in parents:
                if p not in in_fast:
                    if len(in_fast) < fast_memory_budget:
                        ops_final.append(['load', p])
                        in_fast.add(p)
            
            if len(in_fast) < fast_memory_budget:
                ops_final.append(['compute', v])
                in_fast.add(v)
        
        for o in outputs:
            if o not in in_fast:
                if len(in_fast) < fast_memory_budget:
                    ops_final.append(['load', o])
                    in_fast.add(o)
        
        if ops_final and len(ops_final) <= 20000:
            if _validate_operations(ops_final, dag, fast_memory_budget):
                return ops_final
        
    except Exception:
        pass
    
    # Absolute fallback: return empty (will be scored as invalid, but better than timeout)
    return []
# EVOLVE-BLOCK-END
