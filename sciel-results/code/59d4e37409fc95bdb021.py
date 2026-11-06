# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG using a depth-first traversal with stack-based memory management.
    
    The DAG is processed in topological order. We use a stack to track the
    active computation path and evict values from fast memory as soon as
    they are no longer needed, minimizing the number of transfers.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Extract DAG structure
    vertices = dag.get('vertices', [])
    parents_map = dag.get('parents', {})
    inputs = dag.get('inputs', [])
    outputs = dag.get('outputs', [])
    layers = dag.get('layers', None)
    
    if not vertices:
        return []
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    # Determine topological order
    if layers:
        topo_order = [v for layer in layers for v in layer]
    else:
        topo_order = list(vertices)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Compute last_use for each vertex
    # last_use[v] = index in topo_order of the last vertex that requires v
    last_use = {v: 0 for v in vertices}
    
    # Process in reverse topological order to compute last_use
    for idx in range(len(topo_order) - 1, -1, -1):
        v = topo_order[idx]
        if v in output_set:
            last_use[v] = len(topo_order)  # Must be kept until end
        else:
            max_child_use = 0
            for c in children[v]:
                if c in last_use:
                    max_child_use = max(max_child_use, last_use[c])
            last_use[v] = max_child_use
    
    # Process in topological order
    fast_mem = set()
    ops = []
    
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        if v in input_set:
            # Load input if not already in fast memory
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
        else:
            # Ensure all parents are in fast memory
            for p in parents_map.get(v, []):
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
            
            # Compute v
            ops.append(['compute', v])
            fast_mem.add(v)
        
        # Evict values that are no longer needed
        # A value u can be evicted if last_use[u] <= idx and u is not an output
        to_evict = []
        for u in list(fast_mem):
            if u in output_set:
                continue
            if last_use.get(u, 0) <= idx:
                to_evict.append(u)
        
        # Sort by last_use to evict earliest-used first
        to_evict.sort(key=lambda u: last_use.get(u, 0))
        
        for u in to_evict:
            if u in fast_mem:
                ops.append(['store', u])
                fast_mem.remove(u)
                ops.append(['delete', u])
        
        # Enforce fast memory budget
        if len(fast_mem) > fast_memory_budget:
            # Evict values with earliest last_use that are not outputs
            candidates = [u for u in fast_mem if u not in output_set]
            candidates.sort(key=lambda u: last_use.get(u, 0))
            while len(fast_mem) > fast_memory_budget and candidates:
                u = candidates.pop(0)
                if u in fast_mem:
                    ops.append(['store', u])
                    fast_mem.remove(u)
                    ops.append(['delete', u])
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            ops.append(['load', v])
            fast_mem.add(v)
    
    return ops
# EVOLVE-BLOCK-END
