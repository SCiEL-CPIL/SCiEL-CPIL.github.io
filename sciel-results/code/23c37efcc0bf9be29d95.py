def pebble(dag, fast_memory_budget):
    """
    DAG pebbling algorithm that minimizes transfers.
    
    dag: dict with 'vertices', 'parents' (topologically ordered), 'outputs'
    fast_memory_budget: int, number of fast memory slots
    returns: list of [operation, vertex]
    """
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    # Parse DAG structure
    vertices = list(dag['vertices'])
    parents = dag['parents']  # parents[v] = list of parent vertices
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order (vertices are already topologically ordered per task)
    topo_order = vertices
    
    # Identify input vertices (no parents)
    inputs = [v for v in vertices if len(parents[v]) == 0]
    
    # Strategy: 
    # 1. Load all inputs
    # 2. Process in topological order, computing vertices
    # 3. Use fast memory budget wisely - keep outputs and recent intermediates
    # 4. Store intermediate results that will be needed later
    
    # Simple but effective strategy:
    # - Load all inputs
    # - For each vertex in topo order:
    #   - If all parents are in fast memory, compute
    #   - If not, we need to reload some parents
    #   - After computing, decide whether to keep in fast memory or store
    
    # Let's use a simpler approach: process layer by layer
    # Find layers
    layer_of = {}
    for v in topo_order:
        if not parents[v]:
            layer_of[v] = 0
        else:
            layer_of[v] = max(layer_of[p] for p in parents[v]) + 1
    
    max_layer = max(layer_of.values()) if layer_of else 0
    
    # Group by layer
    layers = {}
    for v in topo_order:
        layers.setdefault(layer_of[v], []).append(v)
    
    # For each layer, we need to have all parents available
    # Strategy: load inputs, then for each layer compute all vertices
    # Keep results in fast memory if budget allows, else store and reload
    
    operations = []
    in_fast = set()  # vertices currently in fast memory
    
    # Load all inputs
    for v in inputs:
        operations.append(['load', v])
        in_fast.add(v)
    
    # Process layers
    for layer in range(max_layer + 1):
        if layer not in layers:
            continue
        layer_vertices = layers[layer]
        
        # Check if all parents are in fast memory
        # If not, we may need to reload some
        for v in layer_vertices:
            # Ensure all parents are in fast memory
            for p in parents[v]:
                if p not in in_fast:
                    # Need to reload parent
                    operations.append(['load', p])
                    in_fast.add(p)
            
            # Compute v
            operations.append(['compute', v])
            in_fast.add(v)
            
            # Check time
            if time.time() - start_time > DEADLINE:
                break
        
        # After processing a layer, decide what to keep
        # Keep: inputs (for potential reuse), outputs, and vertices needed by next layers
        
        # Determine which vertices from this layer are needed by future layers
        needed_by_future = set()
        for v in layer_vertices:
            for c in children[v]:
                if layer_of[c] > layer:
                    needed_by_future.add(v)
                    break
        
        # Also keep outputs
        outputs_in_layer = set(v for v in layer_vertices if v in outputs)
        
        # Must keep: outputs + needed_by_future
        must_keep = outputs_in_layer | needed_by_future
        
        # Can we afford to keep them all?
        # Count how many are in fast memory from this layer
        layer_in_fast = [v for v in layer_vertices if v in in_fast]
        
        if len(layer_in_fast) > fast_memory_budget:
            # Need to evict some
            # Keep must_keep, evict others
            keep = set(layer_in_fast)
            must_keep_in_fast = must_keep & keep
            
            # If must_keep fits, keep it and evict the rest
            if len(must_keep_in_fast) <= fast_memory_budget:
                to_evict = keep - must_keep_in_fast
                for v in to_evict:
                    operations.append(['store', v])
                    in_fast.discard(v)
            else:
                # Even must_keep doesn't fit - this is problematic
                # Keep as many as possible
                to_keep = list(must_keep_in_fast)[:fast_memory_budget]
                to_evict = keep - set(to_keep)
                for v in to_evict:
                    operations.append(['store', v])
                    in_fast.discard(v)
    
    # Ensure all outputs are in fast memory (blue) at the end
    for v in outputs:
        if v not in in_fast:
            operations.append(['load', v])
            in_fast.add(v)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Return what we have
        pass
    
    # Truncate if too many operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
