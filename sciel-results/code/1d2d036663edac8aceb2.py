def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Parse DAG structure
    # dag should have: vertices, edges/parents, inputs, outputs
    # Based on typical structure: dag has 'parents' (topologically ordered), 'inputs', 'outputs'
    
    try:
        parents = dag.get('parents', dag.get('edges', []))
        inputs = dag.get('inputs', dag.get('input_vertices', []))
        outputs = dag.get('outputs', dag.get('output_vertices', []))
        n_vertices = dag.get('n_vertices', dag.get('num_vertices', None))
    except:
        # Fallback: try to infer structure
        pass
    
    # Get number of vertices
    if n_vertices is None:
        if parents:
            n_vertices = len(parents)
        else:
            return []
    
    # Convert inputs/outputs to sets
    input_set = set(inputs) if inputs else set()
    output_set = set(outputs) if outputs else set()
    
    # Build children map for efficient deletion
    children = [[] for _ in range(n_vertices)]
    for v in range(n_vertices):
        if v < len(parents):
            for p in parents[v]:
                if p is not None and p >= 0:
                    children[p].append(v)
    
    # Track remaining children count for each vertex
    remaining_children = [len(children[v]) for v in range(n_vertices)]
    
    # State: 'blue' (in fast memory), 'red' (on disk), 'uncomputed'
    state = [None] * n_vertices  # None = uncomputed, 'blue', 'red'
    for v in input_set:
        if v < n_vertices:
            state[v] = 'blue'
    
    fast_mem = set()  # Vertices currently in fast memory (blue)
    operations = []
    
    def transfer_count():
        return len(operations)
    
    def add_op(op, vertex):
        operations.append([op, vertex])
    
    # Process in topological order (0 to n_vertices-1)
    # For each vertex, ensure all parents are blue, compute, then delete parents if no more children
    for v in range(n_vertices):
        if time.time() - start_time > DEADLINE:
            break
        
        # Skip if already computed and blue
        if state[v] == 'blue':
            # Still need to handle deletion of parents
            for p in (parents[v] if v < len(parents) else []):
                if p is not None and p >= 0 and state[p] == 'blue':
                    remaining_children[p] -= 1
                    if remaining_children[p] == 0 and p not in output_set:
                        # Delete p
                        add_op('delete', p)
                        fast_mem.discard(p)
                        state[p] = None
            continue
        
        if state[v] == 'red':
            # Load it
            add_op('load', v)
            state[v] = 'blue'
            fast_mem.add(v)
            # Handle parent deletions
            for p in (parents[v] if v < len(parents) else []):
                if p is not None and p >= 0 and state[p] == 'blue':
                    remaining_children[p] -= 1
                    if remaining_children[p] == 0 and p not in output_set:
                        add_op('delete', p)
                        fast_mem.discard(p)
                        state[p] = None
            continue
        
        # Need to compute v
        # Ensure all parents are blue
        for p in (parents[v] if v < len(parents) else []):
            if p is not None and p >= 0:
                if state[p] == 'red':
                    add_op('load', p)
                    state[p] = 'blue'
                    fast_mem.add(p)
                elif state[p] is None:
                    # This shouldn't happen if we process in topological order
                    # and inputs start blue, but handle gracefully
                    pass
        
        # Compute v
        add_op('compute', v)
        state[v] = 'blue'
        fast_mem.add(v)
        
        # Now delete parents that are no longer needed
        for p in (parents[v] if v < len(parents) else []):
            if p is not None and p >= 0 and state[p] == 'blue':
                remaining_children[p] -= 1
                if remaining_children[p] == 0 and p not in output_set:
                    add_op('delete', p)
                    fast_mem.discard(p)
                    state[p] = None
    
    # Ensure all outputs are blue (store if needed - but they should already be blue)
    # Actually, "store" might mean writing to output storage, separate from fast memory
    # Let me reconsider: outputs must "finish blue" meaning in fast memory
    # So we just need them to be blue at the end
    
    # If we have memory pressure, we might need to delete some non-output values
    # But the above strategy already deletes intermediate values
    
    return operations
