def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with fast memory budget.
    
    dag: dict mapping vertex -> list of parent vertices
         Inputs have no parents (or empty parent list)
         Outputs are vertices that are not parents of any other vertex
    fast_memory_budget: int, number of red values that can be in fast memory
    
    Returns: list of [operation, vertex]
    Operations: 'load', 'store', 'compute', 'delete'
    """
    # Build adjacency and reverse adjacency
    # Determine inputs, outputs, and topological order
    
    # Get all vertices
    all_vertices = set(dag.keys())
    for parents in dag.values():
        for p in parents:
            all_vertices.add(p)
    
    # Build children map
    children = {v: [] for v in all_vertices}
    for v, parents in dag.items():
        for p in parents:
            if p in children:
                children[p].append(v)
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: 0 for v in all_vertices}
    for v, parents in dag.items():
        in_degree[v] = len(parents)
    
    # Wait, parents are already topologically ordered per the task
    # Let me just use the dag keys order if they're topologically sorted
    # Or do a proper topo sort
    
    from collections import deque
    
    in_deg = {v: 0 for v in all_vertices}
    for v, parents in dag.items():
        for p in parents:
            in_deg[v] += 1  # This is wrong, in_deg should be number of parents
    
    # Actually in_deg for topo sort = number of parents
    in_deg = {v: len(dag.get(v, [])) for v in all_vertices}
    
    queue = deque()
    for v in all_vertices:
        if in_deg[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't have all vertices, there's a cycle (shouldn't happen)
    if len(topo_order) != len(all_vertices):
        # Fallback: just use dag keys order
        topo_order = list(dag.keys())
    
    # Identify inputs (no parents) and outputs (no children)
    inputs = [v for v in all_vertices if len(dag.get(v, [])) == 0]
    outputs = [v for v in all_vertices if len(children.get(v, [])) == 0]
    
    # Count how many times each vertex is needed as a parent
    use_count = {v: 0 for v in all_vertices}
    for v, parents in dag.items():
        for p in parents:
            use_count[p] += 1
    
    # Now simulate the pebbling
    # State: which vertices are in fast memory (red), which are blue (on disk/external)
    # Initially: inputs are blue (on disk), everything else doesn't exist yet
    # Goal: outputs must be blue at the end
    
    # Fast memory: set of vertex IDs currently in fast memory
    fast_mem = set()
    # Values that have been computed and stored to disk (blue)
    on_disk = set(inputs)  # inputs start blue
    # Values that are currently in fast memory
    in_fast = set()
    
    operations = []
    
    # Track remaining uses for each vertex
    remaining_uses = dict(use_count)
    
    # Process in topological order
    for v in topo_order:
        parents = dag.get(v, [])
        
        # Load parents that are not in fast memory
        for p in parents:
            if p not in in_fast:
                # Need to load from disk
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        operations.append(['compute', v])
        # v is now in fast memory (red)
        in_fast.add(v)
        
        # Check if we exceed fast memory budget
        # If so, we need to store some values to disk
        # Strategy: store values that are no longer needed, or least recently used
        while len(in_fast) > fast_memory_budget:
            # Find a value to evict
            # Prefer evicting values that are no longer needed
            evicted = None
            for p in list(in_fast):
                # Don't evict v (just computed)
                if p == v:
                    continue
                if remaining_uses.get(p, 0) == 0:
                    evicted = p
                    break
            
            if evicted is None:
                # Evict the one with fewest remaining uses
                candidates = [p for p in in_fast if p != v]
                if candidates:
                    evicted = min(candidates, key=lambda x: remaining_uses.get(x, 0))
            
            if evicted is not None:
                operations.append(['store', evicted])
                in_fast.discard(evicted)
                on_disk.add(evicted)
        
        # Decrement remaining uses for parents
        for p in parents:
            remaining_uses[p] -= 1
    
    # At the end, all outputs must be blue (on disk)
    # Move any outputs from fast memory to disk
    for o in outputs:
        if o in in_fast:
            operations.append(['store', o])
            in_fast.discard(o)
            on_disk.add(o)
    
    # Delete all remaining fast memory values (optional, but keeps state clean)
    for v in list(in_fast):
        operations.append(['delete', v])
        in_fast.discard(v)
    
    return operations
