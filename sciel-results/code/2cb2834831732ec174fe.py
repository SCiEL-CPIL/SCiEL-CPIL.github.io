def pebble(dag, fast_memory_budget):
    # Extract DAG structure
    if hasattr(dag, 'parents'):
        parents = dag.parents
    elif 'parents' in dag:
        parents = dag['parents']
    else:
        raise ValueError("Cannot determine DAG structure")
    
    if hasattr(dag, 'inputs'):
        inputs = set(dag.inputs)
    elif 'inputs' in dag:
        inputs = set(dag['inputs'])
    else:
        inputs = set()
    
    if hasattr(dag, 'outputs'):
        outputs = set(dag.outputs)
    elif 'outputs' in dag:
        outputs = set(dag['outputs'])
    else:
        outputs = set()
    
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif 'vertices' in dag:
        vertices = list(dag['vertices'])
    else:
        vertices = list(parents.keys())
    
    # Compute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological sort
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] = in_degree.get(v, 0) + 1
    
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Process in topological order
    in_memory = set()
    operations = []
    
    def ensure_in_memory(v):
        """Ensure vertex v is in memory by loading or recomputing as needed."""
        if v in in_memory:
            return
        if v in inputs:
            operations.append(['load', v])
            in_memory.add(v)
        else:
            # Need to compute v, ensure all parents are in memory first
            for p in parents.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            in_memory.add(v)
    
    for v in topo_order:
        ensure_in_memory(v)
        
        if v in outputs:
            operations.append(['store', v])
            in_memory.discard(v)
        else:
            # Manage memory: delete vertices if over budget
            while len(in_memory) > fast_memory_budget:
                # Delete the vertex with the fewest children (static heuristic)
                best_to_delete = min(in_memory, key=lambda u: len(children.get(u, [])))
                operations.append(['delete', best_to_delete])
                in_memory.discard(best_to_delete)
    
    return operations
