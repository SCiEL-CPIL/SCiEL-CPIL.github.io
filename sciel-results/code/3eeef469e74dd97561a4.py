"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize transfers.
    """
    # Extract DAG structure
    n = dag.get('n', dag.get('num_vertices', None))
    parents = dag.get('parents', None)
    inputs = dag.get('inputs', None)
    outputs = dag.get('outputs', None)
    layers = dag.get('layers', None)
    
    if n is None:
        raise ValueError("Cannot determine number of vertices")
    
    if parents is None:
        if 'adj' in dag:
            children = dag['adj']
            parents = [[] for _ in range(n)]
            for v in range(n):
                for u in children[v]:
                    parents[u].append(v)
        elif 'edges' in dag:
            edges = dag['edges']
            parents = [[] for _ in range(n)]
            for u, v in edges:
                parents[v].append(u)
        else:
            raise ValueError("Cannot determine parents")
    
    if inputs is None:
        inputs = [v for v in range(n) if len(parents[v]) == 0]
    
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    if outputs is None:
        outputs = [v for v in range(n) if len(children[v]) == 0]
    
    # Compute layers
    if layers is None:
        in_degree = [0] * n
        for v in range(n):
            for p in parents[v]:
                in_degree[v] += 1
        
        topo_order = []
        q = deque([v for v in range(n) if in_degree[v] == 0])
        while q:
            v = q.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    q.append(c)
        
        layer = [0] * n
        for v in topo_order:
            if parents[v]:
                layer[v] = max(layer[p] for p in parents[v]) + 1
            else:
                layer[v] = 0
        
        max_layer = max(layer) if layer else 0
        layers = [[] for _ in range(max_layer + 1)]
        for v in range(n):
            layers[layer[v]].append(v)
    
    fast_memory = set(inputs)
    operations = []
    
    num_layers = len(layers)
    
    for li in range(num_layers):
        current_layer = layers[li]
        if not current_layer:
            continue
        
        for v in current_layer:
            for p in parents[v]:
                if p not in fast_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
            operations.append(['compute', v])
            fast_memory.add(v)
        
        for v in list(fast_memory):
            needed = False
            for c in children[v]:
                if layer[c] > li:
                    needed = True
                    break
            if v in outputs:
                needed = True
            if not needed:
                operations.append(['delete', v])
                fast_memory.discard(v)
    
    for v in outputs:
        if v not in fast_memory:
            for p in parents[v]:
                if p not in fast_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
            operations.append(['compute', v])
            fast_memory.add(v)
    
    return operations

# EVOLVE-BLOCK-END
