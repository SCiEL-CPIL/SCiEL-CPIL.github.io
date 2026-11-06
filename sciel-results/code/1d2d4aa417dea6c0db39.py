"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize load/store transfers.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure robustly
    vertices = []
    parents = {}
    inputs = set()
    outputs = set()
    
    # Try to extract vertices
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif hasattr(dag, 'nodes'):
        vertices = list(dag.nodes)
    elif isinstance(dag, dict):
        vertices = list(dag.keys())
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
    else:
        # Try to infer from other attributes
        if hasattr(dag, 'num_vertices'):
            vertices = list(range(dag.num_vertices))
        elif hasattr(dag, 'n'):
            vertices = list(range(dag.n))
    
    # Extract parents
    if hasattr(dag, 'parents') and isinstance(dag.parents, dict):
        parents = {v: list(dag.parents.get(v, [])) for v in vertices}
    elif hasattr(dag, 'in_edges'):
        parents = {v: list(dag.in_edges.get(v, [])) for v in vertices}
    elif hasattr(dag, 'edges'):
        parents = defaultdict(list)
        edges = dag.edges
        for edge in edges:
            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                p, c = edge[0], edge[1]
                parents[c].append(p)
            elif hasattr(edge, 'src') and hasattr(edge, 'dst'):
                parents[edge.dst].append(edge.src)
        for v in vertices:
            if v not in parents:
                parents[v] = []
    elif isinstance(dag, dict):
        # dag[v] might be list of parents
        parents = {v: list(dag.get(v, [])) for v in vertices}
    else:
        parents = {v: [] for v in vertices}
    
    # Extract inputs and outputs
    if hasattr(dag, 'inputs'):
        inputs = set(dag.inputs)
    elif hasattr(dag, 'sources'):
        inputs = set(dag.sources)
    else:
        inputs = {v for v in vertices if not parents.get(v, [])}
    
    if hasattr(dag, 'outputs'):
        outputs = set(dag.outputs)
    elif hasattr(dag, 'sinks'):
        outputs = set(dag.sinks)
    else:
        children = defaultdict(list)
        for v in vertices:
            for p in parents.get(v, []):
                children[p].append(v)
        outputs = {v for v in vertices if not children.get(v, [])}
    
    # Compute topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Fallback: just use all vertices
        topo_order = list(vertices)
    
    # Track remaining children
    remaining_children = {v: list(children[v]) for v in vertices}
    
    # State
    in_fast_mem = set()
    operations = []
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        # Load parents into fast memory
        for p in parents.get(v, []):
            if p not in in_fast_mem:
                operations.append(['load', p])
                in_fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_fast_mem.add(v)
        
        # Update remaining children
        for p in parents.get(v, []):
            if v in remaining_children[p]:
                remaining_children[p].remove(v)
        
        # Delete parents that are no longer needed
        for p in parents.get(v, []):
            if not remaining_children[p] and p not in outputs:
                if p in in_fast_mem:
                    operations.append(['delete', p])
                    in_fast_mem.discard(p)
        
        # Evict if over budget
        while len(in_fast_mem) > fast_memory_budget:
            best_evict = None
            best_score = float('inf')
            
            for x in in_fast_mem:
                if x in outputs:
                    continue
                score = len(remaining_children[x])
                if score < best_score:
                    best_score = score
                    best_evict = x
            
            if best_evict is None:
                best_evict = next(iter(in_fast_mem))
            
            operations.append(['delete', best_evict])
            in_fast_mem.discard(best_evict)
    
    # Store outputs at the end
    for v in outputs:
        if v in in_fast_mem:
            operations.append(['store', v])
            in_fast_mem.discard(v)
    
    return operations

# EVOLVE-BLOCK-END
