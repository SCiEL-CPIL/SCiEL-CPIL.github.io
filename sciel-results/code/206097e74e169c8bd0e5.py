# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # First, try to understand the dag structure
    # The task says "dag has topologically ordered parents and output vertices"
    
    # Try multiple parsing strategies
    def try_parse(dag):
        """Returns (vertices, parents_dict, outputs) or None."""
        if isinstance(dag, dict):
            # Strategy 1: dag = {vertex: [parents]}
            if all(isinstance(v, (list, tuple, set)) for v in dag.values()):
                vertices = list(dag.keys())
                parents = {k: list(v) for k, v in dag.items()}
                # Find outputs: vertices not in any parent list
                referenced = set()
                for ps in parents.values():
                    for p in ps:
                        referenced.add(p)
                outputs = [v for v in vertices if v not in referenced]
                if outputs:
                    return vertices, parents, outputs
            
            # Strategy 2: dag has 'vertices', 'parents', 'outputs' keys
            if 'vertices' in dag and 'parents' in dag:
                vertices = list(dag['vertices'])
                parents = {k: list(v) for k, v in dag['parents'].items()}
                outputs = list(dag.get('outputs', []))
                if not outputs:
                    referenced = set()
                    for ps in parents.values():
                        for p in ps:
                            referenced.add(p)
                    outputs = [v for v in vertices if v not in referenced]
                return vertices, parents, outputs
            
            # Strategy 3: dag has 'nodes' and 'edges'
            if 'nodes' in dag and 'edges' in dag:
                vertices = list(dag['nodes'])
                edges = dag['edges']
                parents = {v: [] for v in vertices}
                for e in edges:
                    if len(e) == 2:
                        u, w = e
                        if w in parents:
                            parents[w].append(u)
                outputs = list(dag.get('outputs', []))
                if not outputs:
                    has_children = set()
                    for ps in parents.values():
                        for p in ps:
                            has_children.add(p)
                    outputs = [v for v in vertices if v not in has_children]
                return vertices, parents, outputs
        
        elif isinstance(dag, (list, tuple)):
            # Strategy: list of (vertex, [parents]) or similar
            if len(dag) > 0:
                first = dag[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    vertices = []
                    parents = {}
                    for i, item in enumerate(dag):
                        vid = item[0]
                        pars = list(item[1]) if isinstance(item[1], (list, tuple, set)) else [item[1]]
                        vertices.append(vid)
                        parents[i] = pars
                    referenced = set()
                    for ps in parents.values():
                        for p in ps:
                            referenced.add(p)
                    outputs = [i for i in range(len(dag)) if i not in referenced]
                    return vertices, parents, outputs
                elif isinstance(first, (list, tuple)) and len(first) == 1:
                    # List of [vertex] with no parents? Unlikely
                    pass
                else:
                    # List of vertex IDs, no parent info
                    # Can't do much without parent info
                    pass
        
        return None
    
    parsed = try_parse(dag)
    if parsed is None:
        # Fallback to baseline
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    vertices, parents_dict, outputs = parsed
    n = len(vertices)
    
    # Build children map
    children = {v: [] for v in range(n)}
    for v in range(n):
        for p in parents_dict.get(v, []):
            if p in children:
                children[p].append(v)
    
    # In-degree (number of parents)
    indeg = {v: len(parents_dict.get(v, [])) for v in range(n)}
    
    # Topological order (already given, but let's verify)
    topo_order = list(range(n))
    
    # Naive cost: load each input, compute each vertex, store each output
    # Inputs are vertices with indeg 0
    num_inputs = sum(1 for v in range(n) if indeg[v] == 0)
    naive_cost = num_inputs + n + len(outputs)  # loads + computes + stores
    
    # Strategy: Layered pebbling
    # Compute layers
    layer = {}
    for v in range(n):
        ps = parents_dict.get(v, [])
        if not ps:
            layer[v] = 0
        else:
            layer[v] = max(layer.get(p, 0) for p in ps) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group by layer
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # Pebbling algorithm:
    # - Start with all inputs in fast memory (they're blue)
    # - Process layer by layer
    # - Keep intermediate results in fast memory when possible
    # - When memory is full, evict values that are least likely to be needed
    
    fast_mem = set()
    computed = set()
    operations = []
    
    # Initialize: all inputs are in fast memory
    for v in range(n):
        if indeg[v] == 0:
            fast_mem.add(v)
            computed.add(v)
    
    # Process layers in order
    for l in range(1, max_layer + 1):
        if time.time() > deadline:
            break
        
        layer_vertices = layers[l]
        
        # Sort by urgency: vertices that are needed by more uncomputed descendants
        def urgency(v):
            count = 0
            stack = [v]
            visited = set()
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                for c in children.get(u, []):
                    if c not in computed and c not in visited:
                        count += 1
                        stack.append(c)
            return count
        
        # Sort by urgency (descending)
        layer_vertices.sort(key=urgency, reverse=True)
        
        for v in layer_vertices:
            if time.time() > deadline:
                break
            
            ps = parents_dict.get(v, [])
            
            # Ensure all parents are in fast memory
            for p in ps:
                if p not in fast_mem:
                    # Load parent
                    operations.append(['load', p])
                    fast_mem.add(p)
            
            # If over budget, evict
            while len(fast_mem) > fast_memory_budget:
                # Don't evict parents of v
                candidates = [x for x in fast_mem if x not in ps]
                if not candidates:
                    break
                
                # Evict least urgent
                min_urg = float('inf')
                evict_v = None
                for x in candidates:
                    u = urgency(x)
                    if u < min_urg:
                        min_urg = u
                        evict_v = x
                
                if evict_v is not None:
                    operations.append(['store', evict_v])
                    fast_mem.discard(evict_v)
                else:
                    break
            
            # Compute v
            operations.append(['compute', v])
            computed.add(v)
            fast_mem.add(v)
    
    # Ensure outputs are in fast memory
    for out in outputs:
        if out not in fast_mem:
            operations.append(['load', out])
            fast_mem.add(out)
    
    # Truncate if too many operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
