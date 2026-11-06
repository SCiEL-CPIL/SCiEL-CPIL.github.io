# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy: Use a priority-based greedy approach that minimizes transfers
    by keeping frequently-needed values in fast memory and only storing
    values that won't be needed soon.
    
    Operations:
    - load: bring value from slow to fast memory (cost 1)
    - store: move value from fast to slow memory (cost 1)
    - compute: compute a vertex's value from its parents (cost 0, but requires parents in fast)
    - delete: remove value from fast memory (cost 0)
    """
    start_time = time.time()
    deadline = start_time + 45.0  # 45s deadline with safety margin
    
    # Parse DAG structure
    # Assume dag has: 'vertices' (list of vertex ids), 'edges' (list of (parent, child)), 
    # 'inputs' (set of input vertices), 'outputs' (set of output vertices)
    # Or possibly: 'n' (num vertices), 'parents' (dict/list of parent lists), 'outputs' (list)
    
    if isinstance(dag, dict):
        # Try to extract structure from dict
        vertices = dag.get('vertices', dag.get('n', None))
        if vertices is not None and isinstance(vertices, int):
            n = vertices
            vertices_list = list(range(n))
        elif isinstance(vertices, (list, tuple)):
            vertices_list = list(vertices)
            n = len(vertices_list)
        else:
            # Try 'n' key
            n = dag.get('n', 0)
            vertices_list = list(range(n))
        
        # Get parent/child relationships
        parents = dag.get('parents', dag.get('in_edges', None))
        children = dag.get('children', dag.get('out_edges', None))
        edges = dag.get('edges', dag.get('arcs', None))
        
        inputs = set(dag.get('inputs', dag.get('sources', [])))
        outputs = set(dag.get('outputs', dag.get('sinks', [])))
        
        # Build adjacency if not provided
        if parents is None and children is None and edges is not None:
            parents = defaultdict(list)
            children = defaultdict(list)
            for edge in edges:
                if len(edge) == 2:
                    u, v = edge
                    children[u].append(v)
                    parents[v].append(u)
            parents = dict(parents)
            children = dict(children)
        elif parents is None and children is not None:
            parents = defaultdict(list)
            for v, ch in children.items():
                for u in ch:
                    parents[u].append(v)
            parents = dict(parents)
        elif children is None and parents is not None:
            children = defaultdict(list)
            for v, ps in parents.items():
                for u in ps:
                    children[u].append(v)
            children = dict(children)
    else:
        # Assume dag is a tuple or some other structure
        # Fallback: try to handle common formats
        if hasattr(dag, 'shape') and hasattr(dag, 'dtype'):
            # NumPy array - assume adjacency matrix
            import numpy as np
            adj = np.array(dag)
            n = adj.shape[0]
            vertices_list = list(range(n))
            parents = defaultdict(list)
            children = defaultdict(list)
            for i in range(n):
                for j in range(n):
                    if adj[i][j]:
                        children[i].append(j)
                        parents[j].append(i)
            parents = dict(parents)
            children = dict(children)
            inputs = set(i for i in range(n) if not parents[i])
            outputs = set(i for i in range(n) if not children[i])
        else:
            # Last resort: assume it's a list of parent lists
            parents_list = dag
            n = len(parents_list)
            vertices_list = list(range(n))
            parents = {i: parents_list[i] if i < len(parents_list) else [] for i in range(n)}
            children = defaultdict(list)
            for v, ps in parents.items():
                for u in ps:
                    children[u].append(v)
            children = dict(children)
            inputs = set(i for i in range(n) if not parents[i])
            outputs = set(i for i in range(n) if not children[i])
    
    # Ensure all vertices have entries in parents/children
    for v in vertices_list:
        if v not in parents:
            parents[v] = []
        if v not in children:
            children[v] = []
    
    # Identify inputs and outputs if not specified
    if not inputs:
        inputs = set(v for v in vertices_list if not parents[v])
    if not outputs:
        outputs = set(v for v in vertices_list if not children[v])
    
    # Topological order (should already be given, but verify/create)
    in_degree = {v: len(parents[v]) for v in vertices_list}
    topo_order = []
    queue = deque([v for v in vertices_list if in_degree[v] == 0])
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) != n:
        # Cyclic or issue - add remaining
        remaining = [v for v in vertices_list if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Count how many times each vertex is needed (fan-out consideration)
    # For each vertex, count how many children need it
    usage_count = {v: len(children[v]) for v in vertices_list}
    
    # Compute a priority score for each vertex based on usage
    # Higher usage = more important to keep in fast memory
    priority = {v: usage_count[v] + (10 if v in outputs else 0) for v in vertices_list}
    
    # State tracking
    # fast_memory: set of vertices currently in fast memory
    # slow_memory: set of vertices currently in slow memory (stored)
    # computed: set of vertices whose values have been computed
    # For inputs, they start in fast memory (blue)
    
    fast_memory = set(inputs)
    slow_memory = set()
    computed = set(inputs)  # Inputs are already available
    operations = []
    
    # Track which vertices we've processed in topological order
    # We need to compute each vertex when all its parents are in fast memory
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if v in computed:
            continue
        
        # Check if all parents are in fast memory
        parents_in_fast = all(p in fast_memory for p in parents[v])
        
        if parents_in_fast:
            # Compute v
            operations.append(['compute', v])
            computed.add(v)
            fast_memory.add(v)
            
            # Check if we're over budget
            while len(fast_memory) > fast_memory_budget:
                # Need to evict a vertex
                # Choose which to evict: prefer to keep high-priority vertices
                # Among non-input, non-output vertices, evict lowest priority
                candidates = [u for u in fast_memory if u not in inputs]
                
                # Don't evict outputs unless necessary
                non_output = [u for u in candidates if u not in outputs]
                if non_output:
                    to_evict = min(non_output, key=lambda x: priority[x])
                else:
                    # Must evict an output or input
                    to_evict = min(candidates, key=lambda x: priority[x])
                
                # Store to_evict to slow memory
                operations.append(['store', to_evict])
                slow_memory.add(to_evict)
                fast_memory.remove(to_evict)
        else:
            # Some parents not in fast memory - need to load them
            for p in parents[v]:
                if p not in fast_memory:
                    if p in slow_memory:
                        operations.append(['load', p])
                        slow_memory.remove(p)
                        fast_memory.add(p)
                    elif p in computed:
                        # p was computed but deleted from fast and not stored?
                        # This shouldn't happen if we always store before deleting
                        # But handle it: we need to recompute p's parents
                        pass
                    else:
                        # p not computed yet - this is a problem
                        # Should not happen in topological order
                        pass
            
            # Now try to compute v again
            parents_in_fast = all(p in fast_memory for p in parents[v])
            if parents_in_fast:
                operations.append(['compute', v])
                computed.add(v)
                fast_memory.add(v)
                
                while len(fast_memory) > fast_memory_budget:
                    candidates = [u for u in fast_memory if u not in inputs]
                    non_output = [u for u in candidates if u not in outputs]
                    if non_output:
                        to_evict = min(non_output, key=lambda x: priority[x])
                    else:
                        to_evict = min(candidates, key=lambda x: priority[x])
                    
                    operations.append(['store', to_evict])
                    slow_memory.add(to_evict)
                    fast_memory.remove(to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in fast_memory:
            if v in slow_memory:
                operations.append(['load', v])
                slow_memory.remove(v)
                fast_memory.add(v)
            elif v in computed:
                # Need to recompute
                pass
    
    return operations
# EVOLVE-BLOCK-END
