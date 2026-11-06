# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure.
    2. Use a greedy topological schedule with smart eviction.
    3. Minimize load/store operations by keeping high-reuse values in fast memory.
    4. For layered DAGs, exploit layer structure.
    """
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG - try multiple formats
    parents = {}
    vertices = []
    outputs = set()
    
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        vertices = list(parents.keys()) if parents else list(range(len(dag.get('vertices', []))))
        outputs = set(dag.get('outputs', []))
    elif hasattr(dag, 'parents'):
        parents = dict(dag.parents) if hasattr(dag.parents, '__iter__') else {}
        vertices = list(dag.vertices) if hasattr(dag, 'vertices') else list(parents.keys())
        outputs = set(dag.outputs) if hasattr(dag, 'outputs') else set()
    elif isinstance(dag, list):
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                p = item[1] if isinstance(item[1], list) else list(item[1])
                parents[v] = p
        vertices = list(parents.keys())
        outputs = set()
    else:
        vertices = list(range(10))
        parents = {}
        outputs = set()
    
    n_vertices = len(vertices)
    if n_vertices == 0:
        return []
    
    # Build children mapping
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Identify inputs
    inputs = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # Compute in-degree and topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    queue = deque(inputs)
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) < n_vertices:
        # Cycle or issue, use original order
        topo_order = vertices
    
    # Compute reuse count: how many children each vertex has
    reuse_count = {v: len(children.get(v, [])) for v in vertices}
    
    # Determine which vertices are "important" (outputs or high-reuse)
    importance = {}
    for v in vertices:
        imp = reuse_count[v]
        if v in outputs:
            imp += 10000
        importance[v] = imp
    
    # Greedy pebbling
    fast_mem = {}  # vertex -> True
    slow_store = set()
    computed = set()
    ops = []
    
    def is_deadline_exceeded():
        return (time.time() - start_time) > DEADLINE
    
    def evict_least_useful():
        """Evict the vertex with lowest importance from fast memory"""
        if not fast_mem:
            return None
        
        best_v = None
        best_imp = float('inf')
        for v in fast_mem:
            if v in outputs:
                continue  # Never evict outputs
            imp = importance[v]
            # Bonus for vertices whose children are all computed
            remaining_children = sum(1 for c in children.get(v, []) if c not in computed)
            if remaining_children == 0:
                imp -= 5000  # Strong incentive to evict fully-consumed values
            if imp < best_imp:
                best_imp = imp
                best_v = v
        
        if best_v is not None:
            if best_v in slow_store:
                del fast_mem[best_v]
            else:
                ops.append(['store', best_v])
                del fast_mem[best_v]
                slow_store.add(best_v)
        return best_v
    
    def ensure_in_fast(v):
        """Ensure v is in fast memory"""
        if v in fast_mem:
            return
        if len(fast_mem) < fast_memory_budget:
            if v in slow_store:
                ops.append(['load', v])
                fast_mem[v] = True
            else:
                fast_mem[v] = True
        else:
            evict_least_useful()
            if v in slow_store:
                ops.append(['load', v])
                fast_mem[v] = True
            else:
                fast_mem[v] = True
    
    # Initialize: put inputs in fast memory
    for v in inputs:
        if len(fast_mem) < fast_memory_budget:
            fast_mem[v] = True
        else:
            evict_least_useful()
            fast_mem[v] = True
    
    # Process in topological order
    for v in topo_order:
        if is_deadline_exceeded():
            break
        
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            ensure_in_fast(p)
        
        # Compute
        ops.append(['compute', v])
        computed.add(v)
        fast_mem[v] = True
        
        # Try to evict fully-consumed parents
        for p in parents.get(v, []):
            remaining = sum(1 for c in children.get(p, []) if c not in computed)
            if remaining == 0 and p not in outputs:
                if p in fast_mem:
                    if p in slow_store:
                        del fast_mem[p]
                    else:
                        ops.append(['store', p])
                        del fast_mem[p]
                        slow_store.add(p)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            ensure_in_fast(v)
    
    return ops
# EVOLVE-BLOCK-END
