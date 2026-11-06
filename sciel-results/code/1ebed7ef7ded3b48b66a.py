"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dictionary or object with:
        - vertices: list of vertex ids
        - parents: dict mapping vertex -> list of parent vertices (topologically ordered)
        - inputs: list of input vertex ids
        - outputs: list of output vertex ids
        - Or similar structure
    
    fast_memory_budget: int, number of red values that can be held in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load'/'store'/'compute'/'delete'
    """
    # Try to extract DAG structure - handle multiple possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', list(dag.get('parents', {}).keys()))
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    elif hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
        parents = dict(dag.parents) if hasattr(dag, 'parents') else {}
        inputs = list(getattr(dag, 'inputs', []))
        outputs = list(getattr(dag, 'outputs', []))
    else:
        # Fallback: try to infer structure
        vertices = list(dag.keys()) if isinstance(dag, dict) else list(dag)
        parents = {}
        inputs = []
        outputs = []
    
    if not vertices:
        return []
    
    # Build adjacency and topological order
    # parents[v] = list of parent nodes of v
    # children[v] = list of child nodes of v
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p not in children:
                children[p] = []
            children[p].append(v)
    
    # Identify input vertices (no parents) and output vertices (no children)
    if not inputs:
        inputs = [v for v in vertices if not parents.get(v, [])]
    if not outputs:
        outputs = [v for v in vertices if not children.get(v, [])]
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detection - shouldn't happen for DAG
        topo_order = list(vertices)
    
    # Determine which vertices are "compute" vertices vs "input" vertices
    # Input vertices are loaded, others are computed
    # We need to track state: blue (on disk) or red (in fast memory)
    
    # Strategy: Process in topological order, keeping values in memory when possible
    # For each vertex, we need its parents' values to be available
    # If a parent is in memory, use it; otherwise, load it
    
    operations = []
    in_memory = set()  # vertices currently in fast memory (red)
    computed = set()   # vertices that have been computed and stored to disk (blue)
    loaded = set()     # vertices that have been loaded from disk
    
    def memory_full():
        return len(in_memory) >= fast_memory_budget
    
    def evict_one():
        """Evict a vertex from memory (store it to disk)."""
        if not in_memory:
            return
        # Evict a vertex that is no longer needed (all children computed)
        # or the one that will be needed least soon
        # Simple heuristic: evict the one whose children are all done
        evictable = [v for v in in_memory if all(c in computed for c in children.get(v, []))]
        if evictable:
            v = evictable[0]
            in_memory.discard(v)
            operations.append(['store', v])
        else:
            # Evict the one that appears earliest in remaining topo order
            remaining = set(topo_order) - computed - in_memory
            # Find vertex in in_memory that is not needed for any remaining vertex
            needed_by_remaining = set()
            for v in in_memory:
                for c in children.get(v, []):
                    if c in remaining or c in in_memory:
                        needed_by_remaining.add(v)
            evictable = [v for v in in_memory if v not in needed_by_remaining]
            if evictable:
                v = evictable[0]
                in_memory.discard(v)
                operations.append(['store', v])
            else:
                # Just evict any
                v = next(iter(in_memory))
                in_memory.discard(v)
                operations.append(['store', v])
    
    def ensure_available(v):
        """Ensure vertex v is in memory. Load if needed."""
        if v in in_memory:
            return
        # Need to make room
        while memory_full():
            evict_one()
        # Load v from disk
        operations.append(['load', v])
        in_memory.add(v)
    
    def compute_vertex(v):
        """Compute vertex v, assuming parents are available."""
        # Ensure all parents are in memory
        for p in parents.get(v, []):
            ensure_available(p)
        # Perform computation
        operations.append(['compute', v])
        # Result goes into memory
        if memory_full():
            evict_one()
        in_memory.add(v)
        computed.add(v)
    
    # Process vertices in topological order
    for v in topo_order:
        if v in inputs:
            # Input vertex: load it
            if v not in in_memory:
                while memory_full():
                    evict_one()
                operations.append(['load', v])
                in_memory.add(v)
            computed.add(v)
        else:
            # Compute vertex
            compute_vertex(v)
    
    # Ensure all outputs are stored (blue)
    for v in outputs:
        if v in in_memory:
            operations.append(['store', v])
            in_memory.discard(v)
    
    # Delete any remaining in-memory values (optional, but keeps state clean)
    for v in list(in_memory):
        operations.append(['delete', v])
        in_memory.discard(v)
    
    # Truncate to max 20000 operations
    return operations[:20000]
# EVOLVE-BLOCK-END
