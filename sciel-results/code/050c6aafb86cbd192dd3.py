"""Communication-minimizing DAG pebbling with smart eviction."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Solve communication-minimizing DAG pebbling.
    
    Args:
        dag: dict with various possible formats for DAG structure
        fast_memory_budget: int, number of fast memory slots
        
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    # Parse DAG structure - handle different possible formats
    if isinstance(dag, dict):
        n = dag.get('n', 0)
        topo_order = dag.get('topo_order', list(range(n)))
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        # If parents is a list of lists, convert to dict
        if isinstance(parents, list):
            parents = {i: parents[i] for i in range(n)}
    elif isinstance(dag, (list, tuple)):
        # Maybe dag is (n, parents, inputs, outputs)
        if len(dag) >= 4:
            n, parents, inputs, outputs = dag[0], dag[1], set(dag[2]), set(dag[3])
            topo_order = list(range(n))
            if isinstance(parents, list):
                parents = {i: parents[i] for i in range(n)}
        else:
            raise ValueError("Cannot parse dag structure")
    else:
        raise ValueError("Unknown dag format")
    
    # Ensure parents has entries for all vertices
    for v in range(n):
        if v not in parents:
            parents[v] = []
    
    # Build children map
    children = {v: [] for v in range(n)}
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # State tracking
    state = {}  # 'blue', 'red', 'none'
    for v in range(n):
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'none'
    
    fast_mem = set(inputs)
    remaining_uses = {v: len(children[v]) for v in range(n)}
    
    operations = []
    
    def elapsed():
        return time.time() - start_time
    
    def is_deadline():
        return elapsed() > DEADLINE
    
    def evict_if_needed():
        """Evict vertices from fast memory if budget exceeded."""
        while len(fast_mem) > fast_memory_budget:
            # Prefer evicting non-outputs with fewest remaining uses
            candidates = [v for v in fast_mem if v not in outputs]
            if not candidates:
                candidates = list(fast_mem)
            # Evict the one with fewest remaining uses, tie-break by vertex id
            v = min(candidates, key=lambda x: (remaining_uses[x], x))
            fast_mem.discard(v)
            state[v] = 'red'
            operations.append(['store', v])
    
    def ensure_blue(v):
        """Ensure vertex v is in fast memory (blue)."""
        if state[v] == 'blue':
            return
        if state[v] == 'red':
            fast_mem.add(v)
            state[v] = 'blue'
            operations.append(['load', v])
            evict_if_needed()
        else:
            raise ValueError(f"Vertex {v} not computed, cannot ensure blue")
    
    def compute_vertex(v):
        """Compute vertex v."""
        for p in parents[v]:
            ensure_blue(p)
        
        fast_mem.add(v)
        state[v] = 'blue'
        operations.append(['compute', v])
        evict_if_needed()
        
        for p in parents[v]:
            remaining_uses[p] -= 1
    
    # Process in topological order
    for v in topo_order:
        if is_deadline():
            break
        if v in inputs:
            continue
        compute_vertex(v)
    
    # Ensure all outputs are blue at the end
    for v in outputs:
        if is_deadline():
            break
        if state[v] != 'blue':
            ensure_blue(v)
    
    # Final safety: ensure outputs are blue
    for v in outputs:
        if state[v] != 'blue':
            try:
                ensure_blue(v)
            except:
                pass
    
    return operations
# EVOLVE-BLOCK-END
