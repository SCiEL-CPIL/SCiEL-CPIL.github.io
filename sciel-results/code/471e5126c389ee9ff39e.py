"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Improved DAG pebbling using Layered/Topological processing with 
    Farthest-Future-Use (FFU) eviction policy.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s
    
    try:
        return _pebble_ffu(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            # If baseline import fails, return empty or simple schedule
            return []

def _pebble_ffu(dag, fast_memory_budget, deadline):
    # Extract DAG structure
    # Handle different dag object types
    if hasattr(dag, 'get_vertices'):
        vertices = list(dag.get_vertices())
        get_parents = dag.get_parents
        inputs = set(dag.get_inputs())
        outputs = set(dag.get_outputs())
    elif isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('V', [])))
        parents_dict = dag.get('parents', {})
        inputs = set(dag.get('inputs', set()))
        outputs = set(dag.get('outputs', set()))
        def get_parents(v):
            return parents_dict.get(v, [])
    else:
        # Fallback to simple extraction
        vertices = list(getattr(dag, 'vertices', []))
        def get_parents(v):
            return getattr(dag, 'parents', {}).get(v, [])
        inputs = set(getattr(dag, 'inputs', set()))
        outputs = set(getattr(dag, 'outputs', set()))

    n = len(vertices)
    if n == 0:
        return []
    
    # Build parents and children maps
    parents = {v: get_parents(v) for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
            else:
                # If parent not in vertices list? Should not happen in valid DAG
                # But handle gracefully
                pass

    # Identify inputs and outputs if not provided
    if not inputs:
        inputs = set(v for v in vertices if not parents[v])
    if not outputs:
        # Outputs are vertices with no children
        outputs = set(v for v in vertices if not children[v])

    # Topological Sort
    # Using Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for child in children[v]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
        
    # If topo_order length != n, there's a cycle. Fallback to simple order
    if len(topo_order) != n:
        topo_order = vertices

    # Map vertex to its index in topo order
    vertex_index = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute children indices in topo order for each vertex
    children_indices = {v: [] for v in vertices}
    for v in vertices:
        for c in children[v]:
            if c in vertex_index:
                children_indices[v].append(vertex_index[c])
        # Sort to allow binary search or simple iteration
        children_indices[v].sort()

    # Precompute last_use for each vertex
    # last_use[v] is the max index of any child, or infinity if output
    last_use = {}
    for v in vertices:
        if v in outputs:
            last_use[v] = float('inf')
        elif children_indices[v]:
            last_use[v] = max(children_indices[v])
        else:
            # No children, not an output? 
            # If it's an input with no children, it's just loaded and never used?
            # Or it's an isolated node.
            last_use[v] = vertex_index[v] # Use itself as proxy, effectively "now"
            
    # State
    in_memory = set()
    operations = []
    budget = fast_memory_budget
    
    # Helper to get next use index for a vertex given current index
    # Since children_indices[v] is sorted, we can find the first index > current_idx
    def get_next_use(v, current_idx):
        if v in outputs:
            return float('inf')
        indices = children_indices[v]
        if not indices:
            return float('inf') # No future uses
        # Find first index > current_idx
        # Simple linear scan is O(deg), which is fine for small deg
        # For optimization, we could use bisect
        next_idx = float('inf')
        for idx in indices:
            if idx > current_idx:
                next_idx = idx
                break
        return next_idx

    def evict_ffu(current_idx, protected_vertices):
        """
        Evict the vertex in in_memory (not in protected) with the farthest next use.
        Returns the evicted vertex or None.
        """
        best_candidate = None
        best_next_use = -1
        
        for v in in_memory:
            if v in protected_vertices:
                continue
            nu = get_next_use(v, current_idx)
            if nu > best_next_use:
                best_next_use = nu
                best_candidate = v
        
        if best_candidate is not None:
            operations.append(['delete', best_candidate])
            in_memory.discard(best_candidate)
            return best_candidate
        return None

    # Process vertices in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # If v is an input, it needs to be loaded
        # If v is not an input, its parents need to be loaded
        
        if v in inputs:
            # Input starts blue? "Inputs start blue" means they are in fast memory?
            # "Inputs start blue, all outputs must finish blue"
            # Usually, "blue" means in fast memory.
            # If inputs start blue, they are already in memory.
            # But the budget is bounded. If #inputs > budget, we can't keep them all.
            # The prompt says "Inputs start blue". This implies initial state.
            # If #inputs > budget, the problem is ill-posed or we must evict immediately.
            # Let's assume #inputs <= budget or we handle eviction.
            
            if v not in in_memory:
                if len(in_memory) >= budget:
                    # Evict someone
                    evict_ffu(idx, protected_vertices={v})
                operations.append(['load', v])
                in_memory.add(v)
        else:
            # Non-input vertex
            # Ensure parents are in memory
            protected = set(parents[v])
            for p in parents[v]:
                if p not in in_memory:
                    if len(in_memory) >= budget:
                        # Evict someone who is not a parent of v
                        evict_ffu(idx, protected)
                    operations.append(['load', p])
                    in_memory.add(p)
            
            # Compute v
            operations.append(['compute', v])
            in_memory.add(v)
            
            # If v's last use is now, delete it (unless it's an output)
            if last_use[v] == idx and v not in outputs:
                operations.append(['delete', v])
                in_memory.discard(v)
                
            # If memory is still full (e.g. if we added v and didn't delete), evict
            # This handles the case where v is not deleted but memory is tight
            if len(in_memory) > budget:
                # Evict farthest future use, protecting nothing specific now
                # But we just computed v, so v is in memory.
                # We should probably not evict v immediately if it has future uses.
                evict_ffu(idx, protected_vertices=set())

    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_memory:
            if len(in_memory) >= budget:
                # Evict someone who is not an output
                protected = outputs
                evict_ffu(n, protected) # current_idx doesn't matter much for eviction choice if we use last_use logic
                # Actually, get_next_use uses current_idx. At the end, current_idx is n.
                # All next_uses will be inf.
                # So it will pick an arbitrary one.
                # Better to just delete any non-output.
                # Let's refine evict_ffu for the end game
                # For simplicity, if full, delete first non-output found
                to_delete = None
                for u in in_memory:
                    if u not in outputs:
                        to_delete = u
                        break
                if to_delete:
                    operations.append(['delete', to_delete])
                    in_memory.discard(to_delete)
            operations.append(['load', v])
            in_memory.add(v)

    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
# EVOLVE-BLOCK-END
