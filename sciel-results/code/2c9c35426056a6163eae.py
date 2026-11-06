"""Communication-minimizing DAG pebbling with layered strategy."""
import time

# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (transfers).
    
    Args:
        dag: Dict with keys:
            - 'vertices': list of vertex IDs in topological order
            - 'parents': dict mapping vertex -> list of parent vertices
            - 'children': dict mapping vertex -> list of child vertices  
            - 'outputs': set or list of output vertex IDs
        fast_memory_budget: int, number of fast memory slots
    
    Returns:
        List of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    vertices = dag.get('vertices', [])
    parents_map = dag.get('parents', {})
    children_map = dag.get('children', {})
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Helper functions
    def get_parents(v):
        return parents_map.get(v, [])
    
    def get_children(v):
        return children_map.get(v, [])
    
    def is_input(v):
        return len(get_parents(v)) == 0
    
    def is_output(v):
        return v in outputs
    
    # Compute topological depth for each vertex
    depth = {}
    for v in vertices:
        ps = get_parents(v)
        if not ps:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, -1) for p in ps) + 1
    
    max_depth = max(depth.values()) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in vertices:
        layers[depth[v]].append(v)
    
    # State tracking
    fast_mem = set()  # Vertices currently in fast memory (blue)
    computed = set()  # Vertices that have been computed at least once
    ops = []
    
    def add_op(op, v):
        ops.append([op, v])
    
    def can_add_to_fast():
        return len(fast_mem) < fast_memory_budget
    
    def free_slot():
        """Free a fast memory slot by deleting a non-critical vertex."""
        # Find a vertex in fast_mem that is not a parent of any unprocessed vertex
        # and is not an output
        for v in list(fast_mem):
            if is_output(v):
                continue
            # Check if v is a parent of any vertex not yet computed
            children = get_children(v)
            needed = any(c not in computed for c in children)
            if not needed:
                add_op('delete', v)
                fast_mem.discard(v)
                return True
        # If no safe vertex to delete, delete any non-output
        for v in list(fast_mem):
            if not is_output(v):
                add_op('delete', v)
                fast_mem.discard(v)
                return True
        return False
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() > deadline or len(ops) >= 20000:
            break
        
        layer = layers[d]
        
        for v in layer:
            if time.time() > deadline or len(ops) >= 20000:
                break
            
            ps = get_parents(v)
            
            # Load parents that are computed but not in fast memory
            for p in ps:
                if p not in fast_mem and p in computed:
                    if can_add_to_fast():
                        add_op('load', p)
                        fast_mem.add(p)
                    else:
                        if free_slot():
                            add_op('load', p)
                            fast_mem.add(p)
                        else:
                            # Can't free space, skip (shouldn't happen)
                            pass
            
            # Compute v
            if can_add_to_fast():
                add_op('compute', v)
                fast_mem.add(v)
                computed.add(v)
            else:
                if free_slot():
                    add_op('compute', v)
                    fast_mem.add(v)
                    computed.add(v)
        
        # After processing a layer, optionally store some vertices to free space
        # for the next layer. We keep vertices that are parents of the next layer.
        if d < max_depth:
            next_layer = layers[d + 1]
            needed_for_next = set()
            for v in next_layer:
                for p in get_parents(v):
                    needed_for_next.add(p)
            
            # Store vertices that are no longer needed
            for v in list(fast_mem):
                if v not in needed_for_next and not is_output(v):
                    # Check if v is needed for any future layer beyond next
                    # For simplicity, only keep if needed for next layer
                    if v not in needed_for_next:
                        add_op('store', v)
                        fast_mem.discard(v)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in fast_mem and v in computed:
            if can_add_to_fast():
                add_op('load', v)
                fast_mem.add(v)
            else:
                if free_slot():
                    add_op('load', v)
                    fast_mem.add(v)
    
    return ops

# EVOLVE-BLOCK-END
