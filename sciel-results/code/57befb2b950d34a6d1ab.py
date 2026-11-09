# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure
    parents = dag.get('parents', dag.get('parent', []))
    outputs = set(dag.get('outputs', dag.get('output', [])))
    n = len(parents) if parents else dag.get('n', 0)
    
    if n == 0:
        return []
    
    # Build children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Determine inputs (no parents)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    # Compute layers (longest path from input)
    layer = [0] * n
    for v in range(n):
        if is_input[v]:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parents[v]) + 1
    
    max_layer = max(layer) if n > 0 else 0
    
    # Group by layer
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # For each vertex, find the last layer where it's needed as a parent
    last_parent_layer = [-1] * n
    for v in range(n):
        for p in parents[v]:
            last_parent_layer[p] = max(last_parent_layer[p], layer[v])
    
    operations = []
    fast_memory = set()
    
    def elapsed():
        return time.time() - start_time
    
    for L in range(max_layer + 1):
        if elapsed() > DEADLINE:
            break
        
        layer_vertices = layers[L]
        
        # Collect all parents needed for this layer
        needed = set()
        for v in layer_vertices:
            for p in parents[v]:
                needed.add(p)
        
        # Load missing parents
        for p in needed:
            if p not in fast_memory:
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute vertices in this layer
        for v in layer_vertices:
            if is_input[v]:
                if v not in fast_memory:
                    operations.append(['load', v])
                    fast_memory.add(v)
            else:
                operations.append(['compute', v])
                fast_memory.add(v)
        
        # Store outputs in this layer
        for v in layer_vertices:
            if v in outputs:
                operations.append(['store', v])
                fast_memory.discard(v)
        
        # Delete vertices no longer needed
        # A vertex can be deleted if its last use as parent is before current layer
        deletable = set()
        for v in fast_memory:
            if v in layer_vertices:
                continue
            if last_parent_layer[v] < L:
                deletable.add(v)
        
        # If over budget, delete
        if len(fast_memory) > fast_memory_budget:
            # Delete deletable vertices first
            for v in sorted(deletable, key=lambda x: last_parent_layer[x]):
                if v in fast_memory:
                    operations.append(['delete', v])
                    fast_memory.discard(v)
                if len(fast_memory) <= fast_memory_budget:
                    break
            
            # If still over, delete from current layer (non-outputs)
            if len(fast_memory) > fast_memory_budget:
                for v in reversed(layer_vertices):
                    if v in fast_memory and v not in outputs:
                        operations.append(['delete', v])
                        fast_memory.discard(v)
                    if len(fast_memory) <= fast_memory_budget:
                        break
    
    # Store any remaining outputs
    for o in outputs:
        if o in fast_memory:
            operations.append(['store', o])
            fast_memory.discard(o)
    
    # Delete remaining
    for v in list(fast_memory):
        operations.append(['delete', v])
    
    return operations

# EVOLVE-BLOCK-END
