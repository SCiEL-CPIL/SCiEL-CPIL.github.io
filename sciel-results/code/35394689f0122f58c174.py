"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with communication minimization.
    
    dag: dictionary with:
      - 'vertices': list of vertex IDs in topological order
      - 'parents': dict mapping vertex -> list of parent vertices
      - 'outputs': list of output vertex IDs
      - 'inputs': list of input vertex IDs (or inferable)
    
    fast_memory_budget: number of fast memory slots
    
    Returns: list of [operation, vertex] where operation in {'load', 'store', 'compute', 'delete'}
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Parse DAG structure
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Determine inputs: vertices with no parents
    inputs = set()
    for v in vertices:
        if not parents.get(v, []):
            inputs.add(v)
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine which vertices are needed (ancestors of outputs)
    needed = set()
    stack = list(outputs)
    while stack:
        v = stack.pop()
        if v not in needed:
            needed.add(v)
            for p in parents.get(v, []):
                stack.append(p)
    
    # Only consider needed vertices
    needed_vertices = [v for v in vertices if v in needed]
    
    # Topological order of needed vertices
    in_degree = {v: 0 for v in needed_vertices}
    for v in needed_vertices:
        for p in parents.get(v, []):
            if p in needed:
                in_degree[v] += 1
    
    topo_order = []
    queue = deque([v for v in needed_vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            if c in needed:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    # Compute out-degree (number of children that need this vertex)
    out_degree = {v: 0 for v in needed_vertices}
    for v in needed_vertices:
        for c in children.get(v, []):
            if c in needed:
                out_degree[v] += 1
    
    # Greedy pebbling strategy:
    # Process in topological order
    # For each vertex, decide whether to keep in fast memory or recompute later
    # Heuristic: keep in fast memory if out_degree is high (reused often)
    # and fast memory has space
    
    operations = []
    fast_memory = set()  # vertices currently in fast memory
    computed = set()  # vertices that have been computed at least once
    
    # Track when each vertex's value is last needed
    last_use = {}
    for i, v in enumerate(topo_order):
        last_use[v] = i  # default: last use is its own position
        for c in children.get(v, []):
            if c in needed:
                idx = topo_order.index(c) if c in topo_order else -1
                if idx > last_use[v]:
                    last_use[v] = idx
    
    # More accurate last_use: find the last child that needs this vertex
    for v in needed_vertices:
        max_pos = -1
        for c in children.get(v, []):
            if c in needed and c in last_use:
                # Find position of c in topo_order
                try:
                    pos = topo_order.index(c)
                    if pos > max_pos:
                        max_pos = pos
                except ValueError:
                    pass
        if max_pos >= 0:
            last_use[v] = max_pos
        else:
            last_use[v] = topo_order.index(v) if v in topo_order else 0
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # Check if all parents are in fast memory
        parent_ready = all(p in fast_memory for p in parents.get(v, []) if p in needed)
        
        if not parent_ready:
            # Need to load missing parents
            for p in parents.get(v, []):
                if p in needed and p not in fast_memory:
                    # Load parent from slow memory
                    operations.append(['load', p])
                    fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_memory.add(v)
        
        # Check fast memory capacity
        while len(fast_memory) > fast_memory_budget:
            # Need to evict something
            # Evict the vertex with the earliest last_use that isn't an output
            candidates = [x for x in fast_memory if x not in outputs]
            if not candidates:
                # All are outputs, can't evict
                break
            
            # Evict the one with earliest last_use
            to_evict = min(candidates, key=lambda x: last_use.get(x, 0))
            operations.append(['delete', to_evict])
            fast_memory.discard(to_evict)
    
    # Store all outputs
    for v in outputs:
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
        else:
            # Output not in fast memory, need to recompute
            # For now, just load it (simplified)
            operations.append(['load', v])
            operations.append(['store', v])
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate and try to make it valid
        # This is a fallback
        pass
    
    return operations[:20000]
# EVOLVE-BLOCK-END

