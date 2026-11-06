"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication for DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs (topologically ordered)
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'inputs': list of input vertex IDs (start blue)
        - 'outputs': list of output vertex IDs (must finish blue)
    
    fast_memory_budget: number of fast memory slots available
    
    Returns: list of [operation, vertex] where operation is in {load, store, compute, delete}
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    # Build children mapping
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many times each vertex's value is needed by its children
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            usage_count[p] += 1
    
    # Strategy: Process in topological order
    # Maintain a set of vertices currently in fast memory (blue)
    # Minimize loads/stores by keeping values that will be reused
    
    in_memory = set(inputs)  # Inputs start blue
    operations = []
    
    # Track which vertices have been computed
    computed = set(inputs)
    
    for v in vertices:
        if time.time() > deadline:
            break
        
        if v in inputs:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_memory:
                # Need to load p
                # Check memory space
                if len(in_memory) >= fast_memory_budget:
                    # Need to free space
                    # Find a vertex to delete that won't be needed soon
                    # Prefer deleting vertices with no future uses
                    candidates = []
                    for x in in_memory:
                        if x not in parents.get(v, []):  # Don't delete parents we need
                            # Check if x will be used in future
                            future_use = 0
                            for child in children.get(x, []):
                                if child not in computed:
                                    future_use += 1
                            candidates.append((future_use, x))
                    
                    if candidates:
                        candidates.sort()
                        to_delete = candidates[0][1]
                        operations.append(['delete', to_delete])
                        in_memory.remove(to_delete)
                
                operations.append(['load', p])
                in_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # Decide whether to keep v in memory
        # If memory is full and v won't be used again, delete it
        if len(in_memory) > fast_memory_budget:
            # Need to free space
            # Check if v will be used again
            future_use_v = 0
            for child in children.get(v, []):
                if child not in computed:
                    future_use_v += 1
            
            if future_use_v == 0:
                operations.append(['delete', v])
                in_memory.remove(v)
            else:
                # Delete something else
                candidates = []
                for x in in_memory:
                    if x != v:
                        future_use_x = 0
                        for child in children.get(x, []):
                            if child not in computed:
                                future_use_x += 1
                        candidates.append((future_use_x, x))
                
                if candidates:
                    candidates.sort()
                    to_delete = candidates[0][1]
                    operations.append(['delete', to_delete])
                    in_memory.remove(to_delete)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_memory:
            operations.append(['load', v])
            in_memory.add(v)
    
    # Truncate to max 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END

