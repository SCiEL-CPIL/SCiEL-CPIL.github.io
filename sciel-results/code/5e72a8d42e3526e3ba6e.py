"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
        - 'vertices': list of vertex ids (topologically ordered)
        - 'parents': dict mapping vertex -> list of parent vertex ids
        - 'inputs': list of input vertex ids
        - 'outputs': list of output vertex ids
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Extract DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count how many children each vertex has (for knowing when to delete)
    child_count = {v: len(children[v]) for v in vertices}
    
    # Track remaining children (how many children still need this vertex)
    remaining_children = dict(child_count)
    
    # Fast memory: set of vertices currently in fast memory
    fast_mem = set()
    
    # Operations list
    ops = []
    
    # Process vertices in topological order
    for v in vertices:
        # Check deadline
        if time.time() > DEADLINE:
            break
        
        # If v is an input, it starts blue (in memory) - no load needed
        # If v is not an input, we need to compute it from parents
        
        if v in inputs:
            # Input is already blue, nothing to do
            continue
        
        # Need to compute v: ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                # Need to load p into fast memory
                # Check if we have budget
                if len(fast_mem) >= fast_memory_budget:
                    # Need to evict something - find a vertex in fast_mem 
                    # that won't be needed soon
                    # Simple heuristic: evict the vertex with the fewest remaining children
                    evict_candidates = [x for x in fast_mem if remaining_children.get(x, 0) == 0]
                    if evict_candidates:
                        # Evict one that's no longer needed
                        evict_v = min(evict_candidates, key=lambda x: remaining_children.get(x, 0))
                        ops.append(['delete', evict_v])
                        fast_mem.discard(evict_v)
                    else:
                        # Must evict something - pick the one with least remaining use
                        evict_v = min(fast_mem, key=lambda x: remaining_children.get(x, 0))
                        ops.append(['delete', evict_v])
                        fast_mem.discard(evict_v)
                
                ops.append(['load', p])
                fast_mem.add(p)
        
        # Now compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Update remaining children count for parents
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
        
        # Check deadline
        if time.time() > DEADLINE:
            break
    
    # Ensure all outputs are in fast memory (blue)
    for out in outputs:
        if out not in fast_mem:
            # Need to load the output
            if len(fast_mem) >= fast_memory_budget:
                # Evict something
                evict_v = min(fast_mem, key=lambda x: remaining_children.get(x, 0))
                ops.append(['delete', evict_v])
                fast_mem.discard(evict_v)
            ops.append(['load', out])
            fast_mem.add(out)
    
    # Limit to 20000 operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
