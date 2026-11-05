"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with 'vertices' (list of vertex ids in topological order),
         'parents' (dict: vertex -> list of parent vertex ids),
         'outputs' (list of output vertex ids),
         'inputs' (list of input vertex ids)
    fast_memory_budget: int, number of slots in fast memory
    
    Returns list of [operation, vertex] where operation is in {'load', 'store', 'compute', 'delete'}
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    # Parse DAG structure
    vertices = list(dag['vertices'])
    parents = dict(dag['parents'])
    outputs = list(dag['outputs'])
    inputs = list(dag.get('inputs', []))
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order is given by vertices list
    # Compute in-degree (number of uncomputed parents)
    in_degree = {}
    for v in vertices:
        in_degree[v] = len(parents.get(v, []))
    
    # For each vertex, count how many times it's needed (fan-out)
    # This helps prioritize which values to keep in fast memory
    fan_out = {}
    for v in vertices:
        fan_out[v] = len(children[v])
    
    # Fast memory state: set of vertices currently in fast memory
    fast_mem = set()
    # Slow memory: all computed values are implicitly in slow memory
    # We track what's in fast memory
    
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    # Inputs start blue (in fast memory)
    # But we need to load them if not already there
    # "Inputs start blue" means they are available in fast memory initially
    
    # Initialize: inputs are in fast memory
    for v in inputs:
        fast_mem.add(v)
    
    # Track which vertices have been computed
    computed = set()
    
    # Priority queue for vertices ready to compute
    # Priority: higher fan_out first (more valuable to compute early)
    ready = []
    for v in vertices:
        if in_degree[v] == 0:
            heapq.heappush(ready, (-fan_out[v], v))
    
    # Process vertices in topological order, but with priority
    # Actually, we need to respect topological order for computation
    # Let's process in topological order but manage fast memory smartly
    
    # Better approach: process in topological order
    # For each vertex, ensure parents are in fast memory (or load them)
    # Compute the vertex
    # Decide whether to keep it in fast memory or evict something
    
    for v in vertices:
        if time.time() > deadline:
            break
        if op_count >= MAX_OPS:
            break
        
        # Skip inputs (already in fast memory, no compute needed)
        if v in inputs:
            computed.add(v)
            continue
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                # Load parent from slow memory
                if op_count < MAX_OPS:
                    operations.append(['load', p])
                    op_count += 1
                    fast_mem.add(p)
        
        # Compute v
        if op_count < MAX_OPS:
            operations.append(['compute', v])
            op_count += 1
            computed.add(v)
        
        # Now v is computed and in fast memory (implicitly after compute)
        fast_mem.add(v)
        
        # If fast memory is over budget, evict something
        while len(fast_mem) > fast_memory_budget:
            # Choose which vertex to evict
            # Evict the one with lowest priority (lowest fan_out, not an output)
            # Prefer to evict vertices that won't be needed again
            # Compute "remaining uses" for each vertex in fast_mem
            
            # Find vertex with minimum remaining need
            best_evict = None
            best_score = float('inf')
            
            for u in fast_mem:
                if u in outputs:
                    continue  # never evict outputs
                # Count how many children of u are not yet computed
                remaining = sum(1 for c in children[u] if c not in computed)
                # Score: prefer evicting vertices with fewer remaining uses
                # Also consider if it's an input (might not be needed again)
                score = remaining
                if score < best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                # All are outputs or we can't evict
                break
            
            if op_count < MAX_OPS:
                operations.append(['store', best_evict])
                op_count += 1
                fast_mem.remove(best_evict)
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in fast_mem:
            if op_count < MAX_OPS:
                operations.append(['load', v])
                op_count += 1
                fast_mem.add(v)
    
    # Delete non-output vertices from fast memory to free space?
    # Actually, the problem says "all outputs must finish blue"
    # It doesn't say non-outputs must be deleted
    # But let's clean up to be safe
    for v in list(fast_mem):
        if v not in outputs and v not in inputs:
            if op_count < MAX_OPS:
                operations.append(['delete', v])
                op_count += 1
                fast_mem.remove(v)
    
    return operations
# EVOLVE-BLOCK-END

