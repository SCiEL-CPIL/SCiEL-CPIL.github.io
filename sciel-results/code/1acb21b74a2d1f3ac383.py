# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs
        - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
        - 'outputs': list of output vertex IDs
        - 'inputs': list of input vertex IDs (start blue)
    
    fast_memory_budget: int, max number of red pebbles (fast memory slots)
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin within 90s limit
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    inputs = set(dag.get('inputs', []))
    
    if not vertices:
        return []
    
    # Topological order
    topo_order = list(vertices)  # assume already topologically ordered
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in topo_order:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Determine which vertices are inputs (start blue) and which need computation
    # Inputs are already blue (in fast memory or can be loaded)
    # Non-input vertices need to be computed
    
    # Strategy: Use a simple topological traversal with memory management
    # For each vertex in topo order:
    #   - If it's an input: it's already available (blue)
    #   - If not: compute it (needs parents to be blue/red)
    #   - If it's an output: store it at the end
    #   - Manage fast memory by deleting unneeded red pebbles
    
    # Count how many vertices need to be computed (non-inputs)
    to_compute = [v for v in topo_order if v not in inputs]
    
    # Simple baseline: process in topological order, keep parents in fast memory
    # This is the naive approach - let's do something slightly better
    
    # First, let's try a simple schedule:
    # 1. Load all inputs into fast memory (if budget allows) or stream them
    # 2. Compute vertices in topo order
    # 3. Store outputs
    
    operations = []
    
    # Track what's in fast memory (red pebbles)
    fast_mem = set()
    # Track what's blue (available, either in fast mem or on disk)
    blue = set(inputs)
    
    # For a simple approach: process each vertex in topo order
    # Ensure parents are in fast memory (red), compute, then decide whether to keep
    
    # Let's use a more memory-aware approach:
    # For each vertex in topo order:
    #   1. For each parent, if not in fast_mem, load it (if parent is blue)
    #   2. Compute the vertex (adds it to fast_mem)
    #   3. If fast_mem exceeds budget, delete some non-critical vertices
    
    # But we need to be careful: we can't delete a vertex if it's needed later
    # and isn't an input (can't reload)
    
    # Let's compute for each vertex how many children it has that need it
    # and use that to decide when to delete
    
    # Count future uses: for each vertex, count how many times it's needed as a parent
    # in vertices that come after it in topo order
    future_uses = {v: 0 for v in vertices}
    for v in topo_order:
        for p in parents.get(v, []):
            if p in future_uses:
                future_uses[p] += 1
    
    # Now process in topo order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if v in inputs:
            # Input vertices start blue; they may already be in fast_mem or need loading
            # For simplicity, assume inputs can be loaded on demand
            continue
        
        # Need to compute v
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                # Need to load p into fast memory
                # p should be blue (already computed or input)
                if p in blue:
                    # Load p
                    operations.append(['load', p])
                    fast_mem.add(p)
                    blue.discard(p)  # No longer just blue, now in fast mem
                else:
                    # p is not available - this shouldn't happen if we process in topo order
                    # and keep needed parents. But let's handle it.
                    pass
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        blue.discard(v)
        
        # Check if we need to free up space
        while len(fast_mem) > fast_memory_budget:
            # Find a vertex in fast_mem that is no longer needed (future_uses == 0)
            # and is not an output
            to_delete = None
            for fm_v in list(fast_mem):
                if fm_v not in outputs and future_uses.get(fm_v, 0) == 0:
                    to_delete = fm_v
                    break
            if to_delete is not None:
                operations.append(['delete', to_delete])
                fast_mem.discard(to_delete)
                blue.add(to_delete)  # It's back to blue (on disk)
            else:
                # Can't delete anything; this is a problem
                # Just break to avoid infinite loop
                break
    
    # At the end, all outputs must be blue
    # If an output is in fast_mem, we need to store it
    for out_v in outputs:
        if time.time() > deadline:
            break
        if out_v in fast_mem:
            operations.append(['store', out_v])
            fast_mem.discard(out_v)
            blue.add(out_v)
        elif out_v not in blue:
            # Output is not available - shouldn't happen
            pass
    
    return operations
# EVOLVE-BLOCK-END
