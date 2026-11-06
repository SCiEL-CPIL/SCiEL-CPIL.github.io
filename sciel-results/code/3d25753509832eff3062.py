"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
        'vertices': list of vertex IDs (topologically ordered)
        'parents': dict mapping vertex -> list of parent vertices
        'outputs': list of output vertex IDs
        'inputs': list of input vertex IDs (start blue)
    fast_memory_budget: int, max red values in fast memory
    
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    if not vertices:
        return []
    
    n = len(vertices)
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute future use count: number of children that haven't been processed yet
    # We'll track this dynamically
    
    # State
    fast_mem = set()
    on_disk = set()
    computed = set()
    
    ops = []
    
    # remaining_children: how many children of v still need to be computed
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Initial: inputs are in fast memory
    for inp in inputs:
        if inp in vertices:
            fast_mem.add(inp)
            computed.add(inp)
    
    # Process in topological order
    for v in vertices:
        if time.time() > DEADLINE:
            break
        
        if v in inputs:
            # Already in fast memory
            continue
        
        # Ensure parents are available
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in on_disk:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    on_disk.discard(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Update remaining children for parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
        
        # Evict vertices that are no longer needed
        # A vertex can be evicted if:
        # 1. It's not an output
        # 2. All its children have been computed (remaining_children == 0)
        # 3. It's in fast memory
        # Prioritize eviction to stay within budget
        
        if len(fast_mem) > fast_memory_budget:
            # Need to evict
            # Find candidates: in fast_mem, not output, remaining_children == 0
            candidates = [
                x for x in fast_mem 
                if x not in outputs and remaining_children.get(x, 0) == 0
            ]
            
            # Evict until within budget
            while len(fast_mem) > fast_memory_budget and candidates:
                # Evict one (arbitrary for now, could be smarter)
                x = candidates.pop(0)
                ops.append(['store', x])
                fast_mem.discard(x)
                on_disk.add(x)
            
            # If still over budget and no candidates, we have a problem
            # Force evict something non-output
            if len(fast_mem) > fast_memory_budget:
                non_outputs = [x for x in fast_mem if x not in outputs]
                while len(fast_mem) > fast_memory_budget and non_outputs:
                    x = non_outputs.pop(0)
                    ops.append(['store', x])
                    fast_mem.discard(x)
                    on_disk.add(x)
        
        # Proactively evict vertices that are no longer needed to free space
        # This helps avoid future loads
        if len(fast_mem) < fast_memory_budget:
            # We have room, but let's still evict done vertices to keep memory clean
            # Actually, no point - they're not causing issues
            pass
    
    # Ensure outputs are in fast memory at the end
    for out in outputs:
        if out not in fast_mem:
            if out in on_disk:
                ops.append(['load', out])
                fast_mem.add(out)
                on_disk.discard(out)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops

# EVOLVE-BLOCK-END
