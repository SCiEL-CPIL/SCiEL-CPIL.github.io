# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    num_vertices = len(parents)
    
    children = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        for p in parents[i]:
            children[p].append(i)
    
    inputs = [i for i in range(num_vertices) if len(parents[i]) == 0]
    
    # State: which vertices are in fast memory (blue)
    in_fast = set(inputs)
    # Which vertices have been computed
    computed = set(inputs)
    # Pending children count
    pending = {v: len(children[v]) for v in range(num_vertices)}
    
    ops = []
    
    # Process in topological order
    for v in range(num_vertices):
        if time.time() > deadline:
            break
        
        # Ensure parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p
                ops.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        # v is now in fast memory (blue) after compute? Or red?
        # Let's assume compute makes it blue (in fast memory)
        in_fast.add(v)
        
        # Update pending for parents
        for p in parents[v]:
            pending[p] -= 1
            # If p has no more children and is not an output, we can delete it
            if pending[p] == 0 and p not in outputs and p not in inputs:
                if p in in_fast:
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            ops.append(['load', v])
            in_fast.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
