"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure
    if isinstance(dag, dict):
        n = dag.get('n_vertices', dag.get('n', dag.get('num_vertices', None)))
        parents_list = dag.get('parents', dag.get('parent_list', None))
        inputs = dag.get('inputs', dag.get('input_vertices', []))
        outputs = dag.get('outputs', dag.get('output_vertices', []))
    else:
        n = getattr(dag, 'n_vertices', getattr(dag, 'n', getattr(dag, 'num_vertices', None)))
        parents_list = getattr(dag, 'parents', getattr(dag, 'parent_list', None))
        inputs = getattr(dag, 'inputs', getattr(dag, 'input_vertices', []))
        outputs = getattr(dag, 'outputs', getattr(dag, 'output_vertices', []))
    
    if n is None:
        if parents_list is not None:
            n = len(parents_list)
        else:
            n = 0
    
    if parents_list is None:
        parents_list = [[] for _ in range(n)]
    
    if not inputs:
        inputs = [v for v in range(n) if len(parents_list[v]) == 0]
    
    if not outputs:
        has_child = [False] * n
        for v in range(n):
            for p in parents_list[v]:
                has_child[p] = True
        outputs = [v for v in range(n) if not has_child[v]]
    
    inputs_set = set(inputs)
    outputs_set = set(outputs)
    
    operations = []
    in_fast = [False] * n
    fast_used = 0
    
    # Inputs start blue - they're already in fast memory
    for v in inputs:
        in_fast[v] = True
        fast_used += 1
    
    # Compute last use for each vertex
    last_use = [-1] * n
    for v in range(n):
        for p in parents_list[v]:
            last_use[p] = v
    for v in outputs:
        last_use[v] = n - 1
    
    # Process in topological order
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are in fast memory
        for p in parents_list[v]:
            if not in_fast[p]:
                # Need to load p
                # Check if we have space
                if fast_used + 1 > fast_memory_budget:
                    # Evict a vertex that's no longer needed
                    candidates = [u for u in range(n) if in_fast[u] and u not in outputs_set and last_use[u] < v]
                    if candidates:
                        # Evict the one with earliest last_use
                        candidates.sort(key=lambda u: last_use[u])
                        u = candidates[0]
                        in_fast[u] = False
                        fast_used -= 1
                        operations.append(('delete', u))
                operations.append(('load', p))
                in_fast[p] = True
                fast_used += 1
        
        # Compute v
        operations.append(('compute', v))
        in_fast[v] = True
        fast_used += 1
        
        # Optionally delete v if it's not an output and no longer needed
        # But only if we're over budget or it helps
        if v not in outputs_set and last_use[v] < v + 1 and fast_used > fast_memory_budget:
            # We're over budget, delete something
            # Actually, we should have checked before computing
            pass
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if not in_fast[v]:
            operations.append(('load', v))
            in_fast[v] = True
            fast_used += 1
    
    return operations

# EVOLVE-BLOCK-END
