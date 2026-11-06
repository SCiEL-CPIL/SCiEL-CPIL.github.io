"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    try:
        return _pebble_improved(dag, fast_memory_budget, deadline)
    except Exception:
        # Fallback to baseline if anything fails
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_improved(dag, fast_memory_budget, deadline):
    """
    Improved pebbling strategy:
    - Process vertices in topological order
    - Keep values in fast memory as long as possible (reference counting)
    - Only delete when memory is full and value won't be needed again
    - Ensure outputs end up blue (in fast memory)
    """
    # Extract DAG structure
    vertices = list(dag.get_vertices())
    n = len(vertices)
    if n == 0:
        return []
    
    # Get parents for each vertex (topologically ordered)
    parents = {v: dag.get_parents(v) for v in vertices}
    inputs = set(dag.get_inputs())
    outputs = set(dag.get_outputs())
    
    # Build children map for reference counting
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)
    
    # Compute reference count: how many times each value will be needed
    ref_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            ref_count[p] += 1
    
    # Initial state: inputs are blue (in fast memory)
    in_fast_memory = set(inputs)
    # Track which vertices have been computed
    computed = set(inputs)
    operations = []
    
    # Fast memory budget
    budget = fast_memory_budget
    
    # Process vertices in topological order (excluding inputs which are already computed)
    for v in vertices:
        if v in inputs:
            continue
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents[v]:
            if p not in in_fast_memory:
                # Need to load parent from slow memory
                operations.append(['load', p])
                in_fast_memory.add(p)
                # Check if we exceed budget - if so, we need to delete something
                while len(in_fast_memory) > budget:
                    # Find a value to delete: one with ref_count 0 that's not an output
                    deleted = False
                    for candidate in list(in_fast_memory):
                        if ref_count[candidate] == 0 and candidate not in outputs:
                            operations.append(['delete', candidate])
                            in_fast_memory.remove(candidate)
                            deleted = True
                            break
                    if not deleted:
                        # Delete any non-output value
                        for candidate in list(in_fast_memory):
                            if candidate not in outputs:
                                operations.append(['delete', candidate])
                                in_fast_memory.remove(candidate)
                                deleted = True
                                break
                    if not deleted:
                        # Emergency: delete output (shouldn't happen if budget >= num_outputs)
                        for candidate in list(in_fast_memory):
                            operations.append(['delete', candidate])
                            in_fast_memory.remove(candidate)
                            break
        
        # Compute the vertex
        operations.append(['compute', v])
        in_fast_memory.add(v)
        computed.add(v)
        
        # Check if we exceed budget after compute
        while len(in_fast_memory) > budget:
            # Find a value to delete
            deleted = False
            for candidate in list(in_fast_memory):
                if ref_count[candidate] == 0 and candidate not in outputs:
                    operations.append(['delete', candidate])
                    in_fast_memory.remove(candidate)
                    deleted = True
                    break
            if not deleted:
                for candidate in list(in_fast_memory):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
                        deleted = True
                        break
            if not deleted:
                for candidate in list(in_fast_memory):
                    operations.append(['delete', candidate])
                    in_fast_memory.remove(candidate)
                    break
        
        # Decrement reference count for parents
        for p in parents[v]:
            ref_count[p] -= 1
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in in_fast_memory:
            operations.append(['load', v])
            in_fast_memory.add(v)
            # Make room if needed
            while len(in_fast_memory) > budget:
                deleted = False
                for candidate in list(in_fast_memory):
                    if candidate not in outputs:
                        operations.append(['delete', candidate])
                        in_fast_memory.remove(candidate)
                        deleted = True
                        break
                if not deleted:
                    break
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate and try to ensure outputs are still blue
        operations = operations[:20000]
        # Re-ensure outputs are in fast memory
        in_fast_memory_final = set()
        # Reconstruct state from operations (simplified - just ensure outputs are loaded)
        for v in outputs:
            if v not in in_fast_memory_final:
                operations.append(['load', v])
                in_fast_memory_final.add(v)
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
