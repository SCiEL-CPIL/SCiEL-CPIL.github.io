"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag structure (assumed):
      - dag['vertices']: list of vertex IDs
      - dag['parents']: dict vertex -> list of parent vertex IDs (topologically ordered)
      - dag['outputs']: list of output vertex IDs
      - dag['inputs']: list of input vertex IDs (start blue)
    
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    # Try to parse the dag structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    inputs = dag.get('inputs', [])
    
    if not vertices:
        return []
    
    # Build adjacency: children of each vertex
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify input vertices (no parents)
    input_set = set(inputs) if inputs else set(v for v in vertices if not parents.get(v))
    output_set = set(outputs) if outputs else set()
    
    # Layer assignment via topological levels
    level = {}
    for v in vertices:
        if not parents.get(v):
            level[v] = 0
        else:
            level[v] = 1 + max(level.get(p, 0) for p in parents[v])
    
    # Group by level
    levels = defaultdict(list)
    for v in vertices:
        levels[level[v]].append(v)
    
    max_level = max(levels.keys()) if levels else 0
    
    # Naive schedule for comparison: load all inputs, compute in topo order, store outputs
    # Our strategy: process level by level, keep values in fast memory when possible
    
    # Track state
    in_fast_memory = set()  # vertices currently in fast memory (blue)
    in_slow_memory = set()  # vertices stored in slow memory (red, need load to use)
    computed = set()  # vertices that have been computed
    
    operations = []
    
    # Initialize: inputs start blue (in fast memory)
    for v in input_set:
        in_fast_memory.add(v)
        computed.add(v)
    
    # Process level by level
    for lev in range(max_level + 1):
        if time.time() - start_time > DEADLINE:
            # Return what we have so far, padded to be valid
            break
        
        layer_vertices = levels.get(lev, [])
        
        # Sort layer vertices: prioritize those with more children (higher reuse)
        def reuse_score(v):
            # Number of children that are in future layers
            return len(children.get(v, []))
        
        layer_vertices.sort(key=reuse_score, reverse=True)
        
        for v in layer_vertices:
            if time.time() - start_time > DEADLINE:
                break
            
            if v in computed:
                continue
            
            # Check if all parents are available
            parent_list = parents.get(v, [])
            available = True
            for p in parent_list:
                if p not in in_fast_memory and p not in computed:
                    available = False
                    break
                if p not in in_fast_memory:
                    # Need to load parent from slow memory
                    operations.append(["load", p])
                    in_fast_memory.add(p)
                    in_slow_memory.discard(p)
            
            if not available:
                # Can't compute yet, skip for now (will be handled in later pass)
                continue
            
            # Compute v
            operations.append(["compute", v])
            computed.add(v)
            in_fast_memory.add(v)
            
            # Check if fast memory is over budget
            # We need to evict something if we exceed budget
            # But we can only evict non-input, non-output values that won't be needed soon
            if len(in_fast_memory) > fast_memory_budget:
                # Find a value to evict: prefer values with fewest remaining uses
                # and that are not outputs
                def eviction_priority(val):
                    if val in output_set:
                        return -1  # never evict outputs
                    if val in input_set:
                        return 0  # low priority to evict
                    # Count how many children still need this value
                    remaining_uses = sum(1 for c in children.get(val, []) if c not in computed)
                    return remaining_uses
                
                # Find candidate to evict
                candidates = [x for x in in_fast_memory if x not in output_set and x not in input_set]
                if candidates:
                    candidates.sort(key=eviction_priority)
                    evict = candidates[0]
                    operations.append(["store", evict])
                    in_fast_memory.discard(evict)
                    in_slow_memory.add(evict)
        
        # After processing a level, store values that won't be needed again
        # but keep outputs and frequently-used values
        if lev < max_level:
            for v in layer_vertices:
                if v in in_fast_memory and v not in output_set:
                    # Check if this value is needed in future layers
                    future_needed = any(c not in computed for c in children.get(v, []))
                    if not future_needed:
                        # No longer needed, can store to free space
                        operations.append(["store", v])
                        in_fast_memory.discard(v)
                        in_slow_memory.add(v)
    
    # Ensure all outputs are in fast memory (blue)
    for v in output_set:
        if v not in in_fast_memory:
            if v in in_slow_memory:
                operations.append(["load", v])
                in_fast_memory.add(v)
                in_slow_memory.discard(v)
            elif v in computed:
                # Shouldn't happen, but handle it
                operations.append(["load", v])
                in_fast_memory.add(v)
    
    # Truncate to 20000 operations if needed
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
