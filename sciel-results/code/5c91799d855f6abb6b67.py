"""DAG pebbling to minimize communication (memory transfers)."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve DAG pebbling to minimize memory transfers.
    
    Args:
        dag: dict with 'vertices' (list), 'edges' (list of (parent, child) pairs),
             'inputs' (list of input vertex indices), 'outputs' (list of output vertex indices)
        fast_memory_budget: max number of vertices that can be in fast memory simultaneously
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Parse the DAG structure
    vertices = dag['vertices']
    n = len(vertices)
    edges = dag['edges']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build adjacency: parents[v] = list of parents of v, children[v] = list of children of v
    parents = defaultdict(list)
    children = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        parents[v].append(u)
        children[u].append(v)
        in_degree[v] += 1
    
    # Topological order
    in_deg = in_degree[:]
    topo = []
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in children[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    # Naive baseline: load all inputs, compute in topo order, store all outputs
    # This gives us a reference for scoring
    
    # Strategy: Use a greedy pebbling approach
    # State: which vertices are in fast memory (blue), which are computed but not in fast memory (red on disk)
    # We want to minimize loads + stores
    
    # Key insight: A vertex needs to be loaded if it's an input and not yet in fast memory
    # A vertex needs to be stored if it's an output and we want it to finish blue (in fast memory at end)
    # Actually, re-reading: "Inputs start blue, all outputs must finish blue"
    # So inputs start in fast memory (blue), outputs must end in fast memory (blue)
    # Red values consume fast-memory slots
    
    # Let me re-interpret: 
    # - Blue = in fast memory
    # - Red = computed but stored on slow memory (disk)
    # - Inputs start blue (in fast memory)
    # - Outputs must finish blue (in fast memory)
    # - Every red value consumes one fast-memory slot? That's confusing.
    
    # Re-reading: "every red value consumes one fast-memory slot, including newly computed results"
    # So red values DO consume fast memory slots. This means:
    # - Blue = value is in fast memory (ready to use)
    # - Red = value has been computed but is being held in fast memory (consuming a slot) but not yet "active"?
    
    # Actually, I think the model is:
    # - Fast memory holds at most `fast_memory_budget` values
    # - Values in fast memory are "blue" (available for computation)
    # - When you compute a new value, it goes into fast memory (blue) and consumes a slot
    # - You can "store" a value to slow memory, freeing its fast memory slot
    # - You can "load" a value from slow memory back to fast memory
    # - Inputs start in fast memory (blue)
    # - Outputs must end in fast memory (blue)
    # - "Red" might refer to values that have been computed but stored on slow memory?
    
    # Let me assume the standard pebbling model:
    # - Fast memory (register file) holds at most B values
    # - Slow memory (disk) holds all computed values
    # - Load: move value from slow to fast (cost 1 transfer)
    # - Store: move value from fast to slow (cost 1 transfer)
    # - Compute: if all parents are in fast memory, compute child into fast memory
    # - Delete: remove value from fast memory (no cost, but value must be on slow memory or it's lost)
    # - Inputs start in fast memory
    # - Outputs must end in fast memory
    # - A value can only be computed if all its parents are in fast memory
    # - Once computed, a value is on slow memory (can be loaded back if deleted)
    
    # Wait, the problem says "every red value consumes one fast-memory slot"
    # So maybe red = in fast memory but not currently "active" for computation?
    
    # I'll go with the standard interpretation and aim to minimize total transfers (loads + stores)
    
    # Greedy approach:
    # 1. Start with all inputs in fast memory
    # 2. Process vertices in topological order
    # 3. For each vertex, ensure all parents are in fast memory (load if needed)
    # 4. Compute the vertex (it goes into fast memory)
    # 5. If fast memory is full and we need space, store some vertex that won't be needed soon
    # 6. At the end, ensure all outputs are in fast memory (load if needed)
    
    # Track state
    in_fast = set(inputs)  # vertices currently in fast memory
    computed = set(inputs)  # vertices that have been computed (and are on slow memory)
    operations = []
    
    # Count how many times each vertex will be needed as a parent
    # This helps decide what to evict
    usage_count = defaultdict(int)
    for u, v in edges:
        usage_count[u] += 1
    
    # Process in topological order
    for v in topo:
        if time.time() > deadline:
            break
        
        if v in computed:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    # Need to evict something
                    # Evict a vertex that is not an output and has lowest future usage
                    candidates = [x for x in in_fast if x not in outputs and x != v]
                    if not candidates:
                        # All are outputs or v itself, evict any non-output
                        candidates = [x for x in in_fast if x not in outputs]
                    if not candidates:
                        # Must evict an output, pick one with lowest usage
                        candidates = list(in_fast)
                    if candidates:
                        # Pick the one with lowest remaining usage
                        min_usage = min(usage_count.get(x, 0) for x in candidates)
                        to_evict = [x for x in candidates if usage_count.get(x, 0) == min_usage][0]
                        operations.append(['store', to_evict])
                        in_fast.discard(to_evict)
                
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # If fast memory is over budget, evict
        while len(in_fast) > fast_memory_budget:
            # Evict a vertex that is not an output and not needed soon
            candidates = [x for x in in_fast if x not in outputs]
            if not candidates:
                candidates = list(in_fast)
            if candidates:
                min_usage = min(usage_count.get(x, 0) for x in candidates)
                to_evict = [x for x in candidates if usage_count.get(x, 0) == min_usage][0]
                operations.append(['store', to_evict])
                in_fast.discard(to_evict)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something
                candidates = [x for x in in_fast if x not in outputs]
                if not candidates:
                    candidates = list(in_fast)
                if candidates:
                    to_evict = candidates[0]
                    operations.append(['store', to_evict])
                    in_fast.discard(to_evict)
            operations.append(['load', v])
            in_fast.add(v)
    
    # Check operation count limit
    if len(operations) > 20000:
        # Truncate - this shouldn't happen for reasonable DAGs
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
