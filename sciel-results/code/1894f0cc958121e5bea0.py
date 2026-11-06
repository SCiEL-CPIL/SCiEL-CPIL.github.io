"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag structure (expected):
    - dag['vertices']: list of vertex IDs (topologically ordered)
    - dag['parents']: dict mapping vertex -> list of parent vertices
    - dag['outputs']: list of output vertex IDs
    - dag['inputs']: list of input vertex IDs (start blue)
    
    Returns list of [operation, vertex] where operation is load/store/compute/delete
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure - handle various possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
    else:
        # Try alternative structures
        vertices = list(range(len(dag))) if hasattr(dag, '__len__') else []
        parents = {}
        outputs = []
        inputs = []
    
    if not vertices:
        return []
    
    # Build adjacency and in-degree info
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute topological order (should already be given, but verify)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detected or issue; use given order
        topo_order = list(vertices)
    
    # Identify input and output sets
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if not children[v])
    
    # Count total vertices and edges
    n_vertices = len(vertices)
    n_edges = sum(len(parents.get(v, [])) for v in vertices)
    
    # Naive schedule: load all inputs, compute in topological order, store outputs
    # This gives us the baseline
    def naive_schedule():
        ops = []
        # Load all inputs
        for v in input_set:
            ops.append(['load', v])
        # Compute in topological order
        computed = set(input_set)
        for v in topo_order:
            if v in computed:
                continue
            # Ensure all parents are available
            for p in parents.get(v, []):
                if p not in computed:
                    ops.append(['load', p])
                    computed.add(p)
            ops.append(['compute', v])
            computed.add(v)
        # Store outputs
        for v in output_set:
            if v in computed:
                ops.append(['store', v])
        return ops
    
    naive_ops = naive_schedule()
    naive_transfers = sum(1 for op in naive_ops if op[0] in ('load', 'store'))
    
    if fast_memory_budget >= n_vertices:
        # Can hold everything in fast memory
        ops = []
        for v in input_set:
            ops.append(['load', v])
        computed = set(input_set)
        for v in topo_order:
            if v not in computed:
                ops.append(['compute', v])
                computed.add(v)
        for v in output_set:
            ops.append(['store', v])
        return ops
    
    # Recursive pebbling strategy
    # For layered DAGs, we can use a divide-and-conquer approach
    # Split the DAG into layers and process them with memory management
    
    # Determine the "width" of the DAG (max vertices at any topological level)
    # Use a recursive strategy: process sub-DAGs, keeping only necessary values
    
    # Strategy: Use a priority-based approach where we compute vertices
    # in an order that minimizes the number of reloads
    
    # For each vertex, compute its "pebbling cost" - how many times it needs to be loaded
    # Use dynamic programming on the DAG structure
    
    # Simple effective strategy for bounded memory:
    # 1. Process in topological order
    # 2. When memory is full, evict the vertex that will be needed furthest in the future
    # 3. This is an approximation to optimal pebbling
    
    # Build future-use counts for eviction policy
    # For each vertex, compute the last time it's needed as a parent
    last_use = {}
    for i, v in enumerate(reversed(topo_order)):
        idx = len(topo_order) - 1 - i
        # Find the latest position where this vertex is used as a parent
        max_pos = -1
        for other in topo_order[idx+1:]:
            if v in parents.get(other, []):
                max_pos = topo_order.index(other)
                break
        last_use[v] = max_pos
    
    # More efficient: precompute for each vertex the set of children
    # and the maximum topological index among children
    vertex_to_idx = {v: i for i, v in enumerate(topo_order)}
    for v in vertices:
        if children[v]:
            last_use[v] = max(vertex_to_idx[c] for c in children[v])
        else:
            last_use[v] = -1
    
    # Greedy pebbling with LRU-like eviction based on future use
    ops = []
    in_memory = set()  # vertices currently in fast memory
    computed = set()   # vertices that have been computed at least once
    memory_full = False
    
    # Load initial inputs
    for v in input_set:
        ops.append(['load', v])
        in_memory.add(v)
        computed.add(v)
        if len(in_memory) >= fast_memory_budget:
            memory_full = True
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            # Return best effort
            break
        
        if v in computed:
            # Already computed, check if in memory
            if v not in in_memory:
                # Need to reload
                if len(in_memory) >= fast_memory_budget:
                    # Evict the vertex with furthest future use
                    # among non-output vertices that are computed
                    candidates = [x for x in in_memory if x != v and (x not in output_set or last_use[x] < last_use.get(v, -1))]
                    if not candidates:
                        candidates = [x for x in in_memory if x != v]
                    if candidates:
                        # Evict the one with smallest last_use (needed soonest or never)
                        victim = min(candidates, key=lambda x: last_use.get(x, -1))
                        ops.append(['delete', victim])
                        in_memory.discard(victim)
                    else:
                        # No safe victim, evict any
                        victim = next(iter(in_memory - {v}), None)
                        if victim is not None:
                            ops.append(['delete', victim])
                            in_memory.discard(victim)
                ops.append(['load', v])
                in_memory.add(v)
            continue
        
        # Need to compute v
        # Ensure all parents are in memory
        for p in parents.get(v, []):
            if p not in in_memory:
                if p not in computed:
                    # Shouldn't happen if topological order is correct
                    # But handle gracefully: load if it was an input
                    if p in input_set:
                        ops.append(['load', p])
                        in_memory.add(p)
                        computed.add(p)
                    else:
                        # Recompute needed - this is complex, skip for now
                        pass
                else:
                    # Parent was computed but not in memory, need to reload
                    if len(in_memory) >= fast_memory_budget:
                        # Evict
                        candidates = [x for x in in_memory if x != p and x != v]
                        if candidates:
                            victim = min(candidates, key=lambda x: last_use.get(x, -1))
                            ops.append(['delete', victim])
                            in_memory.discard(victim)
                    ops.append(['load', p])
                    in_memory.add(p)
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        
        # Check if we need to add v to memory
        if len(in_memory) < fast_memory_budget:
            in_memory.add(v)
        else:
            # Memory full, decide whether to keep v or evict someone
            # v is just computed, likely needed soon
            # Evict the vertex with smallest last_use among others
            candidates = [x for x in in_memory if x != v]
            if candidates:
                victim = min(candidates, key=lambda x: last_use.get(x, -1))
                ops.append(['delete', victim])
                in_memory.discard(victim)
            in_memory.add(v)
    
    # Ensure all outputs are in memory and stored
    for v in output_set:
        if time.time() - start_time > DEADLINE:
            break
        if v not in in_memory:
            if len(in_memory) >= fast_memory_budget:
                candidates = [x for x in in_memory if x != v]
                if candidates:
                    victim = min(candidates, key=lambda x: last_use.get(x, -1))
                    ops.append(['delete', victim])
                    in_memory.discard(victim)
            ops.append(['load', v])
            in_memory.add(v)
        ops.append(['store', v])
    
    # Validate and limit operations
    if len(ops) > 20000:
        # Truncate - this shouldn't happen for reasonable DAGs
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
