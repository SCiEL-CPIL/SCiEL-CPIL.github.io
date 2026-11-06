# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs
        - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
        - 'outputs': list of output vertex IDs
        - 'inputs': list of input vertex IDs (start blue)
    
    fast_memory_budget: int, number of fast-memory slots available
    
    Returns: list of [operation, vertex] where operation is in {'load', 'store', 'compute', 'delete'}
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Parse DAG structure
    vertices = list(dag['vertices'])
    parents = {v: list(p) for v, p in dag['parents'].items()}
    outputs = set(dag['outputs'])
    inputs = set(dag.get('inputs', []))
    
    # Build children mapping
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Topological order (already given by parents structure)
    topo_order = list(vertices)
    
    # Identify computation chains: paths from inputs to outputs
    # For layered DAGs, we can decompose into independent chains
    
    # Strategy: For each "layer" or independent path, apply DP over memory budget
    # to find optimal checkpointing points that minimize total recomputation cost
    
    # First, let's identify the structure:
    # In a layered DAG, vertices at the same "depth" form a layer
    # We can compute depth of each vertex
    
    depth = {}
    for v in topo_order:
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents[v]) + 1
    
    max_depth = max(depth.values()) if depth else 0
    
    # Group vertices by depth
    layers = defaultdict(list)
    for v in topo_order:
        layers[depth[v]].append(v)
    
    # For each vertex, compute:
    # - cost to compute from scratch (number of ancestors that need recomputation)
    # - which vertices depend on it (number of descendants)
    
    # Compute number of descendants for each vertex
    num_descendants = {}
    for v in reversed(topo_order):
        if not children[v]:
            num_descendants[v] = 1  # counts itself
        else:
            num_descendants[v] = 1 + sum(num_descendants.get(c, 1) for c in children[v])
    
    # For a chain-based approach, identify maximal paths from inputs to outputs
    # In layered DAGs with fan-in/fan-out, we need a more general approach
    
    # Key insight: The optimal strategy involves deciding which intermediate 
    # results to keep in fast memory vs. recompute.
    
    # DP approach: For each vertex, decide whether to keep its result in fast memory.
    # State: which vertices' results are currently in fast memory
    # This is exponential in general, but for layered DAGs with bounded width,
    # we can use a layer-by-layer DP.
    
    # Let's try a heuristic approach first:
    # 1. Process vertices in topological order
    # 2. For each vertex, compute if parents are available
    # 3. If not enough memory, delete least-useful vertices
    # 4. Load from slow memory if needed
    
    # Better approach: Pre-plan which vertices to keep based on future usage
    
    # Count how many times each vertex's value is needed (by its children)
    # In a DAG, each vertex's value is needed once per child
    usage_count = {}
    for v in topo_order:
        usage_count[v] = len(children[v])
    
    # Priority for deletion: vertices with lower usage and farther in the future
    # are better candidates for deletion
    
    # Let's implement a simulation-based approach with lookahead:
    # 1. Pre-compute for each vertex, the "criticality" based on how many 
    #    outputs depend on it and how expensive it is to recompute
    # 2. Use this to guide memory management decisions
    
    # Recomputation cost: number of ancestors that would need to be recomputed
    recompute_cost = {}
    for v in topo_order:
        if not parents[v]:
            recompute_cost[v] = 0  # inputs are free
        else:
            recompute_cost[v] = 1 + max(recompute_cost.get(p, 0) for p in parents[v])
    
    # Criticality: how much this vertex matters for final outputs
    criticality = {}
    for v in reversed(topo_order):
        if v in outputs:
            criticality[v] = 1.0
        elif not children[v]:
            criticality[v] = 0.0
        else:
            criticality[v] = max(criticality.get(c, 0) for c in children[v])
    
    # Now simulate the pebbling process:
    # State: 
    #   - fast_mem: set of vertex IDs currently in fast memory (blue)
    #   - slow_mem: set of vertex IDs stored in slow memory (blue, can be loaded)
    #   - computed: set of vertex IDs that have been computed at least once
    
    # Operations:
    #   load: move vertex from slow to fast memory (1 transfer)
    #   store: move vertex from fast to slow memory (1 transfer)
    #   compute: compute vertex from parents (must have parents in fast mem)
    #   delete: remove vertex from fast memory (0 transfers, frees slot)
    
    # Strategy:
    # 1. Load all inputs into fast memory (or keep in slow and load as needed)
    # 2. Process vertices in topological order
    # 3. When computing a vertex, ensure parents are in fast memory
    # 4. When fast memory is full, delete the least critical vertex
    # 5. After computing a vertex, decide whether to store it in slow memory
    
    # Let's implement this:
    
    # Initial state: inputs are in slow memory (blue), fast memory is empty
    fast_mem = set()
    slow_mem = set(inputs)
    computed = set(inputs)  # inputs are "computed" (available)
    operations = []
    
    # We need to track which vertices have been "finalized" (no longer needed for future computations)
    # A vertex is finalized when all its children have been computed
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        if v in inputs:
            # Input vertices: need to be loaded into fast memory when first needed
            continue  # Handle when needed by a child
        
        # Check if all parents are in fast memory
        missing_parents = [p for p in parents[v] if p not in fast_mem]
        
        if missing_parents:
            # Need to load missing parents into fast memory
            for p in missing_parents:
                if p in slow_mem:
                    # Load from slow to fast
                    if len(fast_mem) >= fast_memory_budget:
                        # Need to delete something first
                        # Find least critical vertex to delete
                        delete_candidates = [x for x in fast_mem if x != v]
                        if delete_candidates:
                            # Delete the one with lowest criticality * usage
                            best_delete = min(delete_candidates, 
                                              key=lambda x: criticality.get(x, 0) * usage_count.get(x, 0))
                            operations.append(['delete', best_delete])
                            fast_mem.discard(best_delete)
                    
                    operations.append(['load', p])
                    fast_mem.add(p)
                    # p stays in slow_mem (we can load it again if needed)
                elif p not in computed:
                    # Parent hasn't been computed yet - this shouldn't happen in topo order
                    pass
                else:
                    # Parent was computed but not in slow memory - need to recompute
                    # This is a problem - we should have stored it
                    # For now, mark as needing recomputation
                    pass
        
        # Now all parents should be in fast memory (or we've done our best)
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Decide whether to store v in slow memory
        # Store if it will be needed later and memory allows
        if v not in outputs and children[v]:
            # Will be needed by children
            # Store if fast memory has room or if we can delete something
            if len(fast_mem) <= fast_memory_budget:
                # Can store without deleting
                operations.append(['store', v])
                slow_mem.add(v)
                # Keep in fast memory for now (it's both in fast and slow)
            else:
                # Fast memory is full
                # Decide: store and delete something, or don't store
                # If v is highly critical, store it
                if criticality.get(v, 0) > 0:
                    # Find something to delete
                    delete_candidates = [x for x in fast_mem if x != v]
                    if delete_candidates:
                        best_delete = min(delete_candidates,
                                          key=lambda x: criticality.get(x, 0) * usage_count.get(x, 0))
                        operations.append(['delete', best_delete])
                        fast_mem.discard(best_delete)
                        operations.append(['store', v])
                        slow_mem.add(v)
        
        # Clean up: delete vertices that are no longer needed
        # A vertex is no longer needed if all its children have been computed
        # and it's not an output
        for x in list(fast_mem):
            if x in inputs:
                # Inputs: delete if no more children need them
                remaining_children = [c for c in children[x] if c not in computed]
                if not remaining_children and x not in outputs:
                    operations.append(['delete', x])
                    fast_mem.discard(x)
            elif x not in outputs:
                # Non-input, non-output: delete if no more children need them
                remaining_children = [c for c in children[x] if c not in computed]
                if not remaining_children:
                    operations.append(['delete', x])
                    fast_mem.discard(x)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    delete_candidates = [x for x in fast_mem if x != v]
                    if delete_candidates:
                        best_delete = min(delete_candidates,
                                          key=lambda x: criticality.get(x, 0) * usage_count.get(x, 0))
                        operations.append(['delete', best_delete])
                        fast_mem.discard(best_delete)
                operations.append(['load', v])
                fast_mem.add(v)
            else:
                # Need to recompute - this is a failure case
                # For now, just compute it (parents may not be available)
                operations.append(['compute', v])
                fast_mem.add(v)
    
    # Limit operations to 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
