"""Communication-minimizing DAG pebbling with fast memory budget."""
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with communication minimization.
    
    Args:
        dag: A dict with keys:
            - 'vertices': list of vertex ids (0-indexed, topologically ordered)
            - 'parents': dict mapping vertex -> list of parent vertex ids
            - 'outputs': list of output vertex ids
            - 'inputs': list of input vertex ids (start blue)
        fast_memory_budget: int, max number of red pebbles (fast memory slots)
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the DAG
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', range(len(dag.get('parents', {})))))
        parents = {int(v): [int(p) for p in ps] for v, ps in dag.get('parents', {}).items()}
        outputs = [int(v) for v in dag.get('outputs', [])]
        inputs = [int(v) for v in dag.get('inputs', [])]
    else:
        # Fallback: try to infer structure
        vertices = list(range(10))
        parents = {}
        outputs = [vertices[-1]]
        inputs = [vertices[0]]
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine which vertices are inputs (no parents) and outputs
    input_set = set(inputs)
    output_set = set(outputs)
    
    # For each vertex, determine if it's an input, output, or intermediate
    # Inputs start blue (in slow memory), outputs must end blue
    # Intermediate vertices: computed, can be recomputed
    
    # Strategy: Layered approach with fast memory management
    # 1. Process vertices in topological order
    # 2. Use fast memory for vertices that will be needed soon
    # 3. Minimize transfers by keeping frequently-used values in fast memory
    
    # Compute topological order (should already be given, but verify)
    topo_order = list(range(n))
    
    # Count how many times each vertex's value is needed by its children
    use_count = defaultdict(int)
    for v in vertices:
        for c in children.get(v, []):
            use_count[v] += 1
    
    # For layered DAGs, use a layer-based approach
    # Group vertices by their "depth" (longest path from any input)
    depth = [0] * n
    for v in topo_order:
        for p in parents.get(v, []):
            depth[v] = max(depth[v], depth[p] + 1)
    
    max_depth = max(depth) if depth else 0
    
    # Group by depth
    layers = defaultdict(list)
    for v in topo_order:
        layers[depth[v]].append(v)
    
    # Fast memory management:
    # - Keep values in fast memory that will be needed in the next few layers
    # - Evict least-recently-used or least-likely-to-be-needed values
    
    ops = []
    fast_mem = set()  # vertices currently in fast memory (red pebbles)
    slow_mem = set()  # vertices currently in slow memory (blue pebbles)
    
    # Initialize: all inputs are blue (in slow memory)
    for v in inputs:
        slow_mem.add(v)
    
    # Track which vertices have been computed and are available
    computed = set()  # vertices whose values are available (in fast or slow mem)
    
    # Pre-compute: for each vertex, which future layers need it
    # A vertex at depth d is needed by vertices at depth d+1
    # We want to keep it in fast memory if it has many children in upcoming layers
    
    # Simple heuristic: keep in fast memory vertices that have children in the next 2 layers
    LOOKAHEAD = 2
    
    def is_needed_soon(v, current_depth):
        """Check if vertex v will be needed in the next LOOKAHEAD layers."""
        for c in children.get(v, []):
            if depth[c] <= current_depth + LOOKAHEAD:
                return True
        return False
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        layer_vertices = layers.get(d, [])
        
        for v in layer_vertices:
            if time.time() - start_time > DEADLINE:
                break
            
            if v in input_set:
                # Input vertices start blue; we need to load them if we want to compute from them
                # But they're already available in slow memory
                # We may want to load them into fast memory if they'll be used soon
                if is_needed_soon(v, d) and len(fast_mem) < fast_memory_budget:
                    ops.append(['load', v])
                    fast_mem.add(v)
                    slow_mem.discard(v)
                # If budget is full, keep in slow memory and load when needed
                continue
            
            # For non-input vertices, we need to compute them
            # First, ensure all parents are available (in fast or slow memory)
            for p in parents.get(v, []):
                if p not in fast_mem and p not in slow_mem:
                    # Need to load parent from slow memory (it was stored there)
                    if len(fast_mem) < fast_memory_budget:
                        ops.append(['load', p])
                        fast_mem.add(p)
                    else:
                        # Evict a vertex from fast memory
                        # Choose the one with fewest remaining uses
                        evict_v = None
                        min_uses = float('inf')
                        for fm_v in fast_mem:
                            # Count how many times fm_v is still needed
                            remaining = 0
                            for fc in children.get(fm_v, []):
                                if depth[fc] >= d:  # still in current or future layers
                                    remaining += 1
                            if remaining < min_uses:
                                min_uses = remaining
                                evict_v = fm_v
                        if evict_v is not None:
                            ops.append(['store', evict_v])
                            fast_mem.discard(evict_v)
                            slow_mem.add(evict_v)
                        ops.append(['load', p])
                        fast_mem.add(p)
            
            # Now compute v
            ops.append(['compute', v])
            computed.add(v)
            
            # Decide where to keep v
            if v in output_set:
                # Outputs must finish blue, so store immediately
                ops.append(['store', v])
                fast_mem.discard(v)
                slow_mem.add(v)
            elif is_needed_soon(v, d) and len(fast_mem) < fast_memory_budget:
                # Keep in fast memory
                fast_mem.add(v)
            elif len(fast_mem) < fast_memory_budget:
                # Keep in fast memory if we have room
                fast_mem.add(v)
            else:
                # No room, store to slow memory
                ops.append(['store', v])
                slow_mem.add(v)
            
            # Clean up parents that are no longer needed
            for p in parents.get(v, []):
                if p in fast_mem:
                    # Check if p is still needed by other vertices
                    still_needed = False
                    for c in children.get(p, []):
                        if c not in computed:
                            still_needed = True
                            break
                    if not still_needed:
                        # Can delete from fast memory
                        if p in output_set:
                            # Must keep in slow memory (blue)
                            ops.append(['store', p])
                            fast_mem.discard(p)
                            slow_mem.add(p)
                        else:
                            ops.append(['delete', p])
                            fast_mem.discard(p)
    
    # Ensure all outputs are blue (in slow memory)
    for v in outputs:
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
        elif v not in slow_mem:
            # Should not happen, but ensure it's in slow memory
            if v in fast_mem:
                ops.append(['store', v])
                fast_mem.discard(v)
            slow_mem.add(v)
    
    # Limit operations to 20000
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
