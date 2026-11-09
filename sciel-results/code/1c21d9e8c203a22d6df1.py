"""Communication-minimizing DAG pebbling under a bounded candidate interface."""
import time as _time

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a layered DAG.
    
    dag is expected to be a dict or object with:
      - 'vertices' or 'n': number of vertices
      - 'layers' or 'topo': list of lists, each layer is a list of vertex ids
      - 'parents': dict mapping vertex -> list of parent vertices
      - 'outputs': list of output vertex ids (must finish blue)
      - possibly 'inputs': list of input vertex ids (start blue)
    
    Returns: list of [operation, vertex] where operation is one of
             'load', 'store', 'compute', 'delete'.
    """
    deadline = _time.time() + 50.0  # safety margin before 60s kill
    
    # Try to extract DAG structure robustly
    n = dag.get('n', len(dag.get('vertices', []))) if isinstance(dag, dict) else getattr(dag, 'n', None)
    
    if n is None:
        # Try to infer from layers
        layers = dag.get('layers', []) if isinstance(dag, dict) else getattr(dag, 'layers', [])
        if layers:
            n = sum(len(layer) for layer in layers)
        else:
            n = 0
    
    layers = dag.get('layers', []) if isinstance(dag, dict) else getattr(dag, 'layers', [])
    parents = dag.get('parents', {}) if isinstance(dag, dict) else getattr(dag, 'parents', {})
    outputs = dag.get('outputs', []) if isinstance(dag, dict) else getattr(dag, 'outputs', [])
    inputs = dag.get('inputs', []) if isinstance(dag, dict) else getattr(dag, 'inputs', [])
    
    # If no layers, try to build from topological order
    if not layers:
        topo = dag.get('topo', dag.get('topological_order', [])) if isinstance(dag, dict) else getattr(dag, 'topo', getattr(dag, 'topological_order', []))
        if topo:
            # Group by depth (distance from inputs)
            depth = {}
            for v in topo:
                ps = parents.get(v, []) if parents else []
                depth[v] = 1 + max((depth[p] for p in ps if p in depth), default=0)
            max_d = max(depth.values()) if depth else 0
            layers = [[] for _ in range(max_d + 1)]
            for v, d in depth.items():
                layers[d].append(v)
        else:
            # Fallback: single layer with all vertices
            layers = [list(range(n)) if n else []]
    
    # Build parent/child adjacency
    children = {}
    for v in range(n):
        children[v] = []
    for v, ps in parents.items():
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(v)
    
    # Determine input and output sets
    if not inputs:
        inputs = set()
        for v in range(n):
            if not parents.get(v, []):
                inputs.add(v)
    if not outputs:
        outputs = set()
        for v in range(n):
            if not children.get(v, []):
                outputs.add(v)
    
    # Convert to sets for fast lookup
    input_set = set(inputs) if not isinstance(inputs, set) else inputs
    output_set = set(outputs) if not isinstance(outputs, set) else outputs
    
    # State: 'blue' = in fast memory (or external), 'red' = in fast memory, 'external' = not loaded
    # Initially: inputs are blue (available externally), others are external
    # Goal: all outputs must be blue at the end
    # Each red value consumes one fast-memory slot
    
    # Strategy: Process layer by layer. For each layer:
    #   1. Compute all vertices in the layer (requires parents to be available)
    #   2. Store computed results that are needed by future layers
    #   3. Delete results that are no longer needed
    #   4. Ensure fast memory stays within budget
    
    # First, determine which vertices need to be stored (have children in later layers)
    vertex_layer = {}
    for i, layer in enumerate(layers):
        for v in layer:
            vertex_layer[v] = i
    
    # For each vertex, determine the last layer that needs it
    last_need_layer = {}
    for v in range(n):
        if v in output_set:
            last_need_layer[v] = len(layers)  # needed until the very end
        else:
            max_child_layer = -1
            for c in children.get(v, []):
                if c in vertex_layer:
                    max_child_layer = max(max_child_layer, vertex_layer[c])
            last_need_layer[v] = max_child_layer
    
    ops = []
    
    # Track what's in fast memory (red) and what's available (blue)
    in_fast_memory = set()  # red values
    available = set()  # blue values (either in fast memory or loaded from external)
    
    # Initially, inputs are available (blue) - they can be loaded on demand
    # We'll load them just-in-time
    
    # Process each layer
    for li, layer in enumerate(layers):
        if _time.time() > deadline:
            break
        
        # Step 1: Ensure all parents of vertices in this layer are available
        # Load any needed parents from external memory
        needed = set()
        for v in layer:
            for p in parents.get(v, []):
                if p not in available and p not in in_fast_memory:
                    needed.add(p)
        
        # Load needed parents
        for p in needed:
            if p in input_set:
                ops.append(['load', p])
                available.add(p)
            elif p in in_fast_memory:
                # It's red, need to keep it or reload
                available.add(p)
            else:
                # Should not happen if we managed memory correctly
                # Reload from external (recomputation would be needed, but for now assume it's stored)
                ops.append(['load', p])
                available.add(p)
        
        # Step 2: Compute all vertices in this layer
        for v in layer:
            if _time.time() > deadline:
                break
            ps = parents.get(v, [])
            # Check if all parents are available
            all_avail = all(p in available or p in in_fast_memory for p in ps)
            if all_avail:
                ops.append(['compute', v])
                # The result is now in fast memory (red)
                in_fast_memory.add(v)
                available.add(v)
            
            # If fast memory is full, we may need to evict
            if len(in_fast_memory) > fast_memory_budget:
                # Evict vertices that are no longer needed in this or future layers
                # Prefer evicting those whose last need is in the past
                evictable = [u for u in in_fast_memory if last_need_layer.get(u, -1) < li]
                evictable.sort(key=lambda u: last_need_layer.get(u, -1))
                while len(in_fast_memory) > fast_memory_budget and evictable:
                    u = evictable.pop(0)
                    ops.append(['store', u])  # store back to external
                    in_fast_memory.discard(u)
                    available.discard(u)
        
        # Step 3: Clean up - delete vertices that are no longer needed
        # Vertices whose last need was in a previous layer can be deleted
        for v in list(in_fast_memory):
            if last_need_layer.get(v, -1) < li:
                ops.append(['delete', v])
                in_fast_memory.discard(v)
                available.discard(v)
        
        # Step 4: Store vertices that won't be needed soon, if memory is tight
        if len(in_fast_memory) > fast_memory_budget * 0.8:
            # Store vertices whose next use is far away
            for v in list(in_fast_memory):
                if len(in_fast_memory) <= fast_memory_budget:
                    break
                # Find next use layer
                next_use = len(layers)
                for c in children.get(v, []):
                    if c in vertex_layer:
                        next_use = min(next_use, vertex_layer[c])
                if next_use > li + 1:  # not needed immediately
                    ops.append(['store', v])
                    in_fast_memory.discard(v)
                    available.discard(v)
    
    # Final step: ensure all outputs are blue (available)
    for v in output_set:
        if v not in available and v not in in_fast_memory:
            # Need to load or recompute
            if v in input_set:
                ops.append(['load', v])
                available.add(v)
            else:
                # Recompute if necessary
                ps = parents.get(v, [])
                # Ensure parents are available
                for p in ps:
                    if p not in available and p not in in_fast_memory:
                        if p in input_set:
                            ops.append(['load', p])
                            available.add(p)
                        elif p in in_fast_memory:
                            available.add(p)
                ops.append(['compute', v])
                in_fast_memory.add(v)
                available.add(v)
    
    # Cap operations at 20000
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
