# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling to minimize transfers.
    
    dag structure:
    - dag['vertices']: list of vertex names
    - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
    - dag['inputs']: list of input vertices (start blue)
    - dag['outputs']: list of output vertices (must finish blue)
    
    fast_memory_budget: number of red values that can be stored simultaneously
    
    Returns list of [operation, vertex] where operation is load/store/compute/delete
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Parse DAG structure robustly
    vertices = []
    parents = {}
    inputs = []
    outputs = []
    
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        
        # Parse parents
        parents_raw = dag.get('parents', {})
        if isinstance(parents_raw, dict):
            # Case 1: dict {child: [parents]}
            parents = {k: list(v) for k, v in parents_raw.items()}
        elif isinstance(parents_raw, list):
            # Case 2: list of pairs [(child, [parents]), ...] or [(parent, child), ...]
            # We need to distinguish. 
            # If elements are lists of length 2, and the second element is a list, it's (child, parents).
            # If the second element is a scalar, it might be (parent, child).
            # Given "topologically ordered parents", (child, [parents]) is more likely for "parents" key.
            # But "edges" usually implies (u, v).
            # Let's check content.
            temp_parents = {}
            is_edge_list = False
            if parents_raw:
                first = parents_raw[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    # Check if second item is a list (parents) or scalar (edge)
                    if isinstance(first[1], (list, tuple)):
                        # (child, [parents])
                        for item in parents_raw:
                            c, ps = item
                            if c not in temp_parents:
                                temp_parents[c] = []
                            temp_parents[c].extend(ps)
                    else:
                        # (parent, child) or (child, parent)
                        # Assume (parent, child) as standard edge direction, but key is 'parents'?
                        # If key is 'parents', it's ambiguous. 
                        # Let's assume if key is 'parents', it's (child, parent_list) or dict.
                        # If key is 'edges', it's (parent, child).
                        # Here key is 'parents'. So likely (child, parent_list) where parent_list was flattened?
                        # Or maybe it's a list of lists [[child, p1], [child, p2]]?
                        # Let's try to detect if first[0] is in vertices (if known) or just assume (child, parent) pairs if single values.
                        # Actually, if it's a list of pairs of scalars, it's likely (child, parent) pairs.
                        for item in parents_raw:
                            c, p = item
                            if c not in temp_parents:
                                temp_parents[c] = []
                            temp_parents[c].append(p)
            
            parents = temp_parents
        else:
            parents = {}
            
        inputs = list(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
        
        # If vertices not provided, infer from parents/inputs/outputs
        if not vertices:
            all_v = set(inputs) | set(outputs)
            for c, ps in parents.items():
                all_v.add(c)
                for p in ps:
                    all_v.add(p)
            vertices = list(all_v)
            
    elif isinstance(dag, list):
        # Case 3: dag is a list of edges [(parent, child), ...]
        edges = dag
        all_v = set()
        temp_parents = defaultdict(list)
        for u, v in edges:
            all_v.add(u)
            all_v.add(v)
            temp_parents[v].append(u)
        vertices = list(all_v)
        parents = dict(temp_parents)
        # Inputs/Outputs not specified in list format? 
        # Maybe first/last? Or we have to guess. 
        # The task says "dag has ... inputs ... outputs". 
        # If dag is just a list, we don't know inputs/outputs. 
        # This case is likely invalid or requires external info. 
        # But to avoid crash, we'll assume no inputs (all are computed?) or all leaves are inputs?
        # Let's assume this format is not the primary one, but handle it gracefully.
        # If no inputs/outputs specified, we can't solve properly. 
        # Return empty or best effort.
        # For now, assume all vertices with no parents are inputs, and all with no children are outputs.
        children_map = defaultdict(list)
        for c, ps in parents.items():
            for p in ps:
                children_map[p].append(c)
        inputs = [v for v in vertices if len(parents.get(v, [])) == 0]
        outputs = [v for v in vertices if len(children_map.get(v, [])) == 0]

    else:
        return []
        
    if not vertices:
        return []
    
    # Build adjacency: children of each vertex
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycles or issues, just use given order
        topo_order = vertices
    
    # Identify layers: group vertices by their longest path from inputs
    layer_of = {}
    for v in topo_order:
        if v in inputs:
            layer_of[v] = 0
        else:
            max_parent_layer = -1
            for p in parents.get(v, []):
                if p in layer_of:
                    max_parent_layer = max(max_parent_layer, layer_of[p])
            layer_of[v] = max_parent_layer + 1 if max_parent_layer >= 0 else 0
    
    # Group by layer
    layers = defaultdict(list)
    for v in topo_order:
        layers[layer_of[v]].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # Count how many times each vertex's value is needed by children
    use_count = {}
    for v in vertices:
        use_count[v] = len(children.get(v, []))
    
    operations = []
    fast_memory = set()  # vertices currently in fast memory (red)
    blue_set = set(inputs)  # vertices currently blue (available without load)
    
    def can_load(v):
        """Check if we can load v into fast memory"""
        if v in fast_memory:
            return True
        if len(fast_memory) < fast_memory_budget:
            return True
        return False
    
    def load_into_fast(v):
        """Load vertex v into fast memory"""
        if v not in fast_memory:
            if len(fast_memory) >= fast_memory_budget:
                # Need to evict something
                evict_candidates = [x for x in fast_memory if x != v]
                if evict_candidates:
                    min_uses = float('inf')
                    evict_v = None
                    for x in evict_candidates:
                        remaining = 0
                        for c in children.get(x, []):
                            if c not in blue_set and c not in fast_memory:
                                remaining += 1
                        if remaining < min_uses:
                            min_uses = remaining
                            evict_v = x
                    if evict_v:
                        operations.append(['delete', evict_v])
                        fast_memory.discard(evict_v)
            operations.append(['load', v])
            fast_memory.add(v)
    
    def store_from_fast(v):
        """Store vertex v from fast memory to slow (make it blue)"""
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
            blue_set.add(v)
    
    # Process layer by layer
    for layer in range(max_layer + 1):
        if layer not in layers:
            continue
        
        layer_vertices = layers[layer]
        
        # Determine which vertices from previous layers we should keep in fast memory
        needed_from_prev = set()
        for v in layer_vertices:
            for p in parents.get(v, []):
                if layer_of.get(p, -1) == layer - 1:
                    needed_from_prev.add(p)
        
        # Decide what to keep in fast memory
        keep_in_fast = set()
        if fast_memory_budget > 0:
            for p in needed_from_prev:
                num_uses_in_layer = sum(1 for v in layer_vertices if p in parents.get(v, []))
                if num_uses_in_layer > 1:
                    keep_in_fast.add(p)
        
        # Evict vertices not in keep_in_fast and not in current layer
        to_evict = fast_memory - keep_in_fast
        to_evict = to_evict - set(inputs)
        
        for v in to_evict:
            if v in fast_memory:
                operations.append(['delete', v])
                fast_memory.discard(v)
        
        # Process vertices in this layer
        for v in layer_vertices:
            if time.time() > DEADLINE:
                break
            
            # Check if all parents are available
            parent_available = all(
                (p in blue_set) or (p in fast_memory) or (p in inputs)
                for p in parents.get(v, [])
            )
            
            if not parent_available:
                for p in parents.get(v, []):
                    if p not in blue_set and p not in fast_memory and p not in inputs:
                        if p in keep_in_fast or use_count.get(p, 0) > 1:
                            load_into_fast(p)
                    elif p not in fast_memory and p not in blue_set and p not in inputs:
                        load_into_fast(p)
            
            # Now compute v
            for p in parents.get(v, []):
                if p not in fast_memory and p not in blue_set and p not in inputs:
                    load_into_fast(p)
            
            # Compute v
            operations.append(['compute', v])
            
            # Decide whether to put v in fast memory
            if use_count.get(v, 0) > 1 and fast_memory_budget > 0:
                if v not in fast_memory:
                    load_into_fast(v)
            else:
                if v in fast_memory:
                    store_from_fast(v)
                else:
                    blue_set.add(v)
        
        # After processing layer, clean up fast memory
        for v in list(fast_memory):
            still_needed = False
            for c in children.get(v, []):
                if layer_of.get(c, -1) > layer:
                    still_needed = True
                    break
            if not still_needed:
                operations.append(['delete', v])
                fast_memory.discard(v)
                if v not in blue_set and v not in inputs:
                    blue_set.add(v)
    
    # Ensure all outputs are blue
    for v in outputs:
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
            blue_set.add(v)
        elif v not in blue_set and v not in inputs:
            if v not in fast_memory:
                load_into_fast(v)
            store_from_fast(v)
    
    # Truncate if too many operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
