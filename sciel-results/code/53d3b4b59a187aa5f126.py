"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Parse the DAG structure
    if not isinstance(dag, dict):
        # Fallback: try to handle other structures
        return []
    
    vertices = dag.get('vertices', [])
    edges = dag.get('edges', [])
    inputs = dag.get('inputs', [])
    outputs = dag.get('outputs', [])
    
    # Build parent/child maps from edges
    parent_map = defaultdict(list)  # child -> list of parents
    child_map = defaultdict(list)   # parent -> list of children
    
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            p, c = edge
            parent_map[c].append(p)
            child_map[p].append(c)
        elif isinstance(edge, dict):
            p = edge.get('from', edge.get('parent', edge.get('src')))
            c = edge.get('to', edge.get('child', edge.get('dst')))
            if p is not None and c is not None:
                parent_map[c].append(p)
                child_map[p].append(c)
    
    # Get all vertices
    all_vertices = set()
    for v in vertices:
        all_vertices.add(v)
    for v in parent_map:
        all_vertices.add(v)
        for p in parent_map[v]:
            all_vertices.add(p)
    for v in child_map:
        all_vertices.add(v)
        for c in child_map[v]:
            all_vertices.add(c)
    for v in inputs:
        all_vertices.add(v)
    for v in outputs:
        all_vertices.add(v)
    
    if not all_vertices:
        return []
    
    vertex_list = sorted(all_vertices)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Topological sort
    in_degree = {v: len(parent_map[v]) for v in vertex_list}
    queue = deque([v for v in vertex_list if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in child_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) < len(vertex_list):
        remaining = set(vertex_list) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute depth
    depth = {}
    for v in topo_order:
        if not parent_map[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parent_map[v]) + 1
    
    max_depth = max(depth.values()) if depth else 0
    
    # Group by layer
    layers = defaultdict(list)
    for v in topo_order:
        layers[depth[v]].append(v)
    
    layer_list = [layers[i] for i in range(max_depth + 1)]
    
    # Strategy: Process layer by layer
    # - Inputs start in fast memory
    # - For each layer, compute vertices
    # - Decide which to keep in fast memory based on future usage
    # - Delete values that won't be needed
    
    operations = []
    fast_memory = set(input_set)  # Inputs start blue
    
    # Count future usage for each vertex
    future_usage = {v: 0 for v in vertex_list}
    for v in vertex_list:
        # Count how many children have this vertex as parent
        future_usage[v] = len(child_map[v])
        # Also add if it's an output
        if v in output_set:
            future_usage[v] += 1
    
    # Process each layer
    for layer_idx in range(max_depth + 1):
        if time.time() > deadline:
            break
        
        if layer_idx >= len(layer_list):
            break
        
        layer = layer_list[layer_idx]
        
        # Compute vertices in this layer
        for v in layer:
            if time.time() > deadline:
                break
            
            # Ensure parents are in fast memory
            for p in parent_map[v]:
                if p not in fast_memory:
                    # Need to load or recompute p
                    if p in input_set:
                        operations.append(['load', p])
                        fast_memory.add(p)
                    else:
                        # Recompute p (simplified - just mark as computed)
                        # In a full solution, we'd recursively compute p
                        operations.append(['compute', p])
                        fast_memory.add(p)
            
            # Compute v
            operations.append(['compute', v])
            
            # Decide whether to keep v in fast memory
            # Keep if: budget allows AND (high future usage OR is output)
            should_keep = (v in output_set) or (future_usage[v] > 1)
            
            if should_keep:
                if len(fast_memory) < fast_memory_budget:
                    fast_memory.add(v)
                else:
                    # Need to evict something
                    # Find vertex with lowest future usage that's not an input
                    evict_candidate = None
                    min_usage = float('inf')
                    for fm_v in fast_memory:
                        if fm_v not in input_set and fm_v not in output_set:
                            if future_usage[fm_v] < min_usage:
                                min_usage = future_usage[fm_v]
                                evict_candidate = fm_v
                    
                    if evict_candidate is not None:
                        operations.append(['delete', evict_candidate])
                        fast_memory.remove(evict_candidate)
                        fast_memory.add(v)
                    # If no good candidate, don't keep v
            else:
                # Don't keep v, will recompute if needed later
                pass
        
        # After layer, delete values not needed in future
        for v in layer:
            if v not in fast_memory or v in input_set:
                continue
            
            # Check if v is needed in any future layer
            needed_future = False
            for future_idx in range(layer_idx + 1, len(layer_list)):
                for fv in layer_list[future_idx]:
                    if v in parent_map[fv]:
                        needed_future = True
                        break
                if needed_future:
                    break
            
            if v in output_set:
                needed_future = True
            
            if not needed_future:
                operations.append(['delete', v])
                fast_memory.remove(v)
    
    # Ensure all outputs are in fast memory
    for v in sorted(output_set):
        if time.time() > deadline:
            break
        if v not in fast_memory:
            if v in input_set:
                operations.append(['load', v])
                fast_memory.add(v)
            else:
                # Recompute
                for p in parent_map[v]:
                    if p not in fast_memory:
                        if p in input_set:
                            operations.append(['load', p])
                            fast_memory.add(p)
                        else:
                            operations.append(['compute', p])
                            fast_memory.add(p)
                operations.append(['compute', v])
                fast_memory.add(v)
    
    # Truncate if needed
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END

