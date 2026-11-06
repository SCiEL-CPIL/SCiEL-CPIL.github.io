"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _parse_dag(dag):
    """Parse DAG structure into vertices, edges, outputs."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        if not vertices:
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
    elif isinstance(dag, (list, tuple)):
        vertices = []
        edges = []
        outputs = []
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    v, parents, is_out = item[0], item[1], item[2]
                    vertices.append(v)
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        return None, None, None
    
    if not vertices:
        return None, None, None
    
    return vertices, edges, outputs

def _topo_sort(all_vertices, parents_map, children_map):
    """Topological sort using Kahn's algorithm."""
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents_map.get(v, []):
            in_degree[v] += 1
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    return topo_order

def _compute_recomp_cost(topo_order, parents_map, children_map):
    """Compute recomputation cost for each vertex."""
    recomp_cost = {}
    for v in topo_order:
        if not parents_map.get(v):
            recomp_cost[v] = 0
        else:
            recomp_cost[v] = 1 + sum(recomp_cost.get(p, 0) for p in parents_map[v])
    return recomp_cost

def pebble(dag, fast_memory_budget):
    """
    Layer-based DAG pebbling with score-aware retention at layer boundaries.
    Processes vertices in topological order grouped by depth; at each depth
    boundary, retains vertices with highest future utility (children count
    weighted by recomputation cost) to minimize total transfers.
    """
    start_time = time.time()
    deadline = start_time + 55
    
    vertices, edges, outputs = _parse_dag(dag)
    if vertices is None:
        return []
    
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set(outputs)
    
    for v in vertices:
        all_vertices.add(v)
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    if not all_vertices:
        return []
    
    topo_order = _topo_sort(all_vertices, parents_map, children_map)
    if len(topo_order) != len(all_vertices):
        return []
    
    if time.time() > deadline:
        return []
    
    recomp_cost = _compute_recomp_cost(topo_order, parents_map, children_map)
    
    # Compute topological depth
    depth = {}
    for v in topo_order:
        if not parents_map.get(v):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents_map[v])
    
    # Group into layers by depth
    layers = defaultdict(list)
    for v in topo_order:
        layers[depth[v]].append(v)
    
    max_depth = max(layers.keys()) if layers else 0
    layer_list = [layers[d] for d in range(max_depth + 1)]
    
    operations = []
    fast_mem = set()
    slow_mem = set()
    computed = set()
    
    for layer_idx, layer in enumerate(layer_list):
        if time.time() > deadline:
            break
        
        # Compute all vertices in this layer
        for v in layer:
            for p in parents_map.get(v, []):
                if p not in fast_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
            
            operations.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)
        
        # At layer boundary, optimize retention
        if layer_idx < max_depth:
            # Identify vertices with future uses
            future_use = {}
            for v in fast_mem:
                if v in output_set:
                    future_use[v] = float('inf')
                    continue
                # Count children in future layers
                fut_children = sum(1 for c in children_map.get(v, []) 
                                   if depth.get(c, 0) > layer_idx)
                if fut_children > 0:
                    # Utility: number of future children * (recomp cost + 1)
                    # This captures both how many times it's needed and how expensive to recompute
                    future_use[v] = fut_children * (recomp_cost.get(v, 0) + 1)
                else:
                    future_use[v] = 0
            
            # If over budget, evict lowest-utility vertices
            if len(fast_mem) > fast_memory_budget:
                # Sort by utility (ascending) to evict lowest first
                sorted_verts = sorted(fast_mem, key=lambda v: future_use.get(v, 0))
                num_to_evict = len(fast_mem) - fast_memory_budget
                for v in sorted_verts[:num_to_evict]:
                    # Never evict outputs
                    if v in output_set:
                        continue
                    operations.append(['store', v])
                    fast_mem.discard(v)
                    slow_mem.add(v)
        
        # Immediately store vertices that are no longer needed
        for v in list(fast_mem):
            if v in output_set:
                continue
            all_children_done = all(c in computed for c in children_map.get(v, []))
            if all_children_done:
                operations.append(['store', v])
                fast_mem.discard(v)
                slow_mem.add(v)
    
    # Ensure outputs are in fast memory at the end
    for v in output_set:
        if v in all_vertices and v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Final cleanup: store non-outputs
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
