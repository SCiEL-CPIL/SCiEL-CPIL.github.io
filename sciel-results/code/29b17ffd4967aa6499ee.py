"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Use a greedy approach with lookahead that prioritizes:
    1. Keeping vertices with high remaining fan-out in fast memory
    2. Evicting vertices that won't be needed again or are needed far in the future
    3. Minimizing redundant loads/stores through better eviction heuristics
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Parse DAG structure
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
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if not vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency structures
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    for v in outputs:
        output_set.add(v)
        all_vertices.add(v)
    
    if not all_vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Topological sort
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
    
    if len(topo_order) != len(all_vertices):
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if time.time() > deadline:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Identify input and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    num_vertices = len(topo_order)
    
    # Precompute for each vertex:
    # 1. Total number of children (fan-out)
    # 2. For each child, its position in topo order
    # 3. The sequence of child positions (sorted)
    
    fan_out = {v: len(children_map.get(v, [])) for v in topo_order}
    
    # For each vertex, precompute the sorted list of child positions
    child_positions = {}
    for v in topo_order:
        child_positions[v] = sorted(topo_pos[c] for c in children_map.get(v, []))
    
    # Greedy pebbling with improved eviction
    operations = []
    fast_mem = set()
    slow_mem = set()
    computed = set()
    
    # Track remaining needs: for each vertex, how many children still need it
    # and what their positions are
    remaining_children = {v: list(children_map.get(v, [])) for v in topo_order}
    
    def eviction_priority(v, current_pos):
        """
        Calculate priority for eviction. Lower priority = evict first.
        We want to keep vertices that:
        1. Have many remaining children (high fan-out remaining)
        2. Have children that are soon in topological order
        3. Are outputs (must be in fast memory at end)
        """
        if v in output_set:
            return float('inf')  # Never evict outputs
        
        remaining = remaining_children.get(v, [])
        if not remaining:
            return 0  # No remaining needs, evict first
        
        # Get positions of remaining children
        remaining_positions = [topo_pos[c] for c in remaining if c in topo_pos]
        
        if not remaining_positions:
            return 0
        
        # Score: combination of count and proximity
        # More remaining children = higher priority to keep
        # Children closer in topo order = higher priority to keep
        
        count_factor = len(remaining_positions)
        min_pos = min(remaining_positions)
        distance = min_pos - current_pos if min_pos > current_pos else 0
        
        # Higher score = more important to keep
        # Use a weighted combination
        score = count_factor * 10 - distance * 0.1
        
        return score
    
    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                operations.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_mem.add(v)
        
        # Remove v from remaining_children of its parents
        for p in parents_map.get(v, []):
            if p in remaining_children and v in remaining_children[p]:
                remaining_children[p].remove(v)
        
        # If over budget, evict something
        while len(fast_mem) > fast_memory_budget:
            # Find vertex to evict: lowest priority
            candidates = list(fast_mem)
            # Sort by eviction priority (lowest first = evict first)
            candidates.sort(key=lambda x: eviction_priority(x, idx))
            if candidates:
                to_evict = candidates[0]
                operations.append(['store', to_evict])
                fast_mem.discard(to_evict)
                slow_mem.add(to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Final cleanup: store everything that's not an output
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
