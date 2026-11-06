"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Layered DP with greedy eviction using priority queue.
    Processes vertices in topological order, evicting based on:
    1. Next-use distance (sooner = keep)
    2. Fan-out (more children = keep)
    3. Recomputation cost (more parents = keep)
    """
    start_time = time.time()
    deadline = start_time + 55
    
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
    
    # Precompute children positions for next-use tracking
    children_positions = {}
    for v in all_vertices:
        if v in children_map:
            children_positions[v] = sorted([topo_pos[c] for c in children_map[v]])
        else:
            children_positions[v] = []
    
    fan_out = {v: len(children_map.get(v, [])) for v in all_vertices}
    recompute_cost = {v: len(parents_map.get(v, [])) for v in all_vertices}
    
    # Greedy pebbling with priority-queue eviction
    operations = []
    fast_mem = set()
    
    # Track remaining uses and next-use index for each vertex
    remaining_uses = {}
    next_use_idx = {}  # Index into children_positions list for next use
    for v in all_vertices:
        remaining_uses[v] = len(children_positions.get(v, []))
        next_use_idx[v] = 0
    
    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Decrement remaining uses for parents and advance next_use_idx
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
                # Advance next_use_idx to point to the next future use
                while next_use_idx[p] < len(children_positions.get(p, [])) and \
                      children_positions[p][next_use_idx[p]] <= idx:
                    next_use_idx[p] += 1
        
        # If over budget, evict using priority queue
        while len(fast_mem) > fast_memory_budget:
            best_to_evict = None
            best_score = float('inf')
            
            for cand in fast_mem:
                # Don't evict outputs if we can avoid it
                if cand in output_set:
                    continue
                    
                rem = remaining_uses.get(cand, 0)
                if rem <= 0:
                    # No more uses, evict this one immediately
                    score = -1000.0
                else:
                    # Find next use position efficiently using precomputed index
                    nui = next_use_idx.get(cand, 0)
                    cps = children_positions.get(cand, [])
                    if nui < len(cps):
                        next_pos = cps[nui]
                        dist = next_pos - idx
                    else:
                        # No future use despite rem > 0 (shouldn't happen), evict
                        score = -900.0
                        best_to_evict = cand
                        best_score = score
                        break
                    
                    if dist <= 0:
                        score = -950.0
                    else:
                        # Score: lower = more urgent to evict
                        # Use a smooth surrogate: keep_value = (fan_out * recompute_cost) / (1 + dist^0.7)
                        # The exponent 0.7 makes the distance penalty less harsh for moderate distances
                        reload_savings = fan_out.get(cand, 1)
                        rcost = recompute_cost.get(cand, 1)
                        # Keep value: high = keep, low = evict
                        # Smoothed distance penalty: use dist^0.7 instead of dist
                        dist_smooth = (1.0 + dist) ** 0.7
                        keep_value = (reload_savings * rcost) / dist_smooth
                        score = -keep_value
            
            if best_to_evict is None:
                # All are outputs or no candidate found, evict any non-output
                for cand in fast_mem:
                    if cand not in output_set:
                        best_to_evict = cand
                        break
                if best_to_evict is None:
                    best_to_evict = next(iter(fast_mem))
            
            operations.append(['store', best_to_evict])
            fast_mem.discard(best_to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
    
    # Only store non-output vertices that are no longer needed
    for v in list(fast_mem):
        if v not in output_set and remaining_uses.get(v, 0) <= 0:
            operations.append(['store', v])
            fast_mem.discard(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
