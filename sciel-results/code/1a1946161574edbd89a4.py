"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Greedy topological-order pebbling with:
    1. Proper state tracking (fast/slow/uncomputed)
    2. O(1) next-use lookup via precomputed arrays
    3. Eviction priority based on reuse frequency and distance
    4. Recomputation-aware eviction (prefer evicting vertices whose parents are also evicted)
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
    
    n = len(topo_order)
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute for each vertex:
    # - list of child positions (sorted)
    # - pointer index into that list for efficient "next use" lookup
    children_pos = {}
    child_ptr = {}
    for v in all_vertices:
        if v in children_map:
            cps = sorted([topo_pos[c] for c in children_map[v]])
            children_pos[v] = cps
            child_ptr[v] = 0
        else:
            children_pos[v] = []
            child_ptr[v] = 0
    
    # Count total uses (fan-out) for each vertex
    fan_out = {v: len(children_map.get(v, [])) for v in all_vertices}
    
    # Track state: 'fast', 'slow', or 'uncomputed'
    state = {v: 'uncomputed' for v in all_vertices}
    fast_mem = set()
    
    # Track remaining uses for each vertex
    remaining_uses = {v: len(children_pos.get(v, [])) for v in all_vertices}
    
    operations = []
    
    def get_next_use(v, current_idx):
        """Get the next position where v is used as a parent, after current_idx."""
        cps = children_pos.get(v, [])
        ptr = child_ptr.get(v, 0)
        # Advance pointer to first position > current_idx
        while ptr < len(cps) and cps[ptr] <= current_idx:
            ptr += 1
        child_ptr[v] = ptr
        if ptr < len(cps):
            return cps[ptr]
        return None
    
    def evict_best(fast_set, current_idx, is_final=False):
        """Choose the best vertex to evict from fast memory."""
        if not fast_set:
            return None
        
        best_to_evict = None
        best_score = float('inf')
        
        for cand in fast_set:
            rem = remaining_uses.get(cand, 0)
            
            if rem <= 0:
                # No more uses - good candidate for eviction
                score = -100.0
            else:
                next_pos = get_next_use(cand, current_idx)
                if next_pos is not None:
                    dist = next_pos - current_idx
                    # Score: higher = more important to keep
                    # Factor in fan-out (more uses = more expensive to reload)
                    # and distance (closer use = more important)
                    fo = fan_out.get(cand, 1)
                    score = (fo + 1.0) / (1.0 + dist)
                    # Bonus if it's an output (we need it at the end)
                    if cand in output_set and not is_final:
                        score *= 2.0
                else:
                    # Shouldn't happen if rem > 0, but handle gracefully
                    score = -50.0
            
            # Penalize evicting outputs (we'll need to reload them)
            if cand in output_set and not is_final:
                score -= 5.0
            
            if score < best_score:
                best_score = score
                best_to_evict = cand
        
        return best_to_evict
    
    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if state[p] != 'fast':
                operations.append(['load', p])
                state[p] = 'fast'
                fast_mem.add(p)
        
        # Compute v (parents are now in fast memory)
        operations.append(['compute', v])
        state[v] = 'fast'
        fast_mem.add(v)
        
        # Decrement remaining uses for parents (they were just used)
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # If over budget, evict until within budget
        while len(fast_mem) > fast_memory_budget:
            to_evict = evict_best(fast_mem, idx)
            if to_evict is None:
                break
            operations.append(['store', to_evict])
            state[to_evict] = 'slow'
            fast_mem.discard(to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if state[v] != 'fast':
            operations.append(['load', v])
            state[v] = 'fast'
            fast_mem.add(v)
    
    # Final cleanup: store non-output vertices
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            state[v] = 'slow'
            fast_mem.discard(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
