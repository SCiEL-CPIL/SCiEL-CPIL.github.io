# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _greedy_pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized Greedy Pebbling with lookahead:
    - Processes vertices in topological order.
    - Uses a priority-based eviction strategy that considers:
      1. Remaining usage count (lower = evict first)
      2. Distance to next use (farther = evict first)
      3. Whether vertex is an output (protect outputs)
    - Pre-loads parents when possible to reduce stalls.
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = set(v for v in vertices if not parents_map.get(v, []))
    output_set = set(outputs)
    
    # Build children map for distance calculations
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Compute topological order
    order = _topo_sort(vertices, parents_map)
    topo_index = {v: i for i, v in enumerate(order)}
    
    # For each vertex, compute the list of children in topological order
    # and the position of each child in the topo order
    next_use_info = {}
    for v in vertices:
        # For each vertex v, track which children use it and when
        # We'll compute this dynamically during execution
        pass
    
    # Count how many times each vertex is needed as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
            
    # remaining_usage will be decremented as we process children
    remaining_usage = dict(usage_count)
    
    # For each vertex, pre-compute the list of children that use it, in topo order
    # This helps us know the "next use" distance
    children_in_topo = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_in_topo[p].append(v)
    # Sort each list by topological index
    for p in children_in_topo:
        children_in_topo[p].sort(key=lambda x: topo_index[x])
    
    fast_memory = set()
    operations = []
    
    # Track which vertices have been "processed" (computed)
    # For distance calculations, we need to know when a vertex will next be needed
    # We'll use a pointer into children_in_topo for each vertex
    
    # current_child_ptr[v] = index into children_in_topo[v] of the next child that needs v
    current_child_ptr = defaultdict(int)
    
    def get_next_use_distance(v, current_pos):
        """Get the distance (in topo steps) to the next use of vertex v."""
        ptr = current_child_ptr[v]
        while ptr < len(children_in_topo[v]):
            child = children_in_topo[v][ptr]
            if topo_index[child] > current_pos:
                return topo_index[child] - current_pos
            ptr += 1
        current_child_ptr[v] = ptr
        return float('inf')
    
    def evict_vertex(v, current_pos):
        """Evict v from fast memory. Returns the operation performed."""
        fast_memory.discard(v)
        if v in input_vertices:
            # Input is still in slow memory. Just delete from fast.
            return ['delete', v]
        else:
            # Computed vertex. Must store to slow memory.
            return ['store', v]
    
    def find_best_victim(v_to_load, current_pos, protected_set):
        """Find the best vertex to evict to make room for v_to_load."""
        best_victim = None
        best_score = None
        
        for x in fast_memory:
            if x in protected_set:
                continue
            
            # Score: lower = better to evict
            # Primary: remaining usage (lower = evict first)
            # Secondary: distance to next use (farther = evict first, i.e., higher distance = lower score)
            # Tertiary: prefer evicting non-outputs, then inputs
            
            rem_usage = remaining_usage.get(x, 0)
            dist = get_next_use_distance(x, current_pos)
            
            # If remaining usage is 0, this is the best candidate
            if rem_usage == 0:
                score = (0, 0, 0)  # Best possible
            else:
                # rem_usage > 0, so we need to consider distance
                # Higher distance means we can afford to evict
                # We want to minimize score, so we use negative distance
                is_output = 1 if x in output_set else 0
                is_input = 1 if x in input_vertices else 0
                # We want to evict: low usage, high distance, non-output, prefer input
                score = (rem_usage, -dist, is_output, -is_input)
            
            if best_score is None or score < best_score:
                best_score = score
                best_victim = x
        
        return best_victim
    
    for idx, v in enumerate(order):
        if time.time() > deadline:
            break
            
        current_pos = idx
        
        # 1. Ensure all parents are in fast memory
        parents_of_v = parents_map.get(v, [])
        
        # Protected set: parents of v that we need to keep
        protected = set(parents_of_v)
        
        for p in parents_of_v:
            if p not in fast_memory:
                # Need to load p. Make room if necessary.
                while len(fast_memory) >= fast_memory_budget:
                    # Find best victim to evict
                    best_victim = find_best_victim(p, current_pos, protected)
                    
                    if best_victim is None:
                        # All vertices in fast_memory are protected
                        # This shouldn't happen if budget >= max in-degree
                        break
                        
                    # Evict best_victim
                    op = evict_vertex(best_victim, current_pos)
                    operations.append(op)
                
                if len(fast_memory) < fast_memory_budget:
                    operations.append(['load', p])
                    fast_memory.add(p)
        
        # 2. Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # 3. Update usage counts and child pointers for parents
        for p in parents_of_v:
            remaining_usage[p] = remaining_usage.get(p, 0) - 1
            # Advance the child pointer for p
            while current_child_ptr[p] < len(children_in_topo[p]):
                if children_in_topo[p][current_child_ptr[p]] == v:
                    current_child_ptr[p] += 1
                    break
                current_child_ptr[p] += 1
    
    # 4. Finalize: Ensure all outputs are Blue (in slow memory)
    for o in outputs:
        if o in fast_memory:
            operations.append(['store', o])
            fast_memory.discard(o)
    
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    try:
        result = _greedy_pebble_optimized(dag, fast_memory_budget, deadline)
        
        # Basic validation: check length and structure
        if result and len(result) <= 20000:
            return result
        else:
            # Fallback: try a simpler approach
            vertices = list(dag.keys())
            parents_map = {v: dag[v] for v in vertices}
            outputs = dag.get('_outputs', [])
            
            if not outputs:
                children_map = defaultdict(list)
                for v in vertices:
                    for p in parents_map.get(v, []):
                        children_map[p].append(v)
                outputs = [v for v in vertices if not children_map[v]]
            
            order = _topo_sort(vertices, parents_map)
            ops = []
            fast_mem = set()
            
            for v in order:
                for p in parents_map.get(v, []):
                    if p not in fast_mem:
                        while len(fast_mem) >= fast_memory_budget:
                            # Evict first non-parent
                            evicted = False
                            for x in list(fast_mem):
                                if x not in parents_map.get(v, []):
                                    if x in set(v2 for v2 in vertices if not parents_map.get(v2, [])):
                                        ops.append(['delete', x])
                                    else:
                                        ops.append(['store', x])
                                    fast_mem.discard(x)
                                    evicted = True
                                    break
                            if not evicted:
                                break
                        if len(fast_mem) < fast_memory_budget:
                            ops.append(['load', p])
                            fast_mem.add(p)
                ops.append(['compute', v])
                fast_mem.add(v)
            
            for o in outputs:
                if o in fast_mem:
                    ops.append(['store', o])
                    fast_mem.discard(o)
            
            if len(ops) <= 20000:
                return ops
            else:
                return []
    except Exception as e:
        # Fall back to baseline on any error
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
