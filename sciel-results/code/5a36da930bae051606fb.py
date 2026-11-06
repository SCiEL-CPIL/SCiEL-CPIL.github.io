# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque
import math

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

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _verify_operations(operations, dag, fast_memory_budget):
    """
    Verify the operations are valid.
    Returns True if valid, False otherwise.
    """
    if not operations:
        return False
    
    vertices = set(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    
    in_fast = set()
    computed_stored = set()
    
    for op, v in operations:
        if op == 'load':
            if v not in input_set and v not in computed_stored:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            computed_stored.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        elif op == 'compute':
            for p in parents_map.get(v, []):
                if p not in in_fast:
                    return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            computed_stored.add(v)
    
    for o in outputs:
        if o not in in_fast:
            return False
    
    return True

def _dp_pebble(dag, fast_memory_budget, deadline):
    """
    Dynamic programming pebbling for small DAGs using Dijkstra with path reconstruction.
    State: (frozenset of vertices in fast memory, frozenset of computed vertices)
    We want to minimize transfers (load/store operations).
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
    output_set = frozenset(outputs)
    input_set = frozenset(v for v in vertices if not parents_map.get(v, []))
    
    # Only use for very small DAGs
    if len(vertices) > 18:
        return None
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # State: (frozenset of in_fast, frozenset of computed)
    start_state = (frozenset(), frozenset())
    
    # Dijkstra: (cost, ops, in_fast, computed)
    pq = [(0, 0, start_state[0], start_state[1])]
    best_cost = {start_state: (0, 0)}
    # Track predecessor for path reconstruction
    pred = {start_state: None}  # state -> (prev_state, op, vertex)
    
    found_goal = None
    
    while pq:
        if time.time() > deadline:
            return None
        
        cost, ops, in_fast, computed = heapq.heappop(pq)
        state = (in_fast, computed)
        
        if (cost, ops) > best_cost.get(state, (float('inf'), float('inf'))):
            continue
        
        # Check if goal reached: all outputs in fast memory
        if output_set.issubset(in_fast):
            found_goal = state
            break
        
        # Generate successor states
        
        # 1. Load an input vertex not in fast memory
        for v in input_set:
            if v not in in_fast and len(in_fast) < fast_memory_budget:
                new_in_fast = in_fast | frozenset([v])
                new_computed = computed
                new_state = (new_in_fast, new_computed)
                new_cost = cost + 1
                new_ops = ops + 1
                if new_state not in best_cost or (new_cost, new_ops) < best_cost[new_state]:
                    best_cost[new_state] = (new_cost, new_ops)
                    pred[new_state] = (state, 'load', v)
                    heapq.heappush(pq, (new_cost, new_ops, new_in_fast, new_computed))
        
        # 2. Store a computed non-output vertex from fast memory
        for v in in_fast:
            if v not in input_set and v not in output_set:
                new_in_fast = in_fast - frozenset([v])
                new_computed = computed | frozenset([v])
                new_state = (new_in_fast, new_computed)
                new_cost = cost + 1
                new_ops = ops + 1
                if new_state not in best_cost or (new_cost, new_ops) < best_cost[new_state]:
                    best_cost[new_state] = (new_cost, new_ops)
                    pred[new_state] = (state, 'store', v)
                    heapq.heappush(pq, (new_cost, new_ops, new_in_fast, new_computed))
        
        # 3. Delete (free space, no transfer cost)
        for v in in_fast:
            if v not in output_set:
                new_in_fast = in_fast - frozenset([v])
                new_computed = computed
                new_state = (new_in_fast, new_computed)
                new_cost = cost
                new_ops = ops + 1
                if new_state not in best_cost or (new_cost, new_ops) < best_cost[new_state]:
                    best_cost[new_state] = (new_cost, new_ops)
                    pred[new_state] = (state, 'delete', v)
                    heapq.heappush(pq, (new_cost, new_ops, new_in_fast, new_computed))
        
        # 4. Compute vertices whose parents are all in fast memory
        for v in vertices:
            if v in in_fast or v in computed:
                continue
            parents = parents_map.get(v, [])
            if all(p in in_fast for p in parents) and len(in_fast) < fast_memory_budget:
                new_in_fast = in_fast | frozenset([v])
                new_computed = computed | frozenset([v])
                new_state = (new_in_fast, new_computed)
                new_cost = cost  # compute doesn't cost a transfer
                new_ops = ops + 1
                if new_state not in best_cost or (new_cost, new_ops) < best_cost[new_state]:
                    best_cost[new_state] = (new_cost, new_ops)
                    pred[new_state] = (state, 'compute', v)
                    heapq.heappush(pq, (new_cost, new_ops, new_in_fast, new_computed))
    
    if found_goal is None:
        return None
    
    # Reconstruct path
    operations = []
    state = found_goal
    while pred[state] is not None:
        prev_state, op, v = pred[state]
        operations.append([op, v])
        state = prev_state
    
    operations.reverse()
    
    if len(operations) > 20000:
        return None
    
    return operations

def _greedy_pebble_v2(dag, fast_memory_budget, deadline):
    """
    Improved greedy pebbling with better eviction strategy using lookahead.
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
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Compute depth and topological order
    order = _topo_sort(vertices, parents_map)
    depth = {}
    for v in order:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents_map[v])
    
    # Usage counts: how many times each vertex is used as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    # Precompute recompute cost (number of transfers to recompute from inputs)
    recompute_cost = {}
    for v in order:
        if v in input_set:
            recompute_cost[v] = 1  # load
        else:
            cost = 0
            for p in parents_map.get(v, []):
                cost += recompute_cost.get(p, 1)
            recompute_cost[v] = cost  # compute is free, only parents matter
    
    fast_memory = set()
    stored_vertices = set()
    operations = []
    
    def get_eviction_score(v):
        """
        Score for eviction: lower means better to evict.
        Uses expected future cost if we evict v.
        """
        if v in output_set:
            return float('inf')
        
        r = max(remaining_usage.get(v, 0), 0)
        cost = recompute_cost.get(v, 1)
        
        # Expected cost if evicted: r * cost
        score = r * cost
        
        # Inputs are cheaper to reload
        if v in input_set:
            score *= 0.7
        
        return score
    
    def evict_one():
        if not fast_memory:
            return None
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
            stored_vertices.add(victim)
        fast_memory.discard(victim)
        return victim
    
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        # Evict if needed
        while len(fast_memory) >= fast_memory_budget:
            evict_one()
        
        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            # Ensure parents are in memory first
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
            stored_vertices.add(v)
    
    # Process vertices in topological order
    for v in order:
        if time.time() > deadline:
            break
        
        ensure_in_memory(v)
        
        # Update remaining usage for v's parents
        for p in parents_map.get(v, []):
            remaining_usage[p] = max(0, remaining_usage.get(p, 0) - 1)
        
        # If v is not needed in future and not an output, evict it
        if v not in output_set and remaining_usage.get(v, 0) == 0:
            if v in fast_memory:
                if v in input_set:
                    operations.append(['delete', v])
                else:
                    operations.append(['store', v])
                    stored_vertices.add(v)
                fast_memory.discard(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
    
    return operations

def _layered_pebble_v2(dag, fast_memory_budget, deadline):
    """
    Layer-based pebbling with optimized memory management.
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
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Compute depth and topological order
    order = _topo_sort(vertices, parents_map)
    depth = {}
    for v in order:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents_map[v])
    
    # Group vertices by depth (layer)
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    
    layer_indices = sorted(layers.keys())
    
    # Remaining usage count
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    # Precompute recompute cost
    recompute_cost = {}
    for v in order:
        if v in input_set:
            recompute_cost[v] = 1
        else:
            cost = 0
            for p in parents_map.get(v, []):
                cost += recompute_cost.get(p, 1)
            recompute_cost[v] = cost
    
    operations = []
    fast_memory = set()
    
    def get_keep_score(v):
        """
        Score for keeping v in fast memory: higher means better to keep.
        """
        if v in output_set:
            return float('inf')
        
        r = max(remaining_usage.get(v, 0), 0)
        cost = recompute_cost.get(v, 1)
        return r * cost
    
    def evict_one():
        """Evict the vertex with lowest keep score."""
        if not fast_memory:
            return None
        
        best_victim = None
        best_score = float('inf')
        
        for fm_v in fast_memory:
            if fm_v in output_set:
                continue
            score = get_keep_score(fm_v)
            if score < best_score:
                best_score = score
                best_victim = fm_v
        
        if best_victim is None:
            for fm_v in fast_memory:
                if fm_v not in output_set:
                    best_victim = fm_v
                    break
        
        if best_victim is None:
            return None
        
        if best_victim in input_set:
            operations.append(['delete', best_victim])
        else:
            operations.append(['store', best_victim])
        fast_memory.discard(best_victim)
        return best_victim
    
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        while len(fast_memory) >= fast_memory_budget:
            evict_one()
        
        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
    
    # Process each layer
    for layer_idx in layer_indices:
        if time.time() > deadline:
            break
        
        layer_vertices = layers[layer_idx]
        
        # Process vertices in this layer
        for v in layer_vertices:
            if time.time() > deadline:
                break
            
            ensure_in_memory(v)
            
            # Update remaining usage
            for p in parents_map.get(v, []):
                remaining_usage[p] = max(0, remaining_usage.get(p, 0) - 1)
        
        # After processing layer, evict vertices no longer needed
        # Determine which vertices are still needed for future layers
        future_needed = set()
        for future_idx in range(layer_idx + 1, max(layer_indices) + 1):
            for v in layers.get(future_idx, []):
                for p in parents_map.get(v, []):
                    future_needed.add(p)
        
        future_needed |= output_set
        
        for v in list(fast_memory):
            if v not in future_needed:
                if v in input_set:
                    operations.append(['delete', v])
                else:
                    operations.append(['store', v])
                fast_memory.discard(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
    
    return operations

def _remove_redundant_ops(operations, dag, fast_memory_budget):
    """
    Post-process to remove redundant operations.
    """
    if not operations:
        return operations
    
    parents_map = {v: dag[v] for v in dag.keys()}
    input_set = set(v for v in dag.keys() if not parents_map.get(v, []))
    
    # Pass 1: Remove store->load cycles where the stored value isn't used in between
    # This is complex, so let's do a simpler optimization:
    # Remove consecutive delete->load of same input
    optimized = []
    for i, (op, v) in enumerate(operations):
        if op == 'load' and v in input_set:
            # Check if previous op was delete of same v
            if optimized and optimized[-1] == ['delete', v]:
                # Skip both
                optimized.pop()
                continue
        
        optimized.append([op, v])
    
    # Pass 2: Remove redundant computes (compute same vertex twice without storing)
    # This requires tracking state, so skip for now
    
    return optimized

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    try:
        vertices = list(dag.keys())
        n = len(vertices)
        
        best_ops = None
        best_transfers = float('inf')
        
        # For very small DAGs, try DP (exact)
        if n <= 16:
            result = _dp_pebble(dag, fast_memory_budget, deadline)
            if result and len(result) <= 20000:
                if _verify_operations(result, dag, fast_memory_budget):
                    t = _count_transfers(result)
                    if t < best_transfers:
                        best_transfers = t
                        best_ops = result
        
        # Try layered approach
        result = _layered_pebble_v2(dag, fast_memory_budget, deadline)
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                optimized = _remove_redundant_ops(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        t = _count_transfers(optimized)
                        if t < best_transfers:
                            best_transfers = t
                            best_ops = optimized
                else:
                    t = _count_transfers(result)
                    if t < best_transfers:
                        best_transfers = t
                        best_ops = result
        
        # Try greedy approach
        result = _greedy_pebble_v2(dag, fast_memory_budget, deadline)
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                optimized = _remove_redundant_ops(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        t = _count_transfers(optimized)
                        if t < best_transfers:
                            best_transfers = t
                            best_ops = optimized
                else:
                    t = _count_transfers(result)
                    if t < best_transfers:
                        best_transfers = t
                        best_ops = result
        
        if best_ops is not None:
            return best_ops
        
        # Fallback: simple topological order
        parents_map = {v: dag[v] for v in vertices}
        outputs = dag.get('_outputs', [])
        if not outputs:
            children_map = defaultdict(list)
            for v in vertices:
                for p in parents_map.get(v, []):
                    children_map[p].append(v)
            outputs = [v for v in vertices if not children_map[v]]
        
        input_set = set(v for v in vertices if not parents_map.get(v, []))
        output_set = set(outputs)
        
        order = _topo_sort(vertices, parents_map)
        fast_memory = set()
        operations = []
        
        for v in order:
            if time.time() > deadline:
                break
            
            # Load parents
            for p in parents_map.get(v, []):
                if p not in fast_memory:
                    # Evict if needed
                    while len(fast_memory) >= fast_memory_budget and fast_memory:
                        evicted = False
                        for fm_v in list(fast_memory):
                            if fm_v not in output_set and fm_v not in input_set:
                                operations.append(['store', fm_v])
                                fast_memory.discard(fm_v)
                                evicted = True
                                break
                        if not evicted:
                            for fm_v in list(fast_memory):
                                if fm_v not in output_set:
                                    if fm_v in input_set:
                                        operations.append(['delete', fm_v])
                                    else:
                                        operations.append(['store', fm_v])
                                    fast_memory.discard(fm_v)
                                    break
                            else:
                                break
                    
                    if p in input_set:
                        operations.append(['load', p])
                    else:
                        for pp in parents_map.get(p, []):
                            if pp not in fast_memory:
                                if pp in input_set:
                                    operations.append(['load', pp])
                                else:
                                    for ppp in parents_map.get(pp, []):
                                        if ppp in input_set:
                                            operations.append(['load', ppp])
                                        else:
                                            operations.append(['load', ppp])
                                fast_memory.add(pp)
                        operations.append(['compute', p])
                    fast_memory.add(p)
            
            # Compute v
            operations.append(['compute', v])
            fast_memory.add(v)
            
            # Store v if not output and not needed later
            children = []
            for w in vertices:
                if v in parents_map.get(w, []):
                    children.append(w)
            
            if v not in output_set and not children:
                if v in input_set:
                    operations.append(['delete', v])
                else:
                    operations.append(['store', v])
                fast_memory.discard(v)
        
        # Ensure outputs
        for o in outputs:
            if o not in fast_memory:
                if o in input_set:
                    operations.append(['load', o])
                else:
                    for p in parents_map.get(o, []):
                        if p not in fast_memory:
                            if p in input_set:
                                operations.append(['load', p])
                            else:
                                operations.append(['load', p])
                            fast_memory.add(p)
                    operations.append(['compute', o])
                fast_memory.add(o)
        
        if operations and len(operations) <= 20000:
            if _verify_operations(operations, dag, fast_memory_budget):
                return operations
        
        return []
        
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
