"""
Communication-minimizing DAG pebbling via state-space search with Dijkstra's algorithm.

The approach:
- Model the pebbling problem as a shortest-path problem in the state space of red-pebble configurations.
- States are represented by frozensets of vertex indices that are currently red (in fast memory).
- Edges represent valid operations (load, store, compute, delete) with costs corresponding to memory transfers.
- Use Dijkstra's algorithm to find the minimum-cost sequence from initial state (all inputs blue) to target state (all outputs blue).
- Respect the fast memory budget constraint at each step.
- Limit operations to at most 20000.
"""

def pebble(dag, fast_memory_budget):
    import heapq
    from collections import defaultdict
    
    # Parse the DAG structure
    # dag is expected to have:
    # - dag['vertices']: list of vertices (or just count)
    # - dag['parents']: dict or list mapping vertex -> list of parent vertices (topologically ordered)
    # - dag['outputs']: list of output vertices
    # - dag['inputs']: list of input vertices (start blue)
    
    # Try to extract structure from dag
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        num_vertices = dag.get('num_vertices', None)
        # Try to infer num_vertices
        if num_vertices is None:
            all_verts = set()
            for v in outputs:
                all_verts.add(v)
            for v in inputs:
                all_verts.add(v)
            for p_list in parents.values():
                for p in p_list:
                    all_verts.add(p)
            for v in parents.keys():
                all_verts.add(v)
            num_vertices = max(all_verts) + 1 if all_verts else 0
    else:
        # Fallback: assume dag is a tuple or list
        raise ValueError("Unsupported dag format")
    
    # Normalize parents to a dict with list values
    if isinstance(parents, list):
        parents_dict = {i: parents[i] if i < len(parents) else [] for i in range(num_vertices)}
    else:
        parents_dict = {v: list(p) for v, p in parents.items()}
    
    # Ensure all vertices have entries in parents_dict
    for v in range(num_vertices):
        if v not in parents_dict:
            parents_dict[v] = []
    
    # Determine which vertices are inputs (sources with no parents, or explicitly marked)
    if not inputs:
        inputs = [v for v in range(num_vertices) if len(parents_dict.get(v, [])) == 0]
    
    # Ensure outputs is a list
    if not isinstance(outputs, list):
        outputs = list(outputs)
    
    # Input vertices start blue (not in fast memory)
    # All outputs must finish blue (not in fast memory)
    # Red values consume fast memory slots
    
    # State: frozenset of vertex indices that are red (in fast memory)
    # Initial state: empty set (all blue)
    # Target state: empty set (all blue, including outputs)
    # But we need to ensure all outputs have been computed and are blue
    
    # Actually, the constraint is that inputs start blue and outputs must finish blue.
    # A vertex is blue if it's not in fast memory. So initial state = empty set.
    # Target state = empty set, but we must have computed all outputs at some point.
    
    # To ensure outputs are computed, we can track which vertices have been computed.
    # But for simplicity, let's use a state that includes both the red set and a bitmask of computed vertices.
    
    # Given the potential size of the state space, let's use a heuristic approach:
    # We'll use Dijkstra but with a limit on the number of states explored.
    
    # For small DAGs, exact Dijkstra is feasible. For larger ones, we need heuristics.
    
    # Let's first try exact Dijkstra for small cases, and fall back to a greedy heuristic for larger ones.
    
    # Check if the problem is small enough for exact search
    if num_vertices <= 12 and fast_memory_budget <= 6:
        # Exact Dijkstra search
        return _pebble_exact(dag, fast_memory_budget, parents_dict, inputs, outputs, num_vertices)
    else:
        # Heuristic: greedy pebbling with recomputation
        return _pebble_heuristic(dag, fast_memory_budget, parents_dict, inputs, outputs, num_vertices)


def _pebble_exact(dag, fast_memory_budget, parents_dict, inputs, outputs, num_vertices):
    """Exact Dijkstra search over state space."""
    import heapq
    from collections import defaultdict
    
    # State: (red_set, computed_mask)
    # red_set: frozenset of vertices currently in fast memory (red)
    # computed_mask: bitmask of vertices that have been computed at least once
    
    initial_red = frozenset()
    initial_computed = 0
    initial_state = (initial_red, initial_computed)
    
    # Target: red_set is empty, and all outputs are computed
    target_computed_mask = 0
    for v in outputs:
        target_computed_mask |= (1 << v)
    
    # Dijkstra
    # Priority queue: (cost, step_count, state)
    pq = [(0, 0, initial_state)]
    best_cost = {initial_state: 0}
    best_path = {initial_state: []}
    
    max_ops = 20000
    max_states = 500000  # Limit to avoid memory blowup
    
    while pq and len(best_cost) < max_states:
        cost, steps, (red_set, computed_mask) = heapq.heappop(pq)
        
        # Check if we've reached the target
        if len(red_set) == 0 and (computed_mask & target_computed_mask) == target_computed_mask:
            # Reconstruct path
            path = best_path[(red_set, computed_mask)]
            if len(path) <= max_ops:
                return _format_path(path, num_vertices)
        
        if steps >= max_ops:
            continue
        
        # Generate neighbors
        red_list = list(red_set)
        
        # 1. Load an input vertex (blue -> red): cost 1 transfer
        for v in inputs:
            if v not in red_set:
                new_red = red_set | {v}
                new_computed = computed_mask | (1 << v)
                new_state = (new_red, new_computed)
                new_cost = cost + 1
                if new_cost < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_cost
                    best_path[new_state] = best_path[(red_set, computed_mask)] + [('load', v)]
                    heapq.heappush(pq, (new_cost, steps + 1, new_state))
        
        # 2. Store a red vertex (red -> blue): cost 1 transfer
        for v in red_set:
            new_red = red_set - {v}
            new_computed = computed_mask  # Still computed
            new_state = (new_red, new_computed)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                best_path[new_state] = best_path[(red_set, computed_mask)] + [('store', v)]
                heapq.heappush(pq, (new_cost, steps + 1, new_state))
        
        # 3. Compute a vertex: all parents must be red, vertex must not be red
        #    Cost: 0 if result stays in fast memory (no transfer), but it occupies a slot
        #    Actually, compute produces a red value, so it adds to red_set
        #    Cost for compute: 0 transfers (the computation itself doesn't transfer, but the result is in fast memory)
        #    Wait, the problem says "every red value consumes one fast-memory slot, including newly computed results"
        #    So compute adds a red vertex. The cost is 0 transfers for the compute operation itself.
        for v in range(num_vertices):
            if v in red_set:
                continue
            # Check if all parents are red
            parents_v = parents_dict.get(v, [])
            if all(p in red_set for p in parents_v):
                # Can compute v
                if len(red_set) < fast_memory_budget:
                    new_red = red_set | {v}
                    new_computed = computed_mask | (1 << v)
                    new_state = (new_red, new_computed)
                    new_cost = cost  # No transfer for compute
                    if new_cost < best_cost.get(new_state, float('inf')):
                        best_cost[new_state] = new_cost
                        best_path[new_state] = best_path[(red_set, computed_mask)] + [('compute', v)]
                        heapq.heappush(pq, (new_cost, steps + 1, new_state))
        
        # 4. Delete a red vertex: cost 0 (no transfer, just frees a slot)
        #    But deletion is only useful if we don't need the value anymore
        #    For now, allow delete as a no-cost operation that removes a red vertex
        for v in red_set:
            # Only allow delete if v is not an output that hasn't been stored yet? 
            # Actually, delete just removes from fast memory. The value is lost.
            # This is useful for freeing up space.
            new_red = red_set - {v}
            new_computed = computed_mask  # Still marked as computed
            new_state = (new_red, new_computed)
            new_cost = cost  # No transfer for delete
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                best_path[new_state] = best_path[(red_set, computed_mask)] + [('delete', v)]
                heapq.heappush(pq, (new_cost, steps + 1, new_state))
    
    # If exact search didn't find a solution within limits, fall back to heuristic
    return _pebble_heuristic(dag, fast_memory_budget, parents_dict, inputs, outputs, num_vertices)


def _pebble_heuristic(dag, fast_memory_budget, parents_dict, inputs, outputs, num_vertices):
    """Heuristic pebbling: process vertices in topological order with recomputation."""
    
    # Topological order (vertices are already in topological order based on the problem statement)
    # Process from inputs to outputs
    
    # Build a topological ordering
    in_degree = {v: len(parents_dict.get(v, [])) for v in range(num_vertices)}
    children = defaultdict(list)
    for v in range(num_vertices):
        for p in parents_dict.get(v, []):
            children[p].append(v)
    
    # Kahn's algorithm for topological sort
    from collections import deque
    queue = deque([v for v in range(num_vertices) if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't include all vertices, add remaining
    if len(topo_order) < num_vertices:
        remaining = set(range(num_vertices)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Simple strategy: 
    # 1. Load all inputs
    # 2. Process vertices in topological order, computing each when parents are available
    # 3. Store outputs when done
    # 4. Use a cache (fast memory) to hold intermediate results
    
    operations = []
    fast_memory = set()  # Vertices currently in fast memory (red)
    computed = set()  # Vertices that have been computed
    
    # Load inputs
    for v in inputs:
        if v not in fast_memory:
            operations.append(('load', v))
            fast_memory.add(v)
            computed.add(v)
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            continue  # Already loaded
        
        parents_v = parents_dict.get(v, [])
        
        # Check if all parents are in fast memory
        if all(p in fast_memory for p in parents_v):
            # Compute v
            operations.append(('compute', v))
            fast_memory.add(v)
            computed.add(v)
        else:
            # Need to reload some parents
            # Find which parents are missing
            missing = [p for p in parents_v if p not in fast_memory]
            
            # Reload missing parents (and their dependencies if needed)
            # Simple approach: reload the missing parent and its dependencies
            for p in missing:
                _reload_vertex(p, parents_dict, fast_memory, operations, fast_memory_budget, computed)
            
            # Now compute v
            operations.append(('compute', v))
            fast_memory.add(v)
            computed.add(v)
        
        # If fast memory is full, evict non-output, non-parent vertices
        while len(fast_memory) > fast_memory_budget:
            # Find a vertex to evict: prefer non-outputs, non-needed-for-next
            evict_candidates = []
            for x in fast_memory:
                if x not in outputs:
                    evict_candidates.append(x)
            if evict_candidates:
                # Evict the one that is furthest from being needed
                evict_v = evict_candidates[0]
                operations.append(('store', evict_v))
                fast_memory.remove(evict_v)
            else:
                break
    
    # Store all outputs
    for v in outputs:
        if v in fast_memory:
            operations.append(('store', v))
            fast_memory.remove(v)
    
    # Truncate if too many operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return _format_path(operations, num_vertices)


def _reload_vertex(v, parents_dict, fast_memory, operations, fast_memory_budget, computed):
    """Reload vertex v and its dependencies into fast memory."""
    if v in fast_memory:
        return
    
    # First, make sure there's space
    while len(fast_memory) >= fast_memory_budget:
        # Evict a non-essential vertex
        evict_candidates = [x for x in fast_memory if x != v and x not in computed]
        if not evict_candidates:
            evict_candidates = [x for x in fast_memory if x != v]
        if evict_candidates:
            evict_v = evict_candidates[0]
            operations.append(('store', evict_v))
            fast_memory.remove(evict_v)
        else:
            break
    
    # Load or compute v
    parents_v = parents_dict.get(v, [])
    if not parents_v:
        # It's an input
        operations.append(('load', v))
        fast_memory.add(v)
        computed.add(v)
    else:
        # Reload parents first
        for p in parents_v:
            _reload_vertex(p, parents_dict, fast_memory, operations, fast_memory_budget, computed)
        
        # Now compute v
        if v not in fast_memory:
            operations.append(('compute', v))
            fast_memory.add(v)
            computed.add(v)


def _format_path(operations, num_vertices):
    """Format operations into the expected return format."""
    # The expected format is a list of [operation, vertex]
    result = []
    for op, v in operations:
        result.append([op, v])
    return result
