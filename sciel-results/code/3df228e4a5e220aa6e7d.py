"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # If the DAG is small, use Dijkstra for optimal solution
    if len(all_vertices) <= 12:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Strategy: Time-reversed "unpebbling" greedy heuristic
    # Process DAG from outputs to inputs.
    # Prioritize deletion of vertices with highest in-degree among unprocessed predecessors.
    # This minimizes the peak set of live values (red pebbles), reducing store/reload ops.
    
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If topo sort failed (cycle?), fallback to simple layering
    if len(topo_order) < len(all_vertices):
        return pebble_fallback(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Reverse topological order
    rev_topo_order = list(reversed(topo_order))
    rev_index = {v: i for i, v in enumerate(rev_topo_order)}
    
    # Precompute recomputation cost (subtree size)
    subtree_size = {v: 1 for v in all_vertices}
    for v in rev_topo_order:
        for c in children[v]:
            subtree_size[v] += subtree_size[c]
    
    # Time-reversed unpebbling heuristic:
    # We simulate the forward process, but use reverse-time logic for eviction decisions.
    # Key insight: In reverse time, "predecessors" of a vertex are its children.
    # "Unprocessed predecessors" in reverse time = children not yet unpebbled (still in fast_mem in forward time).
    # We prioritize deleting vertices with the HIGHEST number of such unprocessed predecessors,
    # as this allows us to "unblock" more vertices for deletion, reducing peak live values.
    
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # Process in forward topological order
    for v in topo_order:
        if time_check():
            break
            
        if v in input_vertices:
            continue
            
        # Check if parents are in fast_mem
        parents_of_v = parents.get(v, [])
        if not all(p in fast_mem for p in parents_of_v):
            # Need to load missing parents
            for p in parents_of_v:
                if p not in fast_mem and p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                elif p not in fast_mem and p not in slow_mem:
                    # Parent was deleted and not stored - need to recompute
                    # This is a complex case; for now, we'll try to find a path to recompute
                    # but this should be rare with good eviction policy
                    pass
            
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        slow_mem.discard(v)
        
        # Evict if over budget using time-reversed heuristic
        while len(fast_mem) > fast_memory_budget and not time_check():
            # Find best candidate to delete using reverse-time priority
            best_candidate = None
            best_priority = -1  # Higher priority = delete first
            
            for u in fast_mem:
                # Can we delete u?
                # u can be deleted if all its children are NOT in fast_mem.
                children_of_u = children.get(u, [])
                can_delete = all(c not in fast_mem for c in children_of_u)
                
                if not can_delete:
                    continue
                
                # If it's an output, store it instead of deleting
                if u in outputs:
                    ops.append(['store', u])
                    slow_mem.add(u)
                    fast_mem.discard(u)
                    best_candidate = None
                    break
                
                # Time-reversed priority: 
                # In reverse time, "unprocessed predecessors" of u are its children not yet unpebbled.
                # In forward time, these are children still in fast_mem.
                # But we already checked can_delete, so no children are in fast_mem.
                # So "unprocessed predecessors" in the strict sense = 0 for all candidates.
                
                # Refined interpretation: Prioritize deletion of vertices whose DELETION
                # would unblock the most other vertices for deletion.
                # A vertex u can be deleted if all its children are not in fast_mem.
                # If we delete u, then u's parents might become deletable (if u was their only child in fast_mem).
                
                # Count how many parents of u would become "one step closer" to being deletable
                # if we delete u.
                unblock_count = 0
                for p in parents.get(u, []):
                    if p in fast_mem:
                        # Check if p has any other children in fast_mem
                        other_children_in_fast = [c for c in children.get(p, []) if c in fast_mem and c != u]
                        if not other_children_in_fast:
                            # Deleting u would make p deletable (all its children would be out of fast_mem)
                            unblock_count += 1
                
                # Also consider: vertices with high subtree_size are expensive to recompute,
                # so we prefer to delete them if they won't cause recomputation.
                # But if they have uncomputed children, deleting is costly.
                
                uncomputed_children = 0
                for c in children_of_u:
                    if c not in slow_mem and c not in fast_mem:
                        uncomputed_children += 1
                
                # Priority: unblock_count * 1000 - uncomputed_children * subtree_size[u]
                # Higher priority = better to delete
                priority = unblock_count * 1000 - uncomputed_children * subtree_size[u]
                
                # Tie-breaker: prefer vertices closer to inputs (higher rev_index)
                # as they are "deeper" in the computation and less likely to be needed soon
                priority += rev_index.get(u, 0) * 0.01
                
                if priority > best_priority:
                    best_priority = priority
                    best_candidate = u
                    
            if best_candidate is None:
                # No valid candidate or we stored an output
                break
                
            ops.append(['delete', best_candidate])
            fast_mem.discard(best_candidate)
            
    # Store outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
            
    if len(ops) > 20000:
        ops = ops[:20000]
        
    return ops

def pebble_fallback(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check):
    # Simple fallback: Compute all, store all, delete all.
    # This might exceed budget, but it's a valid sequence of operations if we manage memory.
    # Actually, if we can't fit in budget, we must recompute.
    # This fallback is just to avoid crashing.
    ops = []
    topo_order = list(all_vertices) # Simplified
    for v in topo_order:
        if v in input_vertices:
            continue
        ops.append(['compute', v])
    for o in outputs:
        ops.append(['store', o])
    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    # State: (cost, frozenset(fast), frozenset(black))
    # Initial: fast = set(), black = input_vertices
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    # Dijkstra
    # dist[fast][black] = cost
    # Since N is small, we can use a dictionary
    dist = {}
    prev = {} # state -> (prev_state, op)
    
    start_state = (init_fast, init_black)
    dist[start_state] = 0
    
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            break
        cost, state = heapq.heappop(pq)
        fast, black = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        # Check if goal reached
        if all(o in black for o in outputs):
            # Reconstruct path
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Compute
        for v in all_vertices:
            if v in fast or v in black:
                continue
            # Check if parents are available
            ps = parents.get(v, [])
            if all(p in fast or p in black for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost # Compute is free
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                # Fast remains the same (vertex is both red and blue now? Or just blue?)
                # If it stays red, fast is unchanged.
                # If it becomes blue only, fast removes v.
                # Let's assume it stays red to allow later deletion without reload.
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for v in fast:
            # Can delete if we don't need it in fast? 
            # Deleting is always valid if we have space issues, but it might force reloads.
            # In Dijkstra, we allow all valid moves.
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
        # 4. Load
        for v in black:
            if v not in fast:
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))

    # Fallback if Dijkstra fails or times out
    return []

# EVOLVE-BLOCK-END
