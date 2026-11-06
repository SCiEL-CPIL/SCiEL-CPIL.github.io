"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

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

    if not input_vertices:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Greedy Solution
    def generate_greedy():
        slow_mem = set(input_vertices)
        fast_mem = set()
        ops = []
        
        # Topological sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        q = deque([v for v in all_vertices if in_degree[v] == 0])
        topo_order = []
        while q:
            v = q.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    q.append(c)
        
        # Precompute "needed" status dynamically? 
        # For greedy, we can just check if a vertex is a parent of any uncomputed vertex.
        # Uncomputed vertices are those later in topo_order.
        
        computed = set()
        for v in topo_order:
            if v in input_vertices:
                continue
            
            # Load parents
            for p in parents.get(v, []):
                if p not in fast_mem:
                    if p in slow_mem:
                        ops.append(['load', p])
                        fast_mem.add(p)
                        slow_mem.discard(p)
                    else:
                        # Should not happen if we don't delete needed parents
                        # But if we did, we can't load it. 
                        # Greedy assumption: we never delete a parent that is still needed.
                        pass
            
            # Compute
            ops.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)
            
            # If output, store immediately
            if v in outputs:
                ops.append(['store', v])
                slow_mem.add(v)
                fast_mem.discard(v)
                
            # Evict if over budget
            while len(fast_mem) > fast_memory_budget:
                # Find a vertex to evict
                # Priority: 
                # 1. Not an output (or output already stored)
                # 2. Not a parent of any uncomputed vertex
                # 3. Oldest computed?
                
                evict_candidate = None
                best_score = -1
                
                for u in fast_mem:
                    # Score: higher is better to evict
                    score = 0
                    if u in outputs and u not in slow_mem:
                        score -= 10000 # Don't evict unstored outputs
                    elif u in outputs:
                        score += 100 # Can evict stored outputs
                    else:
                        # Check if needed
                        needed = False
                        for w in topo_order:
                            if w in computed:
                                continue
                            if u in parents.get(w, []):
                                needed = True
                                break
                        if needed:
                            score -= 100 # Don't evict needed parents
                        else:
                            score += 100 # Can evict
                
                    if score > best_score:
                        best_score = score
                        evict_candidate = u
                        
                if evict_candidate is None:
                    # Fallback: evict any non-output
                    for u in fast_mem:
                        if u not in outputs:
                            evict_candidate = u
                            break
                    if evict_candidate is None:
                        evict_candidate = next(iter(fast_mem))
                        
                ops.append(['delete', evict_candidate])
                fast_mem.discard(evict_candidate)

        # Store any remaining outputs in fast_mem
        for o in outputs:
            if o in fast_mem:
                ops.append(['store', o])
                slow_mem.add(o)
                fast_mem.discard(o)
                
        return ops

    best_ops = generate_greedy()
    best_cost = sum(1 for op in best_ops if op[0] in ['load', 'store'])
    
    # A* Search
    # State: (cost, F_frozen, S_frozen, path)
    # Heuristic: number of outputs not in S
    def heuristic(S):
        return len(set(outputs) - S)
        
    start_F = frozenset()
    start_S = frozenset(input_vertices)
    
    # (priority, cost, F, S, path)
    heap = [(heuristic(start_S), 0, start_F, start_S, [])]
    visited = {} # (F, S) -> min_cost
    
    max_expansions = 10000
    expansions = 0
    
    while heap and expansions < max_expansions and not time_check():
        priority, cost, F, S, path = heapq.heappop(heap)
        expansions += 1
        
        key = (F, S)
        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost
        
        # Check goal
        if set(outputs).issubset(S):
            if cost < best_cost:
                best_cost = cost
                best_ops = list(path)
            continue
            
        # Generate neighbors
        
        # 1. Compute (Cost 0)
        for v in all_vertices:
            if v in F or v in input_vertices:
                continue
            # Check if parents are in F
            if all(p in F for p in parents.get(v, [])):
                new_F = F | {v}
                new_S = S
                new_cost = cost
                new_path = path + [['compute', v]]
                new_key = (new_F, new_S)
                if new_key not in visited or visited[new_key] > new_cost:
                    heapq.heappush(heap, (new_cost + heuristic(new_S), new_cost, new_F, new_S, new_path))
                    
        # 2. Delete (Cost 0)
        for v in F:
            # Can delete if v is in S (already safe) or v is not an output
            if v in S or v not in outputs:
                new_F = F - {v}
                new_S = S
                new_cost = cost
                new_path = path + [['delete', v]]
                new_key = (new_F, new_S)
                if new_key not in visited or visited[new_key] > new_cost:
                    heapq.heappush(heap, (new_cost + heuristic(new_S), new_cost, new_F, new_S, new_path))
                    
        # 3. Load (Cost 1)
        for v in S:
            if v not in F:
                new_F = F | {v}
                new_S = S - {v}
                new_cost = cost + 1
                new_path = path + [['load', v]]
                new_key = (new_F, new_S)
                if new_key not in visited or visited[new_key] > new_cost:
                    heapq.heappush(heap, (new_cost + heuristic(new_S), new_cost, new_F, new_S, new_path))
                    
        # 4. Store (Cost 1)
        for v in F:
            if v not in S:
                new_F = F - {v}
                new_S = S | {v}
                new_cost = cost + 1
                new_path = path + [['store', v]]
                new_key = (new_F, new_S)
                if new_key not in visited or visited[new_key] > new_cost:
                    heapq.heappush(heap, (new_cost + heuristic(new_S), new_cost, new_F, new_S, new_path))

    return best_ops

# EVOLVE-BLOCK-END
