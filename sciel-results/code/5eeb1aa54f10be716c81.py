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
        
    # If no inputs, assume all nodes without parents are inputs
    if not input_vertices:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Helper to generate a greedy solution (fallback and initial guess)
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
        
        # Process in topo order
        for v in topo_order:
            if v in input_vertices:
                continue
            # Load parents
            for p in parents.get(v, []):
                if p not in fast_mem and p in slow_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
            # Compute
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)
            
            # Evict if needed
            while len(fast_mem) > fast_memory_budget:
                # Simple heuristic: delete last added non-output
                evict = None
                for u in list(fast_mem):
                    if u not in outputs:
                        evict = u
                        break
                if evict is None:
                    evict = next(iter(fast_mem))
                ops.append(['delete', evict])
                fast_mem.discard(evict)
                
        # Store outputs
        for o in outputs:
            if o in fast_mem:
                ops.append(['store', o])
                slow_mem.add(o)
                fast_mem.discard(o)
        return ops

    # Initial greedy solution
    best_ops = generate_greedy()
    best_cost = sum(1 for op in best_ops if op[0] in ['load', 'store'])
    
    # Beam Search / Dijkstra over state space
    # State: (cost, fast_mem_frozen, computed_frozen, path)
    # We use a priority queue ordered by cost + heuristic
    
    # Heuristic: number of outputs not yet computed / 1.0
    def heuristic(computed):
        return len(outputs) - len(set(outputs).intersection(computed))
    
    start_state = (0, frozenset(), frozenset(), [])
    # (cost, fast_mem, computed, path)
    beam = [start_state]
    visited = {} # (fast_mem, computed) -> min_cost
    
    max_expansions = 5000
    expansions = 0
    
    while beam and expansions < max_expansions and not time_check():
        current_cost, fast_mem, computed, path = beam.pop(0)
        expansions += 1
        
        # Check goal
        if set(outputs).issubset(computed):
            # Need to ensure outputs are stored? 
            # The problem says "all outputs must finish blue". Blue means in slow memory?
            # "Inputs start blue... all outputs must finish blue".
            # So yes, outputs must be in slow memory at the end.
            # If an output is in fast_mem, we must store it.
            final_ops = list(path)
            final_fast = set(fast_mem)
            for o in outputs:
                if o in final_fast:
                    final_ops.append(['store', o])
                    final_fast.discard(o)
            final_cost = current_cost + sum(1 for op in final_ops if op[0] in ['load', 'store'])
            if final_cost < best_cost:
                best_cost = final_cost
                best_ops = final_ops
            continue
            
        # Generate neighbors
        new_beam = []
        
        # 1. Compute a vertex
        for v in all_vertices:
            if v in computed or v in fast_mem:
                continue
            if v in input_vertices:
                continue
            # Check if parents are in fast_mem
            if all(p in fast_mem for p in parents.get(v, [])):
                new_fast = fast_mem | {v}
                if len(new_fast) <= fast_memory_budget:
                    new_computed = computed | {v}
                    new_cost = current_cost # Compute is free
                    new_path = path + [['compute', v]]
                    key = (new_fast, new_computed)
                    if key not in visited or visited[key] > new_cost:
                        visited[key] = new_cost
                        new_beam.append((new_cost + heuristic(new_computed), new_cost, new_fast, new_computed, new_path))
                        
        # 2. Load a vertex
        for v in all_vertices:
            if v in fast_mem or v in computed:
                continue
            # Can load if it's an input (starts in slow) or was computed and stored/deleted
            # Actually, if v is not in computed, it must be an input to be loadable?
            # Or if it was computed, deleted, and is in slow?
            # My state doesn't track "in slow". 
            # Assumption: If v is not in computed, it's an input (in slow).
            # If v is in computed, it might be in slow (if stored) or nowhere (if deleted).
            # To simplify, we allow loading any v not in fast_mem.
            # Cost 1.
            new_fast = fast_mem | {v}
            if len(new_fast) <= fast_memory_budget:
                new_computed = computed
                new_cost = current_cost + 1
                new_path = path + [['load', v]]
                key = (new_fast, new_computed)
                if key not in visited or visited[key] > new_cost:
                    visited[key] = new_cost
                    new_beam.append((new_cost + heuristic(new_computed), new_cost, new_fast, new_computed, new_path))
                    
        # 3. Delete a vertex
        for v in fast_mem:
            if v in outputs:
                continue # Don't delete outputs unless we store them first? 
                         # Actually, if we delete an output, we lose it. 
                         # We must store it to slow first.
            new_fast = fast_mem - {v}
            new_computed = computed
            new_cost = current_cost
            new_path = path + [['delete', v]]
            key = (new_fast, new_computed)
            if key not in visited or visited[key] > new_cost:
                visited[key] = new_cost
                new_beam.append((new_cost + heuristic(new_computed), new_cost, new_fast, new_computed, new_path))
                
        # 4. Store a vertex
        for v in fast_mem:
            new_fast = fast_mem - {v}
            new_computed = computed
            new_cost = current_cost + 1
            new_path = path + [['store', v]]
            key = (new_fast, new_computed)
            if key not in visited or visited[key] > new_cost:
                visited[key] = new_cost
                new_beam.append((new_cost + heuristic(new_computed), new_cost, new_fast, new_computed, new_path))
                
        # Prune beam
        if new_beam:
            # Sort by estimated total cost
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:50] # Beam width 50

    return best_ops

# EVOLVE-BLOCK-END
