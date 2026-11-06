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

def _layered_pebble(dag, fast_memory_budget, deadline):
    """
    Layered pebbling strategy:
    1. Identify layers based on topological depth.
    2. Process layer by layer.
    3. Within a layer, compute vertices.
    4. Between layers, evict vertices that are no longer needed (their children are in later layers)
       to make room for the next layer's inputs.
    
    This reduces the need for complex per-vertex eviction decisions by batching them.
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
    
    # Compute depth for each vertex (longest path from any input)
    # depth[v] = max(depth[p] for p in parents[v]) + 1
    # For inputs, depth = 0
    depth = {}
    topo_order = _topo_sort(vertices, parents_map)
    for v in topo_order:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents_map[v]) + 1
    
    # Group vertices by depth (layer)
    layers = defaultdict(list)
    for v in vertices:
        layers[depth[v]].append(v)
    
    max_depth = max(depth.values()) if depth else 0
    
    # Precompute which parents are needed for each vertex
    # And which children exist for each vertex
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # remaining_children_count: how many children of v are still unprocessed
    # This helps decide if v can be evicted after its last child is computed.
    # Actually, a vertex v can be evicted once ALL its children have been computed.
    # We can track this with a counter.
    
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    fast_memory = set()
    operations = []
    
    # We process layers 0 to max_depth.
    # Layer 0 contains inputs.
    # Layer k contains vertices whose longest path from input is k.
    
    # Strategy:
    # For each layer L from 0 to max_depth:
    #   1. Ensure all parents of vertices in layer L are in fast_memory.
    #      Parents are in layer L-1 (or earlier, but typically L-1 for layered DAGs).
    #      Actually, parents can be in any earlier layer.
    #      But since we process in topological order, all parents of L are already computed.
    #      We need to load them if they were evicted.
    #   2. Compute all vertices in layer L.
    #   3. After computing layer L, we can evict vertices from layer L-1 (and earlier)
    #      that have no remaining children in layers > L.
    #      i.e., if remaining_children[v] == 0, v can be evicted.
    
    # However, we must be careful:
    # - Inputs (layer 0) can be evicted (delete) because they are in slow memory.
    # - Computed vertices (layer > 0) must be stored (store) if evicted, unless they are outputs?
    #   No, if they are evicted, they must be stored to slow memory.
    #   If they are outputs, they must finish blue, so they must be stored at the end anyway.
    
    # Let's refine the eviction policy:
    # After processing layer L, we have just computed layer L.
    # Vertices in layer L-1 have had their children in layer L processed.
    # Some vertices in layer L-1 might still have children in layer L+1 or later?
    # No, in a layered DAG where edges only go from layer i to layer i+1, 
    # children of layer L-1 are in layer L.
    # But the DAG might not be strictly layered (edges can skip layers).
    # The problem statement says "three layered DAGs", implying a strict structure,
    # but we should be robust.
    
    # Robust approach:
    # After processing a vertex v, decrement remaining_children for its parents?
    # No, remaining_children tracks how many children of v are unprocessed.
    # When we compute a child c of v, we decrement remaining_children[v].
    
    # Let's stick to the per-vertex processing but with a smarter eviction heuristic:
    # Evict a vertex v if:
    # 1. remaining_children[v] == 0 (all children processed)
    # 2. v is not an output (or if it is, and we are at the end, we store it)
    # 3. We need space.
    
    # Actually, the previous greedy approach was already doing something similar.
    # The key improvement for layered DAGs is that we can batch evictions after each layer.
    
    # Let's try a different angle:
    # Keep a set of "live" vertices.
    # A vertex is live if it has been computed and has at least one unprocessed child.
    # Or if it is an input that has been loaded and has at least one unprocessed child.
    
    # When we compute a vertex v:
    # - It becomes live.
    # - For each parent p of v, decrement remaining_children[p].
    #   If remaining_children[p] becomes 0, p is no longer live (can be evicted).
    
    # So, after each compute step, we can check if any parents became dead and evict them if memory is tight.
    
    # This is essentially the same as the greedy approach but with a clearer "dead" concept.
    
    # Let's implement this "dead vertex" eviction strategy.
    
    fast_memory = set()
    operations = []
    
    # remaining_children[v] = number of children of v not yet computed
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    # We process in topological order.
    for v in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in fast_memory
        for p in parents_map.get(v, []):
            if p not in fast_memory:
                # Need to load p
                # Make room if necessary
                while len(fast_memory) >= fast_memory_budget:
                    # Find a vertex to evict
                    # Prefer vertices that are "dead" (remaining_children == 0)
                    # Among dead vertices, prefer non-outputs? Or just any dead vertex.
                    # If no dead vertices, evict the one with lowest remaining_children?
                    
                    best_victim = None
                    min_score = float('inf')
                    
                    for x in fast_memory:
                        # Don't evict parents of current vertex v that are needed
                        if x in parents_map.get(v, []):
                            continue
                        
                        # Score: 
                        # If remaining_children[x] == 0, score = 0 (best to evict)
                        # Else, score = remaining_children[x]
                        # Tie-breaker: prefer evicting inputs (delete is cheaper than store? No, both are 1 transfer. But delete is safer?)
                        # Actually, store and delete are both 1 transfer.
                        # But storing a computed vertex makes it available in slow memory, so if we need it again, we just load it.
                        # Deleting an input makes it only available in slow memory (it was there originally).
                        # So both are equivalent in terms of future load cost.
                        
                        score = remaining_children.get(x, 0)
                        
                        # If x is an output, we should be careful.
                        # If we evict an output, we must store it.
                        # If we don't evict it, it stays in fast memory until the end, then we store it.
                        # So evicting it early doesn't change the total store count for outputs.
                        
                        if score < min_score:
                            min_score = score
                            best_victim = x
                        elif score == min_score:
                            # Tie-breaker: prefer evicting vertices that are inputs?
                            # Or vertices that are not outputs?
                            # Let's prefer evicting non-outputs to keep outputs in fast memory (avoids potential issues?)
                            # Actually, it doesn't matter much.
                            if best_victim in output_set and x not in output_set:
                                best_victim = x
                            
                    if best_victim is None:
                        break
                    
                    # Evict best_victim
                    fast_memory.discard(best_victim)
                    if best_victim in input_vertices:
                        operations.append(['delete', best_victim])
                    else:
                        operations.append(['store', best_victim])
                
                if len(fast_memory) < fast_memory_budget:
                    operations.append(['load', p])
                    fast_memory.add(p)
        
        # 2. Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # 3. Update remaining_children for parents
        for p in parents_map.get(v, []):
            remaining_children[p] -= 1
        
    # 4. Finalize: Store all outputs that are in fast_memory
    for o in outputs:
        if o in fast_memory:
            operations.append(['store', o])
            fast_memory.discard(o)
    
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    try:
        result = _layered_pebble(dag, fast_memory_budget, deadline)
        
        # Basic validation: check length and structure
        if result and len(result) <= 20000:
            return result
        else:
            # Fallback to baseline if optimized version fails or is too long
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        # Fall back to baseline on any error
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
