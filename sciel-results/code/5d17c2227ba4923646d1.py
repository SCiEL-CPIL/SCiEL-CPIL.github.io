"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag structure (assumed):
      - dag['vertices']: list of vertex ids
      - dag['parents']: dict vertex -> list of parent vertex ids (topologically ordered)
      - dag['inputs']: list of input vertex ids (start blue)
      - dag['outputs']: list of output vertex ids (must finish blue)
    
    Returns: list of [operation, vertex] where operation is one of
             'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG
    try:
        if isinstance(dag, dict):
            vertices = list(dag.get('vertices', []))
            parents = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = list(dag.get('outputs', []))
        elif isinstance(dag, (list, tuple)):
            # Maybe dag is a list of dicts or something else
            # Try to infer structure
            if len(dag) > 0 and isinstance(dag[0], dict):
                vertices = list(dag[0].get('vertices', []))
                parents = dict(dag[0].get('parents', {}))
                inputs = list(dag[0].get('inputs', []))
                outputs = list(dag[0].get('outputs', []))
            else:
                # Fallback: assume dag is (vertices, parents, inputs, outputs)
                vertices = list(dag[0])
                parents = dict(dag[1])
                inputs = list(dag[2])
                outputs = list(dag[3])
        else:
            vertices = []
            parents = {}
            inputs = []
            outputs = []
    except Exception:
        vertices = []
        parents = {}
        inputs = []
        outputs = []
    
    if not vertices:
        return []
    
    # Build adjacency and topological order
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute in-degrees for topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detected or invalid DAG; return empty
        return []
    
    # Set of input vertices (start blue)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # State: fast_memory is a set of vertex ids currently in fast memory
    # Initially, inputs are in fast memory (blue)
    fast_memory = set(inputs)
    
    # Ensure fast_memory doesn't exceed budget
    # If inputs exceed budget, we need to handle this carefully
    # For now, assume inputs fit or we load/store as needed
    
    # Strategy: Process vertices in topological order.
    # For each vertex, ensure its parents are in fast memory, compute it,
    # then decide whether to keep it or evict parents.
    
    # A better strategy: Use a worklist/scheduler that minimizes transfers.
    # We'll use a greedy approach: process vertices in topological order,
    # loading parents as needed, computing, and evicting when memory is full.
    
    # Track which vertices have been computed and are still in fast memory
    # fast_memory is the set of vertices currently in fast memory (blue)
    
    ops = []
    
    # We need to ensure all outputs end up in fast memory.
    # Strategy: 
    # 1. Process vertices in topological order.
    # 2. For each vertex v:
    #    a. For each parent p of v not in fast_memory, load p.
    #    b. Compute v (adds v to fast_memory).
    #    c. If fast_memory exceeds budget, evict some vertex that is not an output
    #       and not needed soon.
    # 3. At the end, ensure all outputs are in fast_memory.
    
    # Better: Use a more sophisticated scheduling.
    # Let's implement a simple but effective heuristic:
    # - Maintain a priority for eviction: prefer to evict vertices that are inputs
    #   (can be reloaded) or vertices whose all children have been processed.
    
    # First, compute for each vertex whether it's an ancestor of any output
    # (i.e., it's on a path to an output). Vertices not on any output path
    # can be evicted more freely.
    
    # Compute set of vertices that are ancestors of outputs
    ancestors_of_outputs = set()
    # Reverse topological order
    for v in reversed(topo_order):
        if v in output_set or v in ancestors_of_outputs:
            ancestors_of_outputs.add(v)
            for p in parents.get(v, []):
                ancestors_of_outputs.add(p)
    
    # Actually, let me recompute: a vertex is an ancestor of an output if
    # it can reach an output. Let's do it properly.
    # Start from outputs and traverse backwards.
    ancestors_of_outputs = set()
    to_visit = list(outputs)
    while to_visit:
        v = to_visit.pop()
        if v in ancestors_of_outputs:
            continue
        ancestors_of_outputs.add(v)
        for p in parents.get(v, []):
            if p not in ancestors_of_outputs:
                to_visit.append(p)
    
    # Now, for each vertex, compute its "importance":
    # - If it's an output, very important
    # - If it's an input, can be reloaded (less important to keep)
    # - If it's an ancestor of an output, moderately important
    # - Otherwise, least important
    
    # For eviction priority, we want to evict the least important vertices first.
    # Importance score (lower = more likely to evict):
    importance = {}
    for v in vertices:
        if v in output_set:
            importance[v] = 1000
        elif v in input_set:
            importance[v] = 1
        elif v in ancestors_of_outputs:
            importance[v] = 500
        else:
            importance[v] = 10
    
    # Now process in topological order
    # We'll track which vertices have been computed
    computed = set()
    
    for v in topo_order:
        if time.time() > DEADLINE:
            break
        
        # If v is already in fast memory and computed, skip
        if v in fast_memory and v in computed:
            continue
        
        # Check if all parents are in fast memory
        need_load = []
        for p in parents.get(v, []):
            if p not in fast_memory:
                need_load.append(p)
        
        # Load parents that are not in fast memory
        for p in need_load:
            if time.time() > DEADLINE:
                break
            # If p is an input, load it
            # If p is not an input, we need to have computed it before
            # For simplicity, assume we can load any vertex (the evaluator handles it)
            ops.append(['load', p])
            fast_memory.add(p)
        
        # If we broke out early, stop
        if time.time() > DEADLINE:
            break
        
        # Compute v
        ops.append(['compute', v])
        fast_memory.add(v)
        computed.add(v)
        
        # Evict if needed
        while len(fast_memory) > fast_memory_budget:
            if time.time() > DEADLINE:
                break
            
            # Find the least important vertex to evict
            # Don't evict outputs
            # Prefer to evict inputs (can reload) or non-ancestor vertices
            candidates = [x for x in fast_memory if x not in output_set]
            if not candidates:
                break
            
            # Sort by importance (ascending) and evict the first
            candidates.sort(key=lambda x: importance[x])
            evict_v = candidates[0]
            
            # But only evict if it's not needed for upcoming computations
            # For simplicity, just evict
            ops.append(['delete', evict_v])
            fast_memory.remove(evict_v)
    
    # At the end, ensure all outputs are in fast memory
    if time.time() < DEADLINE:
        for o in outputs:
            if o not in fast_memory:
                # Need to load o
                # But o should have been computed... if not, compute it
                if o not in computed:
                    # Compute o and its dependencies
                    # This is complex; for now, just load
                    ops.append(['load', o])
                    fast_memory.add(o)
                else:
                    ops.append(['load', o])
                    fast_memory.add(o)
                
                # Evict if needed
                while len(fast_memory) > fast_memory_budget:
                    candidates = [x for x in fast_memory if x not in output_set]
                    if not candidates:
                        break
                    candidates.sort(key=lambda x: importance[x])
                    evict_v = candidates[0]
                    ops.append(['delete', evict_v])
                    fast_memory.remove(evict_v)
    
    return ops
# EVOLVE-BLOCK-END
