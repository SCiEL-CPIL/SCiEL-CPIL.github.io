"""DAG pebbling with communication minimization under bounded fast memory."""
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Minimize communication in DAG pebbling under bounded fast memory.
    
    dag: dict with 'vertices' (v -> list of parent vertices) and 'outputs' (list of output vertices)
    fast_memory_budget: int, maximum number of red pebbles (values in fast memory)
    
    Returns: list of [operation, vertex] operations
    """
    start_time = time.time()
    DEADLINE = 80.0
    
    # Parse DAG structure
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertex_parents = {k: list(v) for k, v in dag['vertices'].items()}
        elif 'parents' in dag:
            vertex_parents = {k: list(v) for k, v in dag['parents'].items()}
        else:
            vertex_parents = {k: list(v) for k, v in dag.items()}
        
        outputs = list(dag.get('outputs', dag.get('sinks', [])))
        if not outputs:
            all_v = set(vertex_parents.keys())
            has_child = set()
            for v, ps in vertex_parents.items():
                for p in ps:
                    has_child.add(p)
            outputs = [v for v in all_v if v not in has_child]
    else:
        vertex_parents = {v: list(p) for v, p in dag}
        all_v = set(vertex_parents.keys())
        has_child = set()
        for v, ps in vertex_parents.items():
            for p in ps:
                has_child.add(p)
        outputs = [v for v in all_v if v not in has_child]
    
    all_vertices = set(vertex_parents.keys())
    for v, ps in vertex_parents.items():
        for p in ps:
            all_vertices.add(p)
    
    # Build children map
    children_map = defaultdict(list)
    for v, ps in vertex_parents.items():
        for p in ps:
            children_map[p].append(v)
    
    # Topological order
    in_deg = {v: len(vertex_parents.get(v, [])) for v in all_vertices}
    q = deque([v for v in all_vertices if in_deg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    if len(topo) < len(all_vertices):
        topo = list(vertex_parents.keys())
    
    # Count how many times each vertex is used as a parent
    parent_usage = defaultdict(int)
    for v, ps in vertex_parents.items():
        for p in ps:
            parent_usage[p] += 1
    
    # Greedy pebbling with memory management
    fast_mem = set()  # vertices currently in fast memory
    computed = set()  # vertices that have been computed
    stored = set()    # vertices that have been stored to slow memory
    ops = []
    
    # Track remaining uses for each vertex
    remaining_uses = dict(parent_usage)
    
    for v in topo:
        if time.time() - start_time > DEADLINE:
            break
        
        parents = vertex_parents.get(v, [])
        
        if not parents:
            # Input vertex: load into fast memory
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
        else:
            # Ensure all parents are in fast memory
            for p in parents:
                if p not in fast_mem:
                    # Need to load p from slow memory
                    ops.append(['load', p])
                    fast_mem.add(p)
            
            # Compute v
            ops.append(['compute', v])
            computed.add(v)
            fast_mem.add(v)
        
        # Update remaining uses for parents (they were just used)
        for p in parents:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # Memory management: if over budget, evict
        while len(fast_mem) > fast_memory_budget:
            # Find vertex to evict: lowest remaining_uses, then not an output
            best_evict = None
            best_score = float('inf')
            for fm_v in fast_mem:
                ru = remaining_uses.get(fm_v, 0)
                is_output = 1 if fm_v in set(outputs) else 0
                score = ru * 100 + is_output
                if score < best_score:
                    best_score = score
                    best_evict = fm_v
            
            if best_evict is None:
                break
            
            if best_evict in stored:
                ops.append(['delete', best_evict])
            else:
                ops.append(['store', best_evict])
                stored.add(best_evict)
                ops.append(['delete', best_evict])
            fast_mem.discard(best_evict)
    
    # Ensure all outputs end up stored (blue)
    output_set = set(outputs)
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
            stored.add(v)
        elif v not in stored:
            ops.append(['store', v])
            stored.add(v)
    
    return ops

# EVOLVE-BLOCK-END
