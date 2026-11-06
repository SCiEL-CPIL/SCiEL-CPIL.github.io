import time
import heapq
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, leaving 10s margin for 60s eval
    
    # 1. Parse DAG structure
    if isinstance(dag, (tuple, list)) and len(dag) == 2:
        parents_struct = dag[0]
        outputs = list(dag[1])
    else:
        parents_struct = dag
        outputs = None
        
    # Normalize parents to dict
    if isinstance(parents_struct, list):
        parents_map = {i: list(parents_struct[i]) for i in range(len(parents_struct))}
    elif isinstance(parents_struct, dict):
        parents_map = {k: list(v) for k, v in parents_struct.items()}
    else:
        # Fallback, shouldn't happen
        parents_map = {}
        
    # Determine vertices
    vertices = set(parents_map.keys())
    for v, ps in parents_map.items():
        for p in ps:
            vertices.add(p)
    if outputs:
        for o in outputs:
            vertices.add(o)
            
    # Build children map
    children_map = {v: [] for v in vertices}
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
            
    # Determine outputs if not provided
    if outputs is None:
        outputs = [v for v in vertices if not children_map[v]]
    else:
        outputs = list(outputs)
        
    # Inputs
    inputs = [v for v in vertices if not parents_map.get(v, [])]
    
    # Topological Sort
    in_deg = {v: len(parents_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_deg[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
                
    # If topo_order length < len(vertices), there's a cycle or disconnected component not reachable from inputs?
    # Assuming valid DAG.
    
    # Precompute last_child_index for each vertex
    # last_child_index[v] = max(index of child in topo_order)
    # If no children, -1
    index_in_topo = {v: i for i, v in enumerate(topo_order)}
    last_child_index = {v: -1 for v in vertices}
    for v in vertices:
        for c in children_map[v]:
            if index_in_topo.get(c, -1) > last_child_index[v]:
                last_child_index[v] = index_in_topo[c]
                
    # State
    blue = set(inputs)
    fast = set()
    computed = set(inputs)
    children_count = {v: len(children_map[v]) for v in vertices}
    
    ops = []
    
    def store_vertex(v):
        if v in fast:
            fast.remove(v)
            blue.add(v)
            ops.append(["store", v])
            
    # Process in topo order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        if v in computed:
            continue
            
        # Ensure parents are in fast
        for p in parents_map.get(v, []):
            if p in blue:
                blue.remove(p)
                fast.add(p)
                ops.append(["load", p])
            # p must be in fast or blue. If not, it's an error.
            
        # Compute v
        fast.add(v)
        computed.add(v)
        ops.append(["compute", v])
        
        # Update children count for parents
        for p in parents_map.get(v, []):
            children_count[p] -= 1
            if children_count[p] == 0 and p in fast:
                store_vertex(p)
                
        # Enforce budget
        while len(fast) > fast_memory_budget:
            if time.time() > deadline:
                break
            # Evict candidate
            # Prefer nodes with children_count == 0
            # Then prefer nodes with largest last_child_index (furthest future need)
            candidates = [x for x in fast if x != v]
            if not candidates:
                break
                
            # Score: 
            # 1. children_count == 0 (dead) -> score 0
            # 2. children_count > 0 -> score 1
            # Among same score, evict largest last_child_index?
            # Actually, if children_count == 0, last_child_index is irrelevant (it's past or -1).
            # If children_count > 0, we want to evict the one needed furthest.
            
            def evict_score(x):
                if children_count[x] == 0:
                    return (0, -last_child_index[x]) # Dead nodes first. Among dead, doesn't matter much, maybe keep most recent?
                else:
                    return (1, -last_child_index[x]) # Alive nodes. Evict largest last_child_index (most negative score).
                    
            # We want to minimize the score tuple?
            # (0, ...) < (1, ...)
            # For (1, -L), larger L means smaller -L. So we pick the one with smallest -L, i.e., largest L.
            # This matches "evict furthest future need".
            
            v_to_store = min(candidates, key=evict_score)
            store_vertex(v_to_store)
            
    # Ensure outputs are blue
    for o in outputs:
        if time.time() > deadline:
            break
        if o in fast:
            store_vertex(o)
            
    return ops
