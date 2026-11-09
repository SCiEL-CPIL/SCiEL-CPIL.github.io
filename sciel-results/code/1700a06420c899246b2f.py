import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 45.0

    # Parse DAG
    vertices = []
    parents_map = {}
    is_output_map = {}
    children_map = defaultdict(list)
    
    if isinstance(dag, dict):
        for v_id, info in dag.items():
            if isinstance(info, dict):
                parents = info.get('parents', [])
                is_out = info.get('is_output', False)
            else:
                parents, is_out = info
            vertices.append(v_id)
            parents_map[v_id] = parents
            is_output_map[v_id] = is_out
            for p in parents:
                children_map[p].append(v_id)
    elif isinstance(dag, list):
        for item in dag:
            if len(item) == 3:
                v_id, parents, is_out = item
            elif len(item) == 2:
                v_id, parents = item
                is_out = False
            else:
                continue
            vertices.append(v_id)
            parents_map[v_id] = parents
            is_output_map[v_id] = is_out
            for p in parents:
                children_map[p].append(v_id)
    else:
        return []

    # Topological Sort
    in_degree = {v: len(parents_map[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for w in children_map[u]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)
                
    if len(topo_order) != len(vertices):
        # Cycle or error, fallback to simple order
        topo_order = vertices

    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    # State: 0=Empty, 1=Red, 2=Blue
    state = {v: 1 for v in vertices}
    for v in vertices:
        if not parents_map[v]:
            state[v] = 2 # Inputs start Blue
    
    fast_memory = set(v for v in vertices if state[v] == 2)
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    ops = []
    
    for v in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are Blue
        for p in parents_map[v]:
            if state[p] != 2:
                if state[p] == 1:
                    ops.append(['load', p])
                    state[p] = 2
                    fast_memory.add(p)
                elif state[p] == 0:
                    # Error: deleted prematurely. 
                    # Should not happen with correct logic.
                    pass
        
        # 2. Compute v
        if parents_map[v]:
            if state[v] != 2:
                ops.append(['compute', v])
                state[v] = 2
                fast_memory.add(v)
        
        # 3. Update remaining children for parents of v
        for p in parents_map[v]:
            remaining_children[p] -= 1
        
        # 4. Cleanup and Eviction
        # First, delete any vertices that are no longer needed (non-outputs)
        to_delete = []
        for u in list(fast_memory):
            if remaining_children[u] == 0 and not is_output_map[u]:
                to_delete.append(u)
                
        for u in to_delete:
            ops.append(['delete', u])
            state[u] = 0
            fast_memory.remove(u)
            
        # Second, evict if over budget
        while len(fast_memory) > fast_memory_budget:
            # Find best candidate to store
            # Candidates: all in fast_memory
            # We want to evict the one with the furthest next use
            best_candidate = None
            max_next_use = -1
            
            current_idx = topo_index[v]
            
            for u in fast_memory:
                if remaining_children[u] == 0:
                    # Must be an output (non-outputs were deleted)
                    # Next use is "End"
                    nu = float('inf')
                else:
                    # Find min topo_index of remaining children
                    min_child_idx = float('inf')
                    for c in children_map[u]:
                        if topo_index[c] > current_idx:
                            if topo_index[c] < min_child_idx:
                                min_child_idx = topo_index[c]
                    nu = min_child_idx
                    
                if nu > max_next_use:
                    max_next_use = nu
                    best_candidate = u
                    
            if best_candidate is None:
                break
                
            u = best_candidate
            ops.append(['store', u])
            state[u] = 1
            fast_memory.remove(u)
            
    # Final check: Outputs must be Blue
    for v in vertices:
        if is_output_map[v]:
            if state[v] != 2:
                if state[v] == 1:
                    ops.append(['load', v])
                    state[v] = 2
                    fast_memory.add(v)
                # If state[v] == 0, we lost it. 
                # This implies we deleted an output, which we don't do.
                
    return ops
