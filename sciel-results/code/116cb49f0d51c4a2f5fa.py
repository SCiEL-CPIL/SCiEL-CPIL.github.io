import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # 80s limit for 90s kill
    
    # Parse DAG
    # Assume dag is a dict with keys: 'vertices', 'parents', 'inputs', 'outputs'
    # Or similar structure.
    # We need to be robust to different formats.
    # Common formats:
    # dag = {
    #   'vertices': [0, 1, 2],
    #   'parents': {0: [], 1: [0], 2: [1]},
    #   'inputs': [0],
    #   'outputs': [2]
    # }
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
        
    # Topological Sort
    # Kahn's Algorithm
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    # Note: 'parents' usually means predecessors.
    # So in_degree for topo sort is number of parents.
    # We need to process vertices with 0 parents first.
    
    # Build children map for reverse
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        # For determinism, sort or just pop
        u = queue.pop(0) 
        topo_order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        # Cycle or invalid DAG? Fallback to naive
        return _naive_fallback(dag, fast_memory_budget)
        
    # Remaining children count
    remaining_children = {v: len(children[v]) for v in vertices}
    
    fast_set = set()
    slow_set = set(inputs)
    ops = []
    M = fast_memory_budget
    
    # Helper to evict
    def evict_one():
        if not fast_set:
            return
        # Find min remaining_children
        min_val = float('inf')
        min_v = None
        for v in fast_set:
            if remaining_children[v] < min_val:
                min_val = remaining_children[v]
                min_v = v
        if min_v is not None:
            ops.append(['delete', min_v])
            fast_set.remove(min_v)
            
    for v in topo_order:
        if time.time() > deadline:
            break # Return best so far? Or just stop.
            
        if v in inputs:
            if v not in fast_set:
                if len(fast_set) == M:
                    evict_one()
                ops.append(['load', v])
                fast_set.add(v)
        else:
            # Ensure parents in fast
            for p in parents.get(v, []):
                if p not in fast_set:
                    if len(fast_set) == M:
                        evict_one()
                    ops.append(['load', p])
                    fast_set.add(p)
                    
            # Space for v
            if len(fast_set) == M:
                evict_one()
                
            fast_set.add(v)
            ops.append(['compute', v])
            
            # Update remaining children of parents
            for p in parents.get(v, []):
                remaining_children[p] -= 1
                
            ops.append(['store', v])
            slow_set.add(v)
            
            if v in outputs:
                fast_set.remove(v)
                
    return ops

def _naive_fallback(dag, fast_memory_budget):
    # Simple topological, load all, store all
    # ...
    pass
