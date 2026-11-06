import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    budget = fast_memory_budget
    n = len(vertices)
    
    if budget <= 0 or n == 0:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    topo_order = list(vertices)
    pos_in_topo = {v: i for i, v in enumerate(topo_order)}
    
    last_use = {}
    for v in topo_order:
        max_pos = -1
        for c in children[v]:
            if pos_in_topo[c] > max_pos:
                max_pos = pos_in_topo[c]
        last_use[v] = max_pos if max_pos >= 0 else pos_in_topo[v]
    
    in_memory = set()
    operations = []
    
    for i, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            if children[v] or v in outputs:
                if v not in in_memory:
                    operations.append(['load', v])
                    in_memory.add(v)
        else:
            for p in parents.get(v, []):
                if p not in in_memory:
                    operations.append(['load', p])
                    in_memory.add(p)
            operations.append(['compute', v])
            in_memory.add(v)
        
        to_delete = [w for w in in_memory if last_use[w] < i]
        for w in to_delete:
            operations.append(['delete', w])
            in_memory.discard(w)
        
        while len(in_memory) > budget:
            best_evict = max(in_memory, key=lambda w: last_use[w])
            operations.append(['store', best_evict])
            in_memory.discard(best_evict)
    
    for v in topo_order:
        if v in outputs and v in in_memory:
            operations.append(['store', v])
            in_memory.discard(v)
    
    for v in list(in_memory):
        operations.append(['delete', v])
        in_memory.discard(v)
    
    return operations
