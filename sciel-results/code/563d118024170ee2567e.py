import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85
    
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        topo_order = vertices
    
    state = {v: 'blue' for v in vertices}
    computed = set()
    operations = []
    fast_memory_used = 0
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        for p in parents.get(v, []):
            if state[p] == 'blue':
                operations.append(['load', p])
                state[p] = 'red'
                fast_memory_used += 1
        
        operations.append(['compute', v])
        state[v] = 'red'
        fast_memory_used += 1
        computed.add(v)
        
        for p in parents.get(v, []):
            still_needed = any(c not in computed for c in children[p])
            if not still_needed and p not in outputs:
                if state[p] == 'red':
                    operations.append(['store', p])
                    state[p] = 'blue'
                    fast_memory_used -= 1
    
    for o in outputs:
        if state[o] == 'red':
            operations.append(['store', o])
            state[o] = 'blue'
            fast_memory_used -= 1
    
    return operations
