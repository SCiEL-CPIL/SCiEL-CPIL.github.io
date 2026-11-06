import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    in_fast = set(inputs)
    in_slow = set()
    computed = set()
    
    remaining_uses = {}
    for v in vertices:
        count = len(children[v])
        if v in outputs:
            count += 1
        remaining_uses[v] = count
    
    operations = []
    
    def time_check():
        return time.time() - start_time > DEADLINE
    
    def can_compute(v):
        for p in parents.get(v, []):
            if p not in in_fast:
                return False
        return True
    
    def make_room():
        if len(in_fast) < fast_memory_budget:
            return True
        
        # Find vertex to evict: not an output, prefer low remaining uses
        candidates = []
        for v in in_fast:
            if v in outputs:
                continue
            candidates.append((remaining_uses[v], v))
        
        if not candidates:
            return False
        
        candidates.sort()
        _, v_to_evict = candidates[0]
        in_fast.remove(v_to_evict)
        in_slow.add(v_to_evict)
        operations.append(['store', v_to_evict])
        return True
    
    for v in topo_order:
        if time_check():
            break
        
        if v in inputs:
            computed.add(v)
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                if p in in_slow:
                    if not make_room():
                        break
                    in_slow.remove(p)
                    in_fast.add(p)
                    operations.append(['load', p])
        
        if not can_compute(v):
            continue
        
        # Make room for result
        if len(in_fast) >= fast_memory_budget:
            if not make_room():
                continue
        
        computed.add(v)
        in_fast.add(v)
        operations.append(['compute', v])
        
        for p in parents.get(v, []):
            remaining_uses[p] -= 1
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                if not make_room():
                    break
                in_slow.remove(v)
                in_fast.add(v)
                operations.append(['load', v])
    
    return operations
