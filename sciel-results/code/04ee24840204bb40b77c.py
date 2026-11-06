"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    nodes = dag.get('nodes', [])
    edges = dag.get('edges', [])
    inputs = dag.get('inputs', [])
    outputs = dag.get('outputs', [])
    
    children = defaultdict(list)
    parents = defaultdict(list)
    for edge in edges:
        p, c = edge
        children[p].append(c)
        parents[c].append(p)
    
    in_degree = {n: len(parents[n]) for n in nodes}
    q = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while q:
        n = q.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                q.append(c)
    if len(topo_order) != len(nodes):
        for n in nodes:
            if n not in topo_order:
                topo_order.append(n)
    
    topo_pos = {n: i for i, n in enumerate(topo_order)}
    next_use_pos = {}
    for n in nodes:
        cp = [topo_pos[c] for c in children[n] if topo_pos.get(c, 0) > topo_pos.get(n, 0)]
        next_use_pos[n] = min(cp) if cp else float('inf')
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    fast_mem = set(inputs)
    slow_mem = set()
    computed = set(inputs)
    operations = []
    
    for n in topo_order:
        if time.time() > deadline:
            break
        if n in computed:
            continue
        
        for p in parents[n]:
            if p not in fast_mem:
                operations.append(['load', p])
                if p in slow_mem:
                    slow_mem.discard(p)
                fast_mem.add(p)
        
        operations.append(['compute', n])
        fast_mem.add(n)
        computed.add(n)
        
        if len(fast_mem) > fast_memory_budget:
            cands = [(next_use_pos.get(v, float('inf')), v) 
                     for v in fast_mem if v in computed and v not in output_set]
            cands.sort(reverse=True)
            while len(fast_mem) > fast_memory_budget and cands:
                _, v = cands.pop()
                if v in fast_mem:
                    operations.append(['store', v])
                    fast_mem.discard(v)
                    slow_mem.add(v)
        
        if fast_memory_budget > 0 and len(fast_mem) >= fast_memory_budget * 0.9:
            cands = [(next_use_pos.get(v, float('inf')), v) 
                     for v in fast_mem if v in computed and v not in output_set]
            cands.sort(reverse=True)
            for _, v in cands:
                if len(fast_mem) <= fast_memory_budget:
                    break
                if v in fast_mem:
                    operations.append(['store', v])
                    fast_mem.discard(v)
                    slow_mem.add(v)
    
    for o in outputs:
        if o not in fast_mem:
            operations.append(['load', o])
            if o in slow_mem:
                slow_mem.discard(o)
            fast_mem.add(o)
    
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['delete', v])
            fast_mem.discard(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
