"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    topo_order = vertices
    vertex_index = {v: i for i, v in enumerate(topo_order)}
    
    # When is v's value last needed?
    last_needed = {}
    for v in topo_order:
        idx = vertex_index[v]
        if children.get(v):
            lc = max(vertex_index[c] for c in children[v])
        else:
            lc = idx
        last_needed[v] = len(topo_order) if v in outputs else lc
    
    ops = []
    fm = set(inputs)
    
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        if v in inputs:
            continue
        
        # Load missing parents
        for p in parents.get(v, []):
            if p not in fm:
                ops.append(['load', p])
                fm.add(p)
        
        ops.append(['compute', v])
        fm.add(v)
        
        # Free memory
        while len(fm) > fast_memory_budget:
            cand = [x for x in fm if last_needed[x] < i and x not in outputs]
            if not cand:
                break
            d = min(cand, key=lambda x: last_needed[x])
            ops.append(['delete', d])
            fm.discard(d)
    
    # Store outputs
    for v in topo_order:
        if v in outputs:
            if v not in fm:
                ops.append(['load', v])
                fm.add(v)
            ops.append(['store', v])
    
    return ops[:20000]
# EVOLVE-BLOCK-END
