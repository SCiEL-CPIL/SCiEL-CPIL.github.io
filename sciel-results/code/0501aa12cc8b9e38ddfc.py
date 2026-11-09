"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85
    
    # Parse DAG
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        if not vertices:
            parents_dict = dag.get('parents', {})
            all_v = set()
            for v, ps in parents_dict.items():
                all_v.add(v)
                for p in ps:
                    all_v.add(p)
            vertices = sorted(all_v)
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        parents = {}
        inputs = set()
        outputs = set()
        vertices = []
        for item in dag:
            if isinstance(item, tuple) and len(item) >= 2:
                v = item[0]
                ps = item[1]
                parents[v] = list(ps)
                vertices.append(v)
        if vertices:
            n = len(vertices)
            inputs = set(vertices[:max(1, n//4)])
            outputs = set(vertices[max(0, n - max(1, n//4)):])
    
    if not vertices:
        return []
    
    if not isinstance(parents, dict):
        parents = {}
    
    input_set = set(inputs)
    output_set = set(outputs)
    topo_order = list(vertices)
    
    # Compute last use index for each vertex
    last_use = {}
    for i, v in enumerate(topo_order):
        last_use[v] = i
    for i, v in enumerate(topo_order):
        for p in parents.get(v, []):
            if i > last_use.get(p, -1):
                last_use[p] = i
    
    ops = []
    fast_mem = set()
    stored = set()
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        ps = parents.get(v, [])
        
        for p in ps:
            if p not in fast_mem:
                ops.append(['load', p])
                fast_mem.add(p)
        
        ops.append(['compute', v])
        fast_mem.add(v)
        
        while len(fast_mem) > fast_memory_budget:
            cur_idx = topo_order.index(v)
            # Evict vertex with largest last_use (least urgent)
            # Prefer evicting those no longer needed
            candidates = list(fast_mem)
            no_longer_needed = [x for x in candidates if last_use.get(x, -1) < cur_idx and x not in output_set]
            if no_longer_needed:
                evict = no_longer_needed[0]
            else:
                evict = max(candidates, key=lambda x: last_use.get(x, -1))
            fast_mem.discard(evict)
            if evict not in stored:
                ops.append(['delete', evict])
        
        if v in output_set:
            ops.append(['store', v])
            stored.add(v)
            fast_mem.discard(v)
    
    for v in output_set:
        if v not in stored and v in fast_mem:
            ops.append(['store', v])
            stored.add(v)
            fast_mem.discard(v)
    
    return ops
# EVOLVE-BLOCK-END
