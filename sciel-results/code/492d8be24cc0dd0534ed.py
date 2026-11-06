"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize memory transfers.
    """
    deadline = time.time() + 85
    
    # Parse DAG structure - try common formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', None)
        if inputs is None:
            inputs = [v for v in vertices if not parents.get(v, [])]
    else:
        # Fallback: assume dag is a list of vertices
        vertices = list(dag)
        parents = {}
        outputs = []
        inputs = list(vertices)
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order (assume vertices are already in topo order)
    topo_order = list(vertices)
    
    # Compute last_used for each vertex
    vertex_to_idx = {v: i for i, v in enumerate(topo_order)}
    last_used = {}
    for v in topo_order:
        last_used[v] = vertex_to_idx.get(v, 0)
        for c in children.get(v, []):
            if c in vertex_to_idx:
                last_used[v] = max(last_used[v], vertex_to_idx[c])
    
    # Fast memory
    fast_mem = set()
    slow_mem = set(inputs)
    
    operations = []
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        if v in inputs:
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Evict least useful
                    candidates = list(fast_mem)
                    evict_v = min(candidates, key=lambda x: last_used.get(x, 0))
                    fast_mem.discard(evict_v)
                    operations.append(['delete', evict_v])
                operations.append(['load', v])
                fast_mem.add(v)
            continue
        
        v_parents = parents.get(v, [])
        
        # Load parents not in fast memory
        for p in v_parents:
            if p not in fast_mem:
                if p in slow_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        candidates = [x for x in fast_mem if x not in v_parents]
                        if not candidates:
                            candidates = list(fast_mem)
                        evict_v = min(candidates, key=lambda x: last_used.get(x, 0))
                        fast_mem.discard(evict_v)
                        operations.append(['delete', evict_v])
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute
        if len(fast_mem) >= fast_memory_budget:
            candidates = [x for x in fast_mem if x not in v_parents]
            if not candidates:
                candidates = list(fast_mem)
            evict_v = min(candidates, key=lambda x: last_used.get(x, 0))
            fast_mem.discard(evict_v)
            operations.append(['delete', evict_v])
        
        operations.append(['compute', v])
        fast_mem.add(v)
        slow_mem.add(v)
    
    # Ensure outputs in fast memory
    for out in outputs:
        if out not in fast_mem and out in slow_mem:
            if len(fast_mem) >= fast_memory_budget:
                candidates = [x for x in fast_mem if x not in outputs]
                if not candidates:
                    candidates = list(fast_mem)
                evict_v = min(candidates, key=lambda x: last_used.get(x, 0))
                fast_mem.discard(evict_v)
                operations.append(['delete', evict_v])
            operations.append(['load', out])
            fast_mem.add(out)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END

