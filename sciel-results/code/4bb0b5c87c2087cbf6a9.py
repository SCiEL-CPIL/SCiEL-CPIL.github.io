"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85
    
    vertices = list(dag.get('vertices', []))
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Topological order
    in_deg = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = list(v for v in vertices if in_deg[v] == 0)
    in_deg_work = dict(in_deg)
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_deg_work[c] -= 1
            if in_deg_work[c] == 0:
                queue.append(c)
    
    # For each vertex, find its position in topo order and last use
    pos = {v: i for i, v in enumerate(topo_order)}
    last_use = {}
    for v in reversed(topo_order):
        if not children[v]:
            last_use[v] = pos[v]
        else:
            last_use[v] = max(last_use[c] for c in children[v])
    
    # Remaining uses: how many times v is needed after current point
    # We'll track this dynamically
    
    fast_mem = {}
    slow_available = set(inputs)
    computed = set(inputs)
    ops = []
    
    # Count remaining children for each vertex (to know when a parent is last used)
    remaining_children = {v: len(children[v]) for v in vertices}
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in slow_available:
                    if len(fast_mem) >= fast_memory_budget:
                        # Evict: pick vertex with smallest last_use that isn't an output and isn't a parent of v
                        candidates = [x for x in fast_mem if x not in outputs and x not in parents.get(v, [])]
                        if not candidates:
                            candidates = [x for x in fast_mem if x not in outputs]
                        if candidates:
                            candidates.sort(key=lambda x: last_use.get(x, 0))
                            evict = candidates[0]
                            ops.append(['delete', evict])
                            del fast_mem[evict]
                            ops.append(['store', evict])
                            slow_available.add(evict)
                    ops.append(['load', p])
                    fast_mem[p] = True
        
        # Compute v
        ops.append(['compute', v])
        fast_mem[v] = True
        computed.add(v)
        
        # Decrement remaining children for parents
        for p in parents.get(v, []):
            remaining_children[p] -= 1
            # If p is no longer needed by any remaining child, and p is not an output
            if remaining_children[p] == 0 and p not in outputs:
                if p in fast_mem:
                    ops.append(['delete', p])
                    del fast_mem[p]
                    ops.append(['store', p])
                    slow_available.add(p)
        
        # Enforce memory budget
        while len(fast_mem) > fast_memory_budget:
            candidates = [x for x in fast_mem if x not in outputs]
            if not candidates:
                candidates = list(fast_mem.keys())
            candidates.sort(key=lambda x: last_use.get(x, 0))
            evict = candidates[0]
            ops.append(['delete', evict])
            del fast_mem[evict]
            ops.append(['store', evict])
            slow_available.add(evict)
    
    # Final: ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            if v in slow_available:
                if len(fast_mem) >= fast_memory_budget:
                    candidates = [x for x in fast_mem if x not in outputs]
                    if candidates:
                        candidates.sort(key=lambda x: last_use.get(x, 0))
                        evict = candidates[0]
                        ops.append(['delete', evict])
                        del fast_mem[evict]
                        ops.append(['store', evict])
                        slow_available.add(evict)
                ops.append(['load', v])
                fast_mem[v] = True
    
    return ops

# EVOLVE-BLOCK-END
