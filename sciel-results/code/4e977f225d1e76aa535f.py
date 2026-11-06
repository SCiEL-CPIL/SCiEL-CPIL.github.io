# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    topo_order = dag['topo_order']
    parents_map = dag.get('parents', {})
    outputs = set(dag['outputs'])
    n = len(topo_order)
    idx = {v: i for i, v in enumerate(topo_order)}
    budget = max(1, int(fast_memory_budget))
    
    # Build children map
    children_map = {v: [] for v in topo_order}
    for v in topo_order:
        for p in parents_map.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    # Determine if each vertex is an input
    is_input = {}
    for v in topo_order:
        ps = parents_map.get(v, [])
        is_input[v] = (len(ps) == 0)
    
    # For budget-aware pebbling, we use the following strategy:
    # Process in topological order. Maintain a set of "live" vertices in fast memory.
    # When memory exceeds budget, evict the vertex whose next use is furthest in the future.
    
    ops = []
    live = set()  # vertices in fast memory
    
    def do_load(v):
        ops.append(['load', v])
        live.add(v)
    
    def do_compute(v):
        ops.append(['compute', v])
        live.add(v)
    
    def do_delete(v):
        if v in live:
            ops.append(['delete', v])
            live.discard(v)
    
    def do_store(v):
        ops.append(['store', v])
    
    # Precompute next_use: for each vertex, the earliest topo index at which it's needed again
    # Actually, we need: for vertex v at position i, what's the next position > i where v is a parent
    next_use = {}
    for i, v in enumerate(topo_order):
        kids = children_map[v]
        future_uses = [idx[k] for k in kids if idx[k] > i]
        if v in outputs:
            future_uses.append(n)  # outputs needed at end
        next_use[v] = min(future_uses) if future_uses else i
    
    for i, v in enumerate(topo_order):
        ps = parents_map.get(v, [])
        
        if is_input[v]:
            if v not in live:
                do_load(v)
        else:
            # Ensure parents are live
            for p in ps:
                if p not in live:
                    # Need to restore p. If p is input, load it.
                    # If p is non-input, this is problematic without recomputation.
                    # For a correct solution, we should never have deleted p before v.
                    # If this happens, we recompute p (allowed by problem statement).
                    if is_input[p]:
                        do_load(p)
                    else:
                        # Recompute p
                        # First ensure p's parents are live
                        p_ps = parents_map.get(p, [])
                        for pp in p_ps:
                            if pp not in live:
                                if is_input[pp]:
                                    do_load(pp)
                        do_compute(p)
            
            do_compute(v)
        
        # Now evict if over budget (keep v)
        while len(live) > budget:
            # Evict the live vertex (other than v) with the largest next_use (furthest future need)
            candidates = [u for u in live if u != v]
            if not candidates:
                break
            # Pick the one with largest next_use
            victim = max(candidates, key=lambda u: next_use.get(u, i))
            do_delete(victim)
        
        # Delete vertices that are no longer needed (next_use < i and not v)
        for u in list(live):
            if u != v and next_use.get(u, i) <= i:
                do_delete(u)
        
        # If still over budget after deletions, evict more
        while len(live) > budget:
            candidates = [u for u in live if u != v]
            if not candidates:
                break
            victim = max(candidates, key=lambda u: next_use.get(u, i))
            do_delete(victim)
    
    # Store outputs
    for v in topo_order:
        if v in outputs:
            if v in live:
                do_store(v)
                do_delete(v)
            else:
                # Recompute and store
                if not is_input[v]:
                    ps = parents_map.get(v, [])
                    for p in ps:
                        if p not in live:
                            if is_input[p]:
                                do_load(p)
                            else:
                                p_ps = parents_map.get(p, [])
                                for pp in p_ps:
                                    if pp not in live:
                                        if is_input[pp]:
                                            do_load(pp)
                                do_compute(p)
                    do_compute(v)
                else:
                    do_load(v)
                do_store(v)
                do_delete(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
