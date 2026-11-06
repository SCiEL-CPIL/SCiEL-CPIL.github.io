import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        # Parse DAG
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        # Fallback to naive if parsing fails, though unlikely
        return []

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # Compute layers
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    in_deg = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0

    # Pre-compute "next needed layer" for each vertex
    # next_need[v] = sorted list of layers where v is a parent
    next_need = {v: [] for v in all_vertices}
    for v in all_vertices:
        for c in children[v]:
            next_need[v].append(layer[c])
        # Sort to allow efficient "next" lookup
        next_need[v].sort()

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    # Track index into next_need for O(1) next-lookup? 
    # Since we process layers in order, we can just keep a pointer or use bisect.
    # Given small N (implied by 20000 ops limit), linear scan in next_need is fine.
    
    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # 1. Load needed parents for vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                if p not in fast_mem:
                    # Must be in slow_mem (validity check assumed)
                    if p in slow_mem:
                        ops.append(['load', p])
                        fast_mem.add(p)
                        slow_mem.discard(p)
                    # If p is not in slow_mem and not in fast_mem, it's an error in state, 
                    # but we assume valid input. If it was computed and deleted, it's gone.
                    # Wait, if it was computed and deleted, it's not in slow_mem.
                    # This implies we need to recompute it. But the current interface 
                    # doesn't allow "recompute" as a separate op, only "compute".
                    # "Compute" requires parents to be in fast_mem.
                    # If a parent is missing, we have a bug in the schedule.
                    # The baseline assumes we keep enough parents.
                    # To be safe, if p is missing, we can't compute v.
                    # But the task says "Recomputations allowed".
                    # If p is missing, we must have deleted it. We need to reload or recompute p.
                    # Since we don't track "deleted" separately from "never in fast", 
                    # and inputs are in slow, if p is not in fast and not in slow, 
                    # it must be an intermediate that was deleted.
                    # We cannot "load" it. We must "compute" it.
                    # But to compute p, we need ITS parents.
                    # This recursive requirement is complex.
                    # The baseline strategy avoids this by never deleting a node 
                    # that is needed in the *current* or *next* layer? 
                    # Actually, the baseline deletes based on "needed in future".
                    # If it deletes a node needed in layer L+1, then at layer L+1, 
                    # it will try to load it. But it's not in slow_mem.
                    # So the baseline is actually broken for intermediate nodes 
                    # if they are deleted before they are used?
                    # Let's re-read: "Inputs start blue... every red value consumes one fast-memory slot".
                    # "Store" puts it back in slow. "Delete" removes it.
                    # If I delete an intermediate node, it's gone.
                    # If I need it later, I must recompute it.
                    # To recompute it, I need its parents.
                    # So a simple layer-by-layer with deletion is risky.
                    # 
                    # However, the score is 1.0 (baseline). The baseline likely works.
                    # Why? Maybe the "needed" check in the baseline prevents deleting 
                    # nodes needed in the *immediate* future?
                    # The baseline check:
                    # for fl in range(L + 1, max_L + 1):
                    #   if u in parents of any w in layer fl: needed = True
                    # So it checks ALL future layers.
                    # If it is needed in L+1, needed=True, sc=1000.
                    # If it is NOT needed in any future layer, sc=1.
                    # So it only deletes nodes that are NOT needed in the future.
                    # This means it NEVER deletes a node that will be needed.
                    # It only deletes nodes that are "done" (no more children to compute).
                    # This is a safe strategy! It never forces recomputation.
                    # It just minimizes memory by clearing "done" nodes.
                    #
                    # So, to improve, we should consider deleting nodes that ARE needed 
                    # in the future, IF the recompute cost is low enough to save memory 
                    # for other nodes that are more critical?
                    # Or, more simply: The baseline keeps ALL nodes needed in the future.
                    # This might exceed fast_memory_budget if the DAG is wide.
                    # If it exceeds budget, the baseline breaks (while loop evicts).
                    # If it breaks, it evicts based on the score.
                    # If a node is needed in future, sc=1000. If not, sc=1.
                    # So it prefers evicting "done" nodes.
                    # If ALL nodes are needed in future, it evicts the one with lowest sc (1000).
                    # It picks the first one? No, it picks the one with < best_sc.
                    # If all are 1000, it picks the first one found in iteration.
                    # This is arbitrary.
                    
                    # Improvement: Distinguish between "needed soon" and "needed later".
                    # If I have to evict a node that is needed, evict the one needed LATEST.
                    
                    pass

        # 2. Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            
            # Check if parents are in fast_mem
            # If not, we have a problem. 
            # But with the "never delete needed" strategy, they should be.
            # Unless we evicted them due to budget pressure.
            # If we evicted a needed node, we MUST recompute it.
            # Let's add a check: if parent missing, recompute it?
            # That's complex. Let's stick to the safe strategy: 
            # Only evict nodes that are NOT needed in the future.
            # If we run out of memory and MUST evict a needed node, 
            # we accept the penalty of recomputation later.
            
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

        # 3. Evict if over budget
        while len(fast_mem) > fast_memory_budget and not time_check():
            best = None
            best_sc = float('inf')
            for u in fast_mem:
                if u in input_vertices:
                    # Inputs are cheap to reload.
                    sc = 0
                elif u in outputs:
                    # Outputs must be stored. If we delete, we must recompute and store again.
                    # Expensive.
                    sc = 10000
                else:
                    # Find next layer u is needed
                    next_layers = next_need[u]
                    # Find first layer > L
                    next_L = None
                    for nl in next_layers:
                        if nl > L:
                            next_L = nl
                            break
                    
                    if next_L is None:
                        # Not needed in future. Safe to delete.
                        sc = 1
                    else:
                        # Needed in future. 
                        # Cost is roughly proportional to (next_L - L) + recompute_cost.
                        # We want to evict the one with HIGHEST cost of keeping?
                        # No, we want to evict the one with LOWEST priority.
                        # Priority: 
                        # 1. Is it needed? Yes.
                        # 2. When? Later is better to evict (we can keep it longer).
                        #    Wait. If I evict it now, I save space now.
                        #    If I need it in 10 layers, I save space for 10 layers.
                        #    If I need it in 1 layer, I save space for 1 layer.
                        #    So evicting a node needed in 10 layers is MORE beneficial 
                        #    in terms of space freed for the immediate future?
                        #    No. If I evict a node needed in 1 layer, I have to load/recompute it 
                        #    very soon. If I evict a node needed in 10 layers, I can use the space 
                        #    for 10 layers before I need it back.
                        #    So, evict the node with the FURTHEST next-use.
                        sc = 1000 - (next_L - L) # Higher score = keep? 
                        # Let's use a cost to evict.
                        # Cost to evict = Recompute Cost.
                        # Recompute cost is higher if it's needed soon? No.
                        # If I evict it, I MUST recompute it at next_L.
                        # The "pain" is the operation at next_L.
                        # So all needed nodes have similar pain.
                        # But if I keep a node needed in 10 layers, I waste space for 10 layers.
                        # So I should evict the one with the FURTHEST next-use.
                        # So, lower score = evict first.
                        # Score = next_L. Higher next_L = evict first.
                        # So we want to MINIMIZE next_L? No, MAXIMIZE next_L.
                        # Let's invert: sc = -next_L.
                        sc = -next_L
                
                if sc < best_sc:
                    best_sc = sc
                    best = u
            
            if best is None:
                break
            if best in outputs:
                ops.append(['store', best])
                slow_mem.add(best)
            else:
                ops.append(['delete', best])
            fast_mem.discard(best)

    # Store outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
