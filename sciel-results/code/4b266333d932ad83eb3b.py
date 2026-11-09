"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Minimize transfers in DAG pebbling.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    vertices = []
    parents_map = {}
    outputs = set()
    inputs = set()
    
    if isinstance(dag, dict):
        vlist = dag.get('vertices', dag.get('nodes', []))
        for item in vlist:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    v = item[0]
                    par = item[1] if len(item) > 1 else []
                    is_out = item[2] if len(item) > 2 else False
                    vertices.append(v)
                    parents_map[v] = list(par) if par else []
                    if is_out:
                        outputs.add(v)
                elif len(item) == 1:
                    vertices.append(item[0])
                    parents_map[item[0]] = []
            else:
                vertices.append(item)
                parents_map[item] = []
        if 'parents' in dag:
            for v, p in dag['parents'].items():
                parents_map[v] = list(p) if p else []
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        if 'inputs' in dag:
            inputs = set(dag['inputs'])
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    v = item[0]
                    par = item[1] if len(item) > 1 else []
                    is_out = item[2] if len(item) > 2 else False
                    vertices.append(v)
                    parents_map[v] = list(par) if par else []
                    if is_out:
                        outputs.add(v)
                elif len(item) == 1:
                    vertices.append(item[0])
                    parents_map[item[0]] = []
            else:
                vertices.append(item)
                parents_map[item] = []
    
    all_vertices_set = set(vertices)
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_map.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    if not outputs:
        outputs = set(v for v in vertices if not children_map.get(v, []))
    if not inputs:
        inputs = set(v for v in vertices if not parents_map.get(v, []))
    
    child_count = {v: len(children_map.get(v, [])) for v in vertices}
    
    # Compute a better priority: number of times this value will be needed in the future
    # For each vertex, count how many descendants (including direct children) need it
    # Simple approximation: child_count + (1 if output)
    
    # Also consider: if a vertex is an input, it's already in memory and hard to evict
    # (well, we can evict and reload, but that costs 2 transfers)
    
    # Strategy: 
    # - Keep inputs in memory as long as possible
    # - Keep intermediate results with high child_count in memory
    # - Store results with low child_count when budget is tight
    
    fast_mem = set(inputs)
    ops = []
    
    for v in vertices:
        if time.time() > DEADLINE:
            break
        
        if v in inputs:
            continue
        
        parents = parents_map.get(v, [])
        
        # Ensure parents are in fast memory
        for p in parents:
            if p not in fast_mem:
                ops.append(["load", p])
                fast_mem.add(p)
        
        # Compute v
        ops.append(["compute", v])
        fast_mem.add(v)
        
        # If over budget, evict
        if len(fast_mem) > fast_memory_budget:
            # Find best candidate to evict
            # Don't evict v (just computed)
            # Prefer to evict non-input, non-output vertices with low child_count
            # If all else fails, evict the one with lowest child_count
            
            candidates = [x for x in fast_mem if x != v]
            
            if candidates:
                # Score: inputs get high score (hard to evict), outputs get medium, others by child_count
                def evict_score(x):
                    score = 0
                    if x in inputs:
                        score += 1000  # very hard to evict
                    if x in outputs:
                        score += 500  # hard to evict
                    score += child_count.get(x, 0) * 10
                    return score
                
                # Evict the one with lowest score
                min_score = min(evict_score(c) for c in candidates)
                evict_cands = [c for c in candidates if evict_score(c) == min_score]
                evict = evict_cands[0]
                ops.append(["store", evict])
                fast_mem.remove(evict)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            ops.append(["load", v])
            fast_mem.add(v)
            if len(fast_mem) > fast_memory_budget:
                candidates = [x for x in fast_mem if x != v and x not in outputs]
                if candidates:
                    def evict_score(x):
                        score = 0
                        if x in inputs:
                            score += 1000
                        score += child_count.get(x, 0) * 10
                        return score
                    min_score = min(evict_score(c) for c in candidates)
                    evict_cands = [c for c in candidates if evict_score(c) == min_score]
                    evict = evict_cands[0]
                    ops.append(["store", evict])
                    fast_mem.remove(evict)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END

