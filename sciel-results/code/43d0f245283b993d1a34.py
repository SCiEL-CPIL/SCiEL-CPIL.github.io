"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Greedy DAG pebbling with liveness-based eviction.
    Falls back to baseline if parsing fails.
    """
    try:
        return _pebble_greedy(dag, fast_memory_budget)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

def _pebble_greedy(dag, fast_memory_budget):
    deadline = time.time() + 55
    
    # Try to parse DAG structure robustly
    num_v = None
    if 'num_vertices' in dag:
        num_v = dag['num_vertices']
    elif 'n' in dag:
        num_v = dag['n']
    elif 'vertices' in dag and isinstance(dag['vertices'], list):
        num_v = len(dag['vertices'])
    elif 'vertices' in dag and isinstance(dag['vertices'], int):
        num_v = dag['vertices']
    
    if num_v is None:
        raise ValueError("Cannot determine number of vertices")
    
    # Build parent/child relationships
    parents = {v: [] for v in range(num_v)}
    children = {v: [] for v in range(num_v)}
    
    if 'parents' in dag:
        for v in range(num_v):
            if v in dag['parents']:
                parents[v] = list(dag['parents'][v])
                for p in parents[v]:
                    if v not in children[p]:
                        children[p].append(v)
    elif 'edges' in dag:
        for e in dag['edges']:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                u, w = int(e[0]), int(e[1])
                if 0 <= u < num_v and 0 <= w < num_v:
                    if u not in parents[w]:
                        parents[w].append(u)
                    if w not in children[u]:
                        children[u].append(w)
    
    # Identify inputs and outputs
    inputs = set()
    outputs = set()
    if 'inputs' in dag:
        inputs = set(int(x) for x in dag['inputs'])
    else:
        for v in range(num_v):
            if len(parents[v]) == 0:
                inputs.add(v)
    
    if 'outputs' in dag:
        outputs = set(int(x) for x in dag['outputs'])
    else:
        for v in range(num_v):
            if len(children[v]) == 0:
                outputs.add(v)
    
    # Topological order
    if 'topo_order' in dag:
        topo = [int(x) for x in dag['topo_order']]
    else:
        from collections import deque
        in_deg = {v: len(parents[v]) for v in range(num_v)}
        q = deque([v for v in range(num_v) if in_deg[v] == 0])
        topo = []
        while q:
            v = q.popleft()
            topo.append(v)
            for c in children[v]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        if len(topo) != num_v:
            topo = list(range(num_v))
    
    # Liveness: count of remaining children to process
    rem_child = {v: len(children[v]) for v in range(num_v)}
    
    in_fast = set()
    ops = []
    
    # Process in topological order
    for v in topo:
        if time.time() > deadline:
            break
        if v in in_fast:
            continue
        
        # Ensure parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Evict lowest liveness non-parent
                    cands = [x for x in in_fast if x != p]
                    if cands:
                        evict = min(cands, key=lambda x: (rem_child[x], x))
                        ops.append(['store', evict])
                        in_fast.discard(evict)
                ops.append(['load', p])
                in_fast.add(p)
        
        ops.append(['compute', v])
        in_fast.add(v)
        
        for p in parents[v]:
            rem_child[p] -= 1
    
    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                cands = [x for x in in_fast if x not in outputs]
                if cands:
                    evict = min(cands, key=lambda x: (rem_child[x], x))
                    ops.append(['store', evict])
                    in_fast.discard(evict)
            ops.append(['load', v])
            in_fast.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
