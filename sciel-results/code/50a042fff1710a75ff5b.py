"""Communication-minimizing DAG pebbling: improved schedule with next-use-aware eviction."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with next-use-aware eviction and batched loads.
    """
    try:
        return _pebble(dag, fast_memory_budget)
    except Exception:
        return _pebble_fallback(dag, fast_memory_budget)

def _pebble_fallback(dag, fast_memory_budget):
    """Minimal fallback: load all inputs, compute in topo order, store all."""
    num_v = dag.get('num_vertices', dag.get('n', 0))
    ops = []
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    topo = list(dag.get('topo_order', range(num_v)))
    
    in_fast = set()
    for v in topo:
        for p in parents.get(v, []):
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evictable = [x for x in in_fast if x not in outputs]
                    if evictable:
                        e = evictable[0]
                        ops.append(['store', e])
                        in_fast.discard(e)
                ops.append(['load', p])
                in_fast.add(p)
        ops.append(['compute', v])
        in_fast.add(v)
    
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evictable = [x for x in in_fast if x not in outputs]
                if evictable:
                    e = evictable[0]
                    ops.append(['store', e])
                    in_fast.discard(e)
            ops.append(['load', v])
            in_fast.add(v)
    
    return ops[:20000]

def _pebble(dag, fast_memory_budget):
    deadline = time.time() + 50
    
    # Parse DAG structure
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
                parents[v] = [int(p) for p in dag['parents'][v]]
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
    
    # Precompute last_use in topo order
    last_use = {}
    for i, v in enumerate(topo):
        last_use[v] = i
    
    # For each vertex, compute the set of vertices that will need it
    # next_use[v] = minimum topo position of any child of v
    # We'll maintain this incrementally.
    
    # Initialize next_use for each vertex as min topo position of its children
    next_use = {}
    for v in range(num_v):
        if children[v]:
            next_use[v] = min(last_use[c] for c in children[v])
        else:
            next_use[v] = float('inf')
    
    # Track remaining children count for liveness
    rem_child = {v: len(children[v]) for v in range(num_v)}
    
    in_fast = set()
    ops = []
    
    # Process in topo order with smarter eviction
    for v in topo:
        if time.time() > deadline:
            break
        if v in in_fast:
            # If already in fast memory, just update parents' rem_child
            for p in parents[v]:
                rem_child[p] -= 1
                if rem_child[p] == 0:
                    next_use[p] = float('inf')
                else:
                    # Update next_use[p] if needed
                    min_pos = float('inf')
                    for c in children[p]:
                        if rem_child[c] > 0:
                            min_pos = min(min_pos, last_use[c])
                    next_use[p] = min_pos
            continue
        
        # Collect all parents that need to be loaded
        need_load = []
        for p in parents[v]:
            if p not in in_fast:
                need_load.append(p)
        
        if need_load:
            # If we're at capacity, evict some vertices first
            if len(in_fast) >= fast_memory_budget:
                # Find vertices to evict: those with lowest liveness
                # and not needed for the current computation
                parent_set = set(parents[v])
                cands = [x for x in in_fast if x not in parent_set and x not in outputs]
                if cands:
                    # Sort by next_use descending (evict those needed farthest in future)
                    # Ties broken by rem_child ascending (evict those with fewer remaining uses)
                    cands.sort(key=lambda x: (-next_use[x], rem_child[x]))
                    num_to_evict = min(len(need_load), len(cands))
                    for i in range(num_to_evict):
                        evict = cands[i]
                        ops.append(['store', evict])
                        in_fast.discard(evict)
            
            # Now load all needed parents
            for p in need_load:
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        
        # Update liveness of parents
        for p in parents[v]:
            rem_child[p] -= 1
            if rem_child[p] == 0:
                next_use[p] = float('inf')
            else:
                # Update next_use[p]
                min_pos = float('inf')
                for c in children[p]:
                    if rem_child[c] > 0:
                        min_pos = min(min_pos, last_use[c])
                next_use[p] = min_pos
    
    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                cands = [x for x in in_fast if x not in outputs]
                if cands:
                    # Evict the one with highest next_use (least likely to be needed)
                    cands.sort(key=lambda x: (-next_use[x], rem_child[x]))
                    evict = cands[0]
                    ops.append(['store', evict])
                    in_fast.discard(evict)
            ops.append(['load', v])
            in_fast.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
