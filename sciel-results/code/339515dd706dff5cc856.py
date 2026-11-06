"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

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
    
    # Precompute: for each vertex, the set of vertices that depend on it (transitively)
    # This helps with better eviction decisions
    depends_on_me = {v: set() for v in range(num_v)}
    for v in topo:
        for c in children[v]:
            depends_on_me[v].add(c)
            depends_on_me[v].update(depends_on_me[c])
    
    # Count how many times each vertex will be needed in the future
    # (i.e., how many unprocessed children need it as a parent)
    future_need = {v: 0 for v in range(num_v)}
    for v in topo:
        for p in parents[v]:
            future_need[p] += 1
    
    in_fast = set()
    ops = []
    ops_count = 0
    
    def evict_best(in_fast_set, required_vertices):
        """Evict the vertex with lowest future need, preferring non-required vertices."""
        non_required = [x for x in in_fast_set if x not in required_vertices]
        if non_required:
            # Evict the one with lowest future need
            evict = min(non_required, key=lambda x: (future_need[x], -len(depends_on_me[x]), x))
        else:
            # Must evict a required vertex - pick the one with lowest future need
            evict = min(in_fast_set, key=lambda x: (future_need[x], -len(depends_on_me[x]), x))
        return evict
    
    # Process in topological order
    for v in topo:
        if time.time() > deadline:
            break
        if v in in_fast:
            # Vertex already in fast memory, just update future_need for its parents
            for p in parents[v]:
                future_need[p] -= 1
            continue
        
        # Ensure parents are in fast memory
        parent_set = set(parents[v])
        for p in parents[v]:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Evict best candidate
                    evict = evict_best(in_fast, parent_set)
                    ops.append(['store', evict])
                    ops_count += 1
                    in_fast.discard(evict)
                ops.append(['load', p])
                ops_count += 1
                in_fast.add(p)
        
        ops.append(['compute', v])
        ops_count += 1
        in_fast.add(v)
        
        # Update future_need for parents
        for p in parents[v]:
            future_need[p] -= 1
    
    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                cands = [x for x in in_fast if x not in outputs]
                if cands:
                    evict = min(cands, key=lambda x: (future_need[x], -len(depends_on_me[x]), x))
                    ops.append(['store', evict])
                    ops_count += 1
                    in_fast.discard(evict)
            ops.append(['load', v])
            ops_count += 1
            in_fast.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
