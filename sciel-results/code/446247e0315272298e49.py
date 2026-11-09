"""Communication-minimizing DAG pebbling."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Minimize communication in DAG pebbling.
    
    Returns list of [operation, vertex] with operations: load, store, compute, delete.
    """
    t_start = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    vertices = list(dag.get('vertices', []))
    parents_map = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    k = fast_memory_budget
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    # Topological index
    topo_idx = {v: i for i, v in enumerate(vertices)}
    
    # Identify inputs
    input_vertices = set(v for v in vertices if not parents_map.get(v))
    
    # For each vertex, compute last_use: the latest topological position
    # where this vertex's value is needed.
    last_use = {}
    for v in vertices:
        lu = topo_idx[v]  # at least its own position
        for c in children.get(v, []):
            lu = max(lu, topo_idx[c])
        if v in outputs:
            lu = max(lu, topo_idx[v])  # need to store it
        last_use[v] = lu
    
    # If budget >= n, trivial solution
    if k >= n:
        ops = []
        for v in vertices:
            if v in input_vertices:
                ops.append(['load', v])
            else:
                ops.append(['compute', v])
        for v in sorted(outputs, key=lambda x: topo_idx[x]):
            ops.append(['store', v])
        return ops[:20000]
    
    # Greedy pebbling with eviction based on last_use
    fast = set()          # vertices in fast memory
    in_slow = set(input_vertices)  # vertices with value in slow memory
    computed = set()      # vertices that have been computed
    
    ops = []
    
    def do_load(v):
        if v in fast:
            return
        if v not in in_slow:
            return
        if len(fast) >= k:
            _evict()
        fast.add(v)
        in_slow.discard(v)
        ops.append(['load', v])
    
    def do_store(v):
        if v not in fast:
            return
        fast.discard(v)
        in_slow.add(v)
        ops.append(['store', v])
    
    def do_compute(v):
        for p in parents_map.get(v, []):
            if p not in fast:
                return False
        if len(fast) >= k:
            _evict()
        fast.add(v)
        computed.add(v)
        ops.append(['compute', v])
        return True
    
    def do_delete(v):
        if v in fast:
            fast.discard(v)
            ops.append(['delete', v])
    
    def _evict():
        """Evict a vertex from fast to make room."""
        if not fast:
            return
        # Score each vertex: prefer evicting those with furthest last_use
        # or those we won't need again
        candidates = []
        for v in fast:
            lu = last_use.get(v, -1)
            # If we won't need it again, it's a great candidate
            if lu < topo_idx.get(v, 0) + 1:
                # Won't be needed as parent again (no children after this)
                if v in outputs:
                    score = 0  # Must store, not delete
                else:
                    score = -1  # Can delete
            else:
                score = lu  # Higher = evict later
            candidates.append((score, v))
        
        # Sort by score: delete candidates first (score -1), then by last_use
        candidates.sort()
        best = candidates[0]
        if best[0] == -1:
            do_delete(best[1])
        else:
            do_store(best[1])
    
    # Process in topological order
    for v in vertices:
        if time.time() - t_start > DEADLINE:
            break
        
        if v in input_vertices:
            # Input: don't compute, but ensure it's loadable
            continue
        
        # Ensure parents are in fast
        for p in parents_map.get(v, []):
            if p not in fast:
                if p in in_slow:
                    do_load(p)
                # If p not in in_slow, it was lost - problem
        
        # Compute v
        if do_compute(v):
            pass
    
    # Store all outputs
    for v in sorted(outputs, key=lambda x: topo_idx[x]):
        if time.time() - t_start > DEADLINE:
            break
        if v in fast:
            do_store(v)
    
    return ops[:20000]
# EVOLVE-BLOCK-END
