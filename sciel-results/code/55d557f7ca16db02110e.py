"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _topo_sort(parents, vertices):
    """Return topological order of vertices."""
    vset = set(vertices)
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)  # cycle fallback
    return topo, children

def _pebble_postorder(dag, budget):
    """
    Postorder pebbling: recursively pebble subtrees, compute, delete intermediates.
    Each vertex is computed at most once. Intermediates are deleted when no longer needed.
    This minimizes transfers vs naive load-everything.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    # Build children map
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    
    # Topological order
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)
    
    # Postorder: process leaves first, then their parents
    # We need to process in reverse topological order (sinks first)
    # Actually, we want to compute in topological order but use recursion for subtrees
    
    ops = []
    loaded = set()   # vertices currently in fast memory
    computed = set() # vertices that have been computed (and may or may not be loaded)
    
    # Determine which vertices are "needed" (ancestors of outputs)
    needed = set()
    def mark_needed(v):
        if v in needed:
            return
        needed.add(v)
        for p in parents.get(v, []):
            if p in vset:
                mark_needed(p)
    for o in outputs:
        mark_needed(o)
    
    # Only process needed vertices
    # Use recursive pebbling in topological order
    # For each vertex in topo order, if it's needed:
    #   - ensure parents are loaded (recursively pebble them)
    #   - compute v
    #   - if v is output: store and unload
    #   - else: keep loaded if needed by later vertices, else delete
    
    # Track which vertices still need each loaded vertex
    # Use a counter: for each vertex, count how many children haven't been processed yet
    remaining_children = {}
    for v in vertices:
        if v in needed:
            remaining_children[v] = len([c for c in children.get(v, []) if c in needed])
    
    for v in topo:
        if v not in needed:
            continue
        # Ensure all parents are loaded
        for p in parents.get(v, []):
            if p not in vset:
                continue
            if p not in loaded:
                # Need to load p: recursively ensure p's parents are loaded, compute p
                # But p should already be computed if it's in topo before v
                # If p is not loaded, it was deleted - need to recompute
                # For simplicity in postorder, we ensure p is loaded before computing v
                if p not in computed:
                    # Recompute p's subtree
                    for pp in parents.get(p, []):
                        if pp in vset and pp not in loaded:
                            if pp not in computed:
                                # This shouldn't happen in proper topo order
                                pass
                            else:
                                ops.append(["load", pp])
                                loaded.add(pp)
                        elif pp in vset and pp not in loaded:
                            ops.append(["load", pp])
                            loaded.add(pp)
                    ops.append(["compute", p])
                    computed.add(p)
                    loaded.add(p)
                else:
                    # p was computed but deleted, reload
                    ops.append(["load", p])
                    loaded.add(p)
        # Compute v
        ops.append(["compute", v])
        computed.add(v)
        loaded.add(v)
        
        # If v is an output, store it
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        
        # Delete parents that are no longer needed
        for p in parents.get(v, []):
            if p not in vset:
                continue
            if p not in loaded:
                continue
            # Decrement remaining children count
            remaining_children[p] = remaining_children.get(p, 0) - 1
            if remaining_children[p] <= 0 and p not in outset:
                ops.append(["delete", p])
                loaded.discard(p)
    
    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    """A* over loaded-set states. Only for small n."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 12:
        return None  # Too large for A*
    
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            if p in idx:
                m |= (1 << idx[p])
        par_mask[i] = m
    
    def popcount(x):
        return bin(x).count("1")
    
    def heuristic(loaded):
        return popcount(out_mask & ~loaded)
    
    start = 0
    goal = out_mask
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    
    while open_heap:
        if time.time() > deadline:
            return None
        f, g, _, state = heapq.heappop(open_heap)
        if state == goal:
            seq = []
            s = goal
            while s != start:
                prev, op, v = parent_op[s]
                seq.append([op, v])
                s = prev
            seq.reverse()
            return seq
        if g > g_score.get(state, float("inf")):
            continue
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            # load
            if not (state & bit):
                ns = state | bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "load", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # store
            if (state & bit) and (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # delete
            if (state & bit) and not (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # compute
            if (state & par_mask[i]) == par_mask[i]:
                ns = state | bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return None

def _pebble_greedy(dag, budget):
    """Original greedy fallback as last resort."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    ops = []
    outset = set(outputs)
    
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)
    
    for v in topo:
        for p in parents.get(v, []):
            ops.append(["load", p])
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
        for p in parents.get(v, []):
            still = any(p in parents.get(w, []) for w in topo[topo.index(v)+1:] if w in vset)
            if not still and p not in outset:
                ops.append(["delete", p])
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50  # Safety margin before 60s limit
    try:
        # Try A* for small DAGs
        result = _pebble_astar(dag, fast_memory_budget, deadline)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Postorder pebbling (efficient for larger DAGs)
    try:
        result = _pebble_postorder(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    # Last resort: greedy
    try:
        result = _pebble_greedy(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    
    return []
# EVOLVE-BLOCK-END
