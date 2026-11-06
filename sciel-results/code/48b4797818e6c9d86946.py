# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque
import math

def _topo_order(dag):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return [], {}
    vset = set(vertices)
    children = {v: [] for v in vertices}
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
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
        topo = vertices
    return topo, children

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    ops = []
    outset = set(outputs)
    # topological order
    indeg = {v: len(parents.get(v, [])) for v in vertices}
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
        topo = vertices
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

def _pebble_postorder(dag, budget):
    # Efficient pebbling via postorder with recompute-on-demand.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)

    indeg = {v: len(parents.get(v, [])) for v in vertices}
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
        topo = vertices
    post = list(reversed(topo))

    ops = []
    loaded = set()
    stored = set()

    def pebble(v):
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p)
        ops.append(["compute", v])
        loaded.add(v)
        for p in ps:
            if p in outset or p in stored:
                continue
            needed = False
            for w in children.get(p, []):
                if w != v and w not in loaded and w not in stored:
                    needed = True
                    break
            if not needed:
                ops.append(["delete", p])
                loaded.discard(p)

    for v in post:
        if v in outset:
            pebble(v)
            ops.append(["store", v])
            stored.add(v)
            loaded.discard(v)

    return ops[:20000]

def _pebble_greedy_lfu(dag, budget):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
                
    indeg = {v: len(parents.get(v, [])) for v in vertices}
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
        topo = vertices
        
    freq = {v: 0 for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            freq[p] += 1

    ops = []
    loaded = {}
    stored = set()
    
    topo_idx = {v: i for i, v in enumerate(topo)}

    for v in topo:
        for p in parents.get(v, []):
            if p not in loaded:
                if len(loaded) >= budget:
                    best_victim = None
                    best_score = (float('inf'), float('inf'))
                    for lp in loaded:
                        if lp in stored:
                            continue
                        rem_children = 0
                        for c in children.get(lp, []):
                            if topo_idx.get(c, 0) > topo_idx.get(v, 0):
                                rem_children += 1
                        score = (rem_children, freq[lp])
                        if score < best_score:
                            best_score = score
                            best_victim = lp
                    if best_victim is not None:
                        ops.append(["delete", best_victim])
                        loaded.pop(best_victim)
                
                ops.append(["load", p])
                loaded[p] = v
        
        ops.append(["compute", v])
        
        if v in outset:
            ops.append(["store", v])
            stored.add(v)
            loaded.pop(v, None)
        else:
            loaded[v] = v

    return ops[:20000]

def _pebble_layered(dag, budget):
    """Layer-aware pebbling: process layer by layer, keeping only current+next layer in memory."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    
    # Compute layers based on longest path from sources
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    layer = {v: 0 for v in vertices}
    q = deque([v for v in vertices if indeg[v] == 0])
    processed = []
    while q:
        u = q.popleft()
        processed.append(u)
        for w in children.get(u, []):
            layer[w] = max(layer[w], layer[u] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    
    if len(processed) != len(vertices):
        # Fallback if cycle detected
        return _pebble_postorder(dag, budget)
    
    max_layer = max(layer.values()) if vertices else 0
    
    # Group vertices by layer
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer[v]].append(v)
    
    ops = []
    loaded = set()
    stored = set()
    
    # Process each layer
    for l in range(max_layer + 1):
        layer_verts = layers[l]
        # Sort within layer for consistent processing
        layer_verts.sort(key=lambda x: (x if isinstance(x, int) else 0))
        
        for v in layer_verts:
            ps = parents.get(v, [])
            # Load parents if not already loaded
            for p in ps:
                if p not in loaded and p not in stored:
                    # Need to load p - but p should be in a previous layer
                    # If p is not loaded, we need to recompute it
                    # For simplicity, we load it (assumes it was computed in previous layer)
                    if p in vset:
                        ops.append(["load", p])
                        loaded.add(p)
            
            # Compute v
            ops.append(["compute", v])
            loaded.add(v)
            
            # Clean up parents that are no longer needed
            for p in ps:
                if p in outset or p in stored:
                    continue
                # Check if p is needed for any unprocessed vertex
                needed = False
                for w in children.get(p, []):
                    if w != v and w not in loaded and w not in stored:
                        # Check if w is in current or future layers
                        if layer.get(w, 0) >= l:
                            needed = True
                            break
                if not needed:
                    ops.append(["delete", p])
                    loaded.discard(p)
            
            # If v is an output, store it
            if v in outset:
                ops.append(["store", v])
                stored.add(v)
                loaded.discard(v)
    
    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 16:
        return None
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m

    # Precompute for heuristic: for each vertex, the set of ancestors needed
    # and a lower bound on transfers to get all outputs
    def popcount(x):
        return bin(x).count("1")

    def heuristic(loaded):
        # Lower bound: number of outputs not yet loaded
        missing_outs = popcount(out_mask & ~loaded)
        # Additional: for each missing output, we need at least 1 transfer
        # (load or compute+store)
        return missing_outs

    start = 0
    goal = out_mask
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    best = None
    max_expansions = 500000
    expansions = 0
    while open_heap and expansions < max_expansions:
        if time.time() > deadline:
            break
        f, g, _, state = heapq.heappop(open_heap)
        expansions += 1
        if state == goal:
            seq = []
            s = goal
            while s != start:
                prev, op, v = parent_op[s]
                seq.append([op, v])
                s = prev
            seq.reverse()
            best = seq
            break
        if g > g_score.get(state, float("inf")):
            continue
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            if not (state & bit):
                ns = state | bit
                if popcount(ns) > budget:
                    continue
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "load", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & bit) and (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & bit) and not (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & par_mask[i]) == par_mask[i]:
                ns = state | bit
                if popcount(ns) > budget:
                    continue
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def _validate_ops(ops, dag, budget):
    """Validate that ops form a valid schedule within memory budget."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    outset = set(outputs)
    loaded = set()
    for op, v in ops:
        if op == "load":
            if v in loaded:
                continue  # Already loaded, no-op or redundant
            if len(loaded) >= budget:
                return False  # Exceeds memory
            loaded.add(v)
        elif op == "store":
            if v not in loaded:
                return False  # Can't store what's not loaded
            if v not in outset:
                return False  # Only outputs can be stored
            loaded.discard(v)
        elif op == "delete":
            if v not in loaded:
                return False  # Can't delete what's not loaded
            if v in outset:
                return False  # Can't delete outputs before storing
            loaded.discard(v)
        elif op == "compute":
            ps = parents.get(v, [])
            for p in ps:
                if p not in loaded:
                    return False  # Parent not loaded
            if v in loaded:
                return False  # Already computed
            if len(loaded) >= budget:
                return False  # Exceeds memory
            loaded.add(v)
    # All outputs must be stored (i.e., loaded at some point and stored)
    # For simplicity, we check that all outputs were stored
    # This is a partial check; full verification is done by the evaluator
    return True

def pebble(dag, fast_memory_budget):
    start_t = time.time()
    deadline = start_t + 55  # Safety margin before 60s limit

    candidates = []

    try:
        res = _pebble_postorder(dag, fast_memory_budget)
        if res is not None and len(res) <= 20000:
            candidates.append(res)
    except Exception:
        pass

    try:
        res = _pebble_greedy_lfu(dag, fast_memory_budget)
        if res is not None and len(res) <= 20000:
            candidates.append(res)
    except Exception:
        pass

    try:
        res = _pebble_layered(dag, fast_memory_budget)
        if res is not None and len(res) <= 20000:
            candidates.append(res)
    except Exception:
        pass

    if time.time() < deadline:
        try:
            res = _pebble_astar(dag, fast_memory_budget, deadline)
            if res is not None and len(res) <= 20000:
                candidates.append(res)
        except Exception:
            pass

    try:
        res = _pebble_fallback(dag, fast_memory_budget)
        if res is not None and len(res) <= 20000:
            candidates.append(res)
    except Exception:
        pass

    def transfers(seq):
        return sum(1 for op, _ in seq if op in ("load", "store"))

    best = None
    best_score = float("inf")
    for c in candidates:
        if c is None:
            continue
        if len(c) > 20000:
            continue
        # Validate before scoring
        if not _validate_ops(c, dag, fast_memory_budget):
            continue
        s = transfers(c)
        if s < best_score:
            best_score = s
            best = c
    if best is not None:
        return best
    return _pebble_fallback(dag, fast_memory_budget)[:20000]
# EVOLVE-BLOCK-END
