"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _topo_order(parents, vertices, vset):
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
    return topo, children

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices, vset)
    # last-use index for each vertex
    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            last_use[p] = i
    ops = []
    loaded = set()
    for i, v in enumerate(topo):
        for p in parents.get(v, []):
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
        else:
            loaded.add(v)
        # delete parents no longer needed (last use passed, not outputs)
        for p in list(loaded):
            if p not in outset and last_use.get(p, -1) < i:
                ops.append(["delete", p])
                loaded.discard(p)
    return ops[:20000]

def _pebble_postorder(dag, budget):
    # Improved postorder pebbling with better memory management.
    # Process outputs in order of smallest subtree first to minimize peak memory.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    topo, children = _topo_order(parents, vertices, vset)
    
    # Compute subtree sizes to order outputs by smallest first
    subtree_size = {v: 1 for v in vertices}
    for v in reversed(topo):
        for p in parents.get(v, []):
            if p in subtree_size:
                subtree_size[p] = max(subtree_size[p], subtree_size.get(v, 1) + 1)
    
    # Sort outputs by subtree size (smallest first) to reduce memory pressure
    sorted_outputs = sorted(outputs, key=lambda v: subtree_size.get(v, 1))
    
    ops = []
    loaded = set()
    stored = set()
    
    # Precompute last use for each vertex across all output computations
    # We need to track when each vertex is last needed
    last_use = {v: -1 for v in vertices}
    
    # For each output in order, compute the last time each vertex is needed
    for out_v in sorted_outputs:
        # BFS/DFS to find all vertices in subtree
        def get_subtree_vertices(v, visited=None):
            if visited is None:
                visited = set()
            if v in visited:
                return
            visited.add(v)
            for p in parents.get(v, []):
                get_subtree_vertices(p, visited)
            return visited
        
        subtree = get_subtree_vertices(out_v)
        # For vertices in this subtree, update last_use
        # We process in reverse topological order within the subtree
        sub_topo = [v for v in reversed(topo) if v in subtree]
        for i, v in enumerate(sub_topo):
            for p in parents.get(v, []):
                if p in subtree:
                    last_use[p] = max(last_use[p], i)
    
    def pebble(v, current_output):
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p, current_output)
        ops.append(["compute", v])
        loaded.add(v)
        
        # Delete parents that are no longer needed for any remaining output
        for p in ps:
            if p in outset or p in stored:
                continue
            # Check if p is needed for any remaining output
            needed = False
            for remaining_out in sorted_outputs:
                if remaining_out == current_output:
                    continue
                # Check if p is in the subtree of remaining_out and still needed
                def is_in_subtree(target, start):
                    stack = [start]
                    seen = set()
                    while stack:
                        node = stack.pop()
                        if node in seen:
                            continue
                        seen.add(node)
                        if node == target:
                            return True
                        for pp in parents.get(node, []):
                            stack.append(pp)
                    return False
                
                if is_in_subtree(p, remaining_out):
                    needed = True
                    break
            
            if not needed:
                ops.append(["delete", p])
                loaded.discard(p)
    
    for v in sorted_outputs:
        pebble(v, v)
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
    if n > 15:
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
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)

    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")

    # Precompute for each vertex: which outputs depend on it
    output_deps = [0] * n  # bitmask of outputs that depend on vertex i
    for o in outputs:
        oi = idx[o]
        # BFS to find all ancestors of o
        stack = [oi]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            output_deps[node] |= (1 << oi)
            for p in parents.get(vertices[node], []):
                if p in idx:
                    stack.append(idx[p])

    def heuristic(loaded):
        # Improved heuristic: for each missing output, estimate minimum transfers needed
        # An output v needs: load of all parents not in loaded + compute + store
        # But we can recompute, so it's more complex. Use a lower bound.
        h = 0
        for o in outputs:
            oi = idx[o]
            if loaded & (1 << oi):
                continue
            # This output is not loaded
            # At minimum, we need to compute it (1 op) and store it (1 op)
            # Plus we need all its parents loaded
            par = par_mask[oi]
            missing_parents = par & ~loaded
            h += 2 + popcount(missing_parents)  # compute + store + loads
        return h

    start = 0
    goal = out_mask
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    best = None
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, state = heapq.heappop(open_heap)
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
                if popcount(state) < b:  # respect memory budget
                    ns = state | bit
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
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 40  # safety margin before 60s / 90s kill
    candidates = []
    
    # Try A* for small DAGs (up to 15 vertices)
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Postorder pebbling with improved memory management
    try:
        r = _pebble_postorder(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    # Greedy fallback
    try:
        r = _pebble_fallback(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass
    
    if not candidates:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []
    
    # Pick candidate with fewest transfers (load+store)
    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))
    best = min(candidates, key=transfers)
    return best
# EVOLVE-BLOCK-END
