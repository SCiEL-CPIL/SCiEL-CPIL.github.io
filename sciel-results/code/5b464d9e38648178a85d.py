"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    ops = []
    cost_cache = {}
    def ccost(v):
        if v not in cost_cache:
            ps = parents.get(v, [])
            cost_cache[v] = 1 + (sum(ccost(p) for p in ps) if ps else 0)
        return cost_cache[v]
    for v in vertices:
        ccost(v)
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
    outset = set(outputs)
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
    # Pebble a subtree (pebbling its children, computing, unpebbling intermediates),
    # keeping only outputs resident. Minimizes transfers vs naive full-load.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    # children map (vertex -> list of vertices that depend on it)
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)

    # topological order to process; we need postorder (children before parent)
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
    # reverse topo = postorder (leaves/children first)
    post = list(reversed(topo))

    ops = []
    loaded = set()  # vertices currently in fast memory (blue)
    stored = set()  # outputs that have been stored (finished blue)

    def pebble(v):
        # Ensure v is computed and blue (in fast memory)
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p)
        ops.append(["compute", v])
        loaded.add(v)
        # unpebble parents that are no longer needed (not outputs, not needed by other children)
        for p in ps:
            if p in outset or p in stored:
                continue
            # is p needed by any other not-yet-processed vertex?
            needed = False
            for w in children.get(p, []):
                if w != v and w not in loaded and w not in stored:
                    # check if w still needs p (i.e., w not yet computed)
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
        else:
            # non-output: only pebble if needed by some output downstream
            # (pebble lazily via outputs; here we skip to save ops)
            pass

    return ops[:20000]

def _pebble_mincostflow(dag, budget, deadline):
    """
    Formulate pebbling as minimum-cost flow on a vertex-splitting network.
    Each DAG node v is split into v_in and v_out with edge (v_in, v_out) cap=1, cost=1.
    For each parent p of v, add edge (p_out, v_in) cap=inf, cost=0 (free reuse if p is still in memory).
    Source connects to all input nodes (no parents) with cap=inf, cost=0.
    Each output node connects to sink with cap=1, cost=0.
    We want to send |outputs| units of flow from source to sink, minimizing total cost.
    The cost represents the number of loads (each time we traverse v_in->v_out, we "load" v).
    Then reconstruct the schedule from the flow.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    
    idx = {v: i for i, v in enumerate(vertices)}
    outset = set(outputs)
    outset_idx = set(idx[o] for o in outputs)
    
    # Build adjacency for the flow network
    # Nodes: 0..2*n-1 where node i_in = 2*i, i_out = 2*i+1
    # Plus source = 2*n, sink = 2*n+1
    N = 2 * n + 2
    SRC = 2 * n
    SNK = 2 * n + 1
    
    # Adjacency list: adj[u] = list of (v, cap, cost, edge_id)
    adj = [[] for _ in range(N)]
    # Reverse edges for residual graph
    rev_adj = [[] for _ in range(N)]
    # Edge storage: edges[edge_id] = (u, v, cap, cost, flow)
    edges = []
    
    def add_edge(u, v, cap, cost):
        eid = len(edges)
        edges.append([u, v, cap, cost, 0])
        adj[u].append((v, eid))
        # Reverse edge
        eid_rev = len(edges)
        edges.append([v, u, 0, -cost, 0])
        adj[v].append((u, eid_rev))
        return eid
    
    # For each vertex v:
    # - Add edge (v_in, v_out) with cap=1, cost=1 (loading v costs 1)
    # - If v has no parents (input), add edge (SRC, v_in) with cap=inf, cost=0
    # - If v is an output, add edge (v_out, SNK) with cap=1, cost=0
    # - For each parent p of v, add edge (p_out, v_in) with cap=inf, cost=0
    
    INF_CAP = n + len(outputs) + 1
    
    for i, v in enumerate(vertices):
        vin = 2 * i
        vout = 2 * i + 1
        # Load edge
        add_edge(vin, vout, 1, 1)
        # Input source edge
        if not parents.get(v, []):
            add_edge(SRC, vin, INF_CAP, 0)
        # Output sink edge
        if v in outset:
            add_edge(vout, SNK, 1, 0)
        # Parent edges
        for p in parents.get(v, []):
            if p in idx:
                pout = 2 * idx[p] + 1
                add_edge(pout, vin, INF_CAP, 0)
    
    # Min-cost max-flow using successive shortest paths with potentials
    # We need to send |outputs| units of flow
    target_flow = len(outputs)
    if target_flow == 0:
        return []
    
    # Bellman-Ford for initial potentials
    def bellman_ford(s):
        dist = [float('inf')] * N
        dist[s] = 0
        for _ in range(N - 1):
            updated = False
            for u in range(N):
                if dist[u] == float('inf'):
                    continue
                for v, eid in adj[u]:
                    if edges[eid][2] > 0 and dist[u] + edges[eid][3] < dist[v]:
                        dist[v] = dist[u] + edges[eid][3]
                        updated = True
            if not updated:
                break
        return dist
    
    potentials = bellman_ford(SRC)
    
    total_cost = 0
    flow_sent = 0
    
    while flow_sent < target_flow:
        if time.time() > deadline:
            break
        # Dijkstra with potentials
        dist = [float('inf')] * N
        prev_node = [-1] * N
        prev_edge = [-1] * N
        dist[SRC] = 0
        heap = [(0, SRC)]
        visited = [False] * N
        
        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v, eid in adj[u]:
                if edges[eid][2] > 0:
                    # Reduced cost
                    rc = edges[eid][3] + potentials[u] - potentials[v]
                    if dist[u] + rc < dist[v]:
                        dist[v] = dist[u] + rc
                        prev_node[v] = u
                        prev_edge[v] = eid
                        heapq.heappush(heap, (dist[v], v))
        
        if dist[SNK] == float('inf'):
            break  # No more augmenting path
        
        # Update potentials
        for i in range(N):
            if dist[i] < float('inf'):
                potentials[i] += dist[i]
        
        # Find bottleneck
        path_flow = float('inf')
        v = SNK
        while v != SRC:
            eid = prev_edge[v]
            path_flow = min(path_flow, edges[eid][2])
            v = prev_node[v]
        
        if path_flow == 0 or path_flow == float('inf'):
            break
        
        # Augment
        v = SNK
        while v != SRC:
            eid = prev_edge[v]
            edges[eid][4] += path_flow
            edges[eid ^ 1][4] -= path_flow
            v = prev_node[v]
        
        flow_sent += path_flow
        total_cost += path_flow * (potentials[SNK] - potentials[SRC])
    
    if flow_sent < target_flow:
        return None  # Infeasible
    
    # Reconstruct schedule from flow
    # For each vertex, if flow on (v_in, v_out) is 1, it was loaded once.
    # We need to produce a valid sequence of operations.
    # Strategy: topologically process vertices. For each vertex, if it was loaded,
    # ensure its parents are loaded first (they should be if flow exists on parent edges).
    
    # Determine which vertices were loaded (flow > 0 on their load edge)
    loaded_vertices = []
    for i, v in enumerate(vertices):
        vin = 2 * i
        vout = 2 * i + 1
        # Find the forward edge (vin, vout)
        for v2, eid in adj[vin]:
            if v2 == vout and edges[eid][3] == 1:
                if edges[eid][4] > 0:
                    loaded_vertices.append(v)
                break
    
    # Build children map
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    
    # Topological order
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
    
    # Generate operations: for each vertex in topological order, if it was loaded,
    # emit load + compute. If it's an output, emit store. Delete when no longer needed.
    ops = []
    in_memory = set()
    loaded_set = set(loaded_vertices)
    
    # Determine for each vertex which children need it
    # A vertex p is needed by child v if flow went from p_out to v_in
    needed_by = {v: set() for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in idx and v in idx:
                pout = 2 * idx[p] + 1
                vin = 2 * idx[v]
                # Check if there's flow on edge (pout, vin)
                for v2, eid in adj[pout]:
                    if v2 == vin and edges[eid][3] == 0:
                        if edges[eid][4] > 0:
                            needed_by[p].add(v)
                        break
    
    for v in topo:
        if v not in loaded_set:
            continue
        # Load v (if not already in memory from a previous load - but in this formulation,
        # each vertex is loaded at most once due to cap=1 on load edge)
        if v not in in_memory:
            # Need to ensure parents are in memory
            for p in parents.get(v, []):
                if p in loaded_set and p not in in_memory:
                    # This shouldn't happen if flow is consistent, but handle it
                    ops.append(["load", p])
                    ops.append(["compute", p])
                    in_memory.add(p)
            ops.append(["load", v])
            ops.append(["compute", v])
            in_memory.add(v)
        
        if v in outset:
            ops.append(["store", v])
            in_memory.discard(v)
        else:
            # Check if v is still needed by any unprocessed child
            still_needed = False
            for w in needed_by.get(v, []):
                if w not in in_memory and w in loaded_set:
                    # w will be processed later and needs v
                    # Check if w is not yet processed (w comes after v in topo)
                    if topo.index(w) > topo.index(v):
                        still_needed = True
                        break
            if not still_needed:
                ops.append(["delete", v])
                in_memory.discard(v)
    
    if len(ops) > 20000:
        return None
    
    return ops

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    # A* over subsets is exponential; only attempt for small n within deadline.
    if n > 14:
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
    start_t = time.time()
    deadline = start_t + 50  # safety margin before 60s / 90s kill

    candidates = []

    # 1) Try the efficient postorder pebbling (polynomial, minimizes transfers).
    try:
        res = _pebble_postorder(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

    # 2) Try min-cost flow formulation (polynomial, may find better schedule).
    if time.time() < deadline:
        try:
            res = _pebble_mincostflow(dag, fast_memory_budget, deadline)
            if res is not None and len(res) <= 20000:
                candidates.append(res)
        except Exception:
            pass

    # 3) Try A* only for small instances and if time remains.
    if time.time() < deadline:
        try:
            res = _pebble_astar(dag, fast_memory_budget, deadline)
            if res is not None and len(res) <= 20000:
                candidates.append(res)
        except Exception:
            pass

    # 4) Greedy fallback.
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        res = pebble_baseline(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass
    try:
        res = _pebble_fallback(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

    # Pick the candidate with the fewest transfers (load + store), valid and within op limit.
    def transfers(seq):
        return sum(1 for op, _ in seq if op in ("load", "store"))

    best = None
    best_score = float("inf")
    for c in candidates:
        if c is None:
            continue
        if len(c) > 20000:
            continue
        s = transfers(c)
        if s < best_score:
            best_score = s
            best = c
    if best is not None:
        return best
    # last resort
    return _pebble_fallback(dag, fast_memory_budget)[:20000]
# EVOLVE-BLOCK-END
