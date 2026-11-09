"""Decompose each fixed instance graph into edge-disjoint cycles + single edges."""
import time
import random
from random import Random
import sys


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_all_cycles_up_to_5(adj, n):
    """Find all cycles of length 3, 4, 5 in the graph."""
    mat = [[False] * n for _ in range(n)]
    for u, nbrs in adj.items():
        for v in nbrs:
            mat[u][v] = True
            mat[v][u] = True

    cycles = []

    # 3-cycles
    for u in range(n):
        for v in range(u + 1, n):
            if not mat[u][v]:
                continue
            for w in range(v + 1, n):
                if mat[v][w] and mat[w][u]:
                    cycles.append([u, v, w])

    # 4-cycles
    for u in range(n):
        for v in range(u + 1, n):
            if not mat[u][v]:
                continue
            for w in range(v + 1, n):
                if not mat[v][w]:
                    continue
                for x in range(w + 1, n):
                    if mat[w][x] and mat[x][u] and not mat[u][w] and not mat[v][x]:
                        cycles.append([u, v, w, x])

    # 5-cycles (only for small n to avoid O(n^5) blowup)
    if n <= 25:
        for u in range(n):
            for v in range(u + 1, n):
                if not mat[u][v]:
                    continue
                for w in range(v + 1, n):
                    if not mat[v][w]:
                        continue
                    for x in range(w + 1, n):
                        if not mat[w][x]:
                            continue
                        for y in range(x + 1, n):
                            if mat[x][y] and mat[y][u] and not mat[u][w] and not mat[u][x] and not mat[u][y] and not mat[v][x] and not mat[v][y] and not mat[w][y]:
                                cycles.append([u, v, w, x, y])

    return cycles


def _max_flow_fordfulkerson(source, sink, adj_list, cap):
    """Standard Edmonds-Karp max flow."""
    from collections import deque

    # Build residual graph
    flow = {}
    for u in adj_list:
        flow[u] = {}
        for v in adj_list.get(u, []):
            flow[u][v] = 0

    def bfs():
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in adj_list.get(u, []):
                if v not in parent and cap[u][v] - flow[u].get(v, 0) > 0:
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    total_flow = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        # Find bottleneck
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            residual = cap[u][v] - flow[u].get(v, 0)
            path_flow = min(path_flow, residual)
            v = u
        # Update flows
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] = flow[u].get(v, 0) + path_flow
            if v not in flow[u]:
                flow[u][v] = 0
            flow.setdefault(v, {})
            flow[v][u] = flow[v].get(u, 0) - path_flow
            v = u
        total_flow += path_flow

    return total_flow, flow


def _find_max_2_matching(edges, n, deadline):
    """
    Find maximum set of edge-disjoint cycles using max-flow.
    Model: each edge (u,v) can be 'used' in a cycle. We want to maximize
    the number of edges covered by cycles.
    
    Approach: Use the fact that in a graph, the maximum 2-matching (union of
    cycles and paths) can be found. But we specifically want cycles.
    
    Simpler approach: iteratively find and remove cycles using a clever heuristic
    that prioritizes cycles that are 'harder' to form (shorter, more constrained).
    """
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    avail = set(tuple(sorted(e)) for e in edges)
    cycles = []

    # Strategy: Find all small cycles first, then use a greedy set cover approach
    all_small = _find_all_cycles_up_to_5(adj, n)

    # Sort cycles: prefer shorter cycles (they're more "valuable" per edge)
    # but also prefer cycles that share fewer edges with other cycles
    # Actually, for maximizing total edges covered, we want a max-weight set
    # of edge-disjoint cycles. Use greedy: pick cycle with most edges first?
    # No - pick shortest first to lock in hard-to-form cycles.
    
    # Better: use a weighted greedy where weight = 1/len(cycle) to prefer shorter
    # But actually we want to maximize total edges, so each edge contributes 1.
    # Greedy: repeatedly pick the cycle that covers the most "new" edges.
    
    # For small graphs, try multiple orderings and keep best
    best_cycles = []
    best_edge_count = 0

    orderings = [
        sorted(all_small, key=lambda c: len(c)),          # shortest first
        sorted(all_small, key=lambda c: -len(c)),          # longest first  
        sorted(all_small, key=lambda c: len(c), reverse=True),  # same as above
    ]

    # Also add random orderings
    rng = Random(123)
    for _ in range(5):
        shuffled = all_small[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        if time.time() > deadline:
            break
        
        used_edges = set()
        selected = []
        
        for cyc in ordering:
            if time.time() > deadline:
                break
            cyc_edges = set()
            valid = True
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                e = (min(a, b), max(a, b))
                if e not in avail:
                    valid = False
                    break
                if e in used_edges:
                    valid = False
                    break
                cyc_edges.add(e)
            
            if valid and cyc_edges:
                # Check if this cycle adds value (it always does if valid)
                selected.append(cyc)
                used_edges |= cyc_edges

        if len(used_edges) > best_edge_count:
            best_edge_count = len(used_edges)
            best_cycles = selected[:]

    return best_cycles


def _decompose(edges, deadline):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    avail = set(tuple(sorted(e)) for e in edges)
    pieces = []
    n = max(max(e) for e in edges) + 1 if edges else 0

    # Phase 1: Find maximum set of edge-disjoint small cycles (3-5)
    small_cycles = _find_max_2_matching(edges, n, deadline)
    
    for cyc in small_cycles:
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (min(a, b), max(a, b))
            avail.discard(e)
            if e in avail:
                adj[a].discard(b)
                adj[b].discard(a)

    pieces.extend(small_cycles)

    # Phase 2: Greedy extraction of longer cycles from remaining edges
    def try_greedy_cycle(start, max_attempts=5):
        """Try to extend from start to form a cycle back to start. Dry run."""
        for _ in range(max_attempts):
            path = [start]
            visited = {start}
            cur = start
            for _ in range(len(adj) * 3):
                if time.time() > deadline:
                    return None
                nbrs = [nb for nb in adj[cur] if (min(cur, nb), max(cur, nb)) in avail]
                if not nbrs:
                    return None
                if len(path) >= 3 and start in nbrs:
                    return path[:]
                unvisited = [nb for nb in nbrs if nb not in visited]
                if unvisited:
                    # Prefer neighbors with higher degree (more options)
                    weights = [len(adj[nb]) + 1 for nb in unvisited]
                    total = sum(weights)
                    r = random.random() * total
                    cum = 0
                    chosen = unvisited[0]
                    for nb, w in zip(unvisited, weights):
                        cum += w
                        if r <= cum:
                            chosen = nb
                            break
                else:
                    if start in nbrs and len(path) >= 3:
                        return path[:]
                    return None
                path.append(chosen)
                visited.add(chosen)
                cur = chosen
        return None

    # Rebuild adj from remaining avail edges
    adj = {}
    for e in avail:
        u, v = e
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    while time.time() < deadline:
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break

        best_cycle = None
        best_len = 0

        rng_local = Random(int(time.time() * 1000) % 10000)
        num_starts = min(len(candidates), 30)
        starts = rng_local.sample(candidates, num_starts) if num_starts < len(candidates) else candidates

        for s in starts:
            if time.time() > deadline:
                break
            cycle = try_greedy_cycle(s, max_attempts=8)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle

        if not best_cycle:
            # Try with more attempts on fewer starts
            for s in starts[:10]:
                if time.time() > deadline:
                    break
                cycle = try_greedy_cycle(s, max_attempts=15)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle

        if not best_cycle:
            break

        # Commit the best cycle found
        for i in range(len(best_cycle)):
            a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = (min(a, b), max(a, b))
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(best_cycle)

    for e in sorted(avail):
        pieces.append(list(e))

    return pieces


def construct():
    deadline = time.time() + 135.0
    random.seed(42)
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    total_size = sum(n * n for n, _, _ in specs)
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            results.append([list(e) for e in edges])
            continue
        frac = (n * n) / total_size
        graph_deadline = time.time() + max(5.0, remaining_time * frac * 0.85)
        
        best_pieces = None
        best_count = float('inf')
        
        restarts = 15
        
        for r in range(restarts):
            if time.time() > graph_deadline:
                break
            random.seed(42 + r * 7)
            pieces = _decompose(edges, graph_deadline)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)
    return results
