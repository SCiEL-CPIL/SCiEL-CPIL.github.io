import time
import random
from collections import deque

_T0 = time.time()
_DEADLINE = 145.0  # wall-clock limit with margin (evaluator kills at 150s)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _adj_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_long_cycle_dfs(adj, n, start, max_depth, deadline):
    """Backtracking DFS to find a long cycle through start. Returns cycle or None."""
    best = [None]
    best_len = [0]
    path = [start]
    used = {start}

    def dfs(v, depth):
        if time.time() > deadline:
            return
        if depth >= max_depth:
            return
        # Check if we can close the cycle
        if start in adj[v] and depth >= 3:
            if depth + 1 > best_len[0]:
                best_len[0] = depth + 1
                best[0] = path[:] + [start]
            return
        # Explore neighbors
        neighbors = list(adj[v])
        # Sort by degree (prefer high-degree neighbors for longer paths)
        neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
        for w in neighbors:
            if w in used:
                continue
            used.add(w)
            path.append(w)
            dfs(w, depth + 1)
            path.pop()
            used.discard(w)
            if time.time() > deadline:
                return
            if best_len[0] >= max_depth:
                return

    dfs(start, 1)
    return best[0]


def _find_long_cycle(adj, n, deadline, max_depth=30):
    """Find a long cycle in the graph. Try multiple starting vertices."""
    best_cycle = None
    best_len = 0

    # Try from high-degree vertices first
    starts = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    for start in starts:
        if time.time() > deadline:
            break
        if len(adj[start]) < 2:
            continue
        # Limit search per start
        start_deadline = min(deadline, time.time() + 0.5)
        found = _find_long_cycle_dfs(adj, n, start, max_depth, start_deadline)
        if found and len(found) > best_len:
            best_cycle = found
            best_len = len(found)
            if best_len >= n:  # Hamiltonian cycle
                break
    return best_cycle


def _find_short_cycle(adj, n):
    """Quick BFS-based cycle finding. Returns cycle or None."""
    for start in range(n):
        if len(adj[start]) < 2:
            continue
        parent = {start: -1}
        depth = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for w in adj[v]:
                if w not in depth:
                    depth[w] = depth[v] + 1
                    parent[w] = v
                    q.append(w)
                elif w != parent.get(v, -1) and depth[w] < depth[v] - 1:
                    cycle = []
                    cur = v
                    while cur != w:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(w)
                    return cycle
    return None


def _decompose_into_cycles(adj, n, deadline):
    """Decompose a graph into cycles (assumes all vertices have even degree or we handle odd).
    Returns list of cycles and the remaining edges."""
    cycles = []
    remaining_edges = set()

    # Work on a copy
    work_adj = [set(s) for s in adj]

    while True:
        if time.time() > deadline:
            break
        # Check if any edges remain
        has_edge = False
        for v in range(n):
            if work_adj[v]:
                has_edge = True
                break
        if not has_edge:
            break

        # Find a long cycle
        cycle = _find_long_cycle(work_adj, n, min(deadline, time.time() + 1.0), max_depth=30)
        if cycle is None:
            cycle = _find_short_cycle(work_adj, n)
        if cycle is None:
            # No more cycles; collect remaining edges
            for v in range(n):
                for w in work_adj[v]:
                    if v < w:
                        remaining_edges.add((v, w))
            break

        # Remove cycle edges
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            work_adj[u].discard(v)
            work_adj[v].discard(u)
        cycles.append(cycle)

    return cycles, remaining_edges


def _find_even_subgraph(adj, n, edges, deadline):
    """Find a large even-degree subgraph. 
    Strategy: Start with all edges, then find and remove cycles to make degrees even.
    Actually, we want to KEEP edges in cycles. Let me think differently.
    
    Better: Find a maximum set of edges that can be decomposed into cycles.
    This is equivalent to finding a maximum even-degree subgraph.
    
    Approach: 
    1. Start with all edges
    2. While there are odd-degree vertices, find a path between two odd vertices and remove it
       (this makes them even)
    3. The remaining edges form an even-degree subgraph
    
    Actually, the standard approach:
    - The minimum number of edges to remove to make all degrees even is related to 
      finding minimum T-join where T is the set of odd-degree vertices.
    - But we want to MAXIMIZE kept edges, so minimize removed edges.
    
    Simpler greedy approach:
    1. Find cycles and keep them
    2. For remaining edges, try to pair up odd-degree vertices with paths
    
    Let me use a different approach:
    1. Decompose into cycles greedily
    2. For remaining edges (which form a graph with at most some odd vertices),
       try to find paths between odd vertices to convert more edges into cycles
    """
    # First, decompose what we can into cycles
    cycles, remaining = _decompose_into_cycles(adj, n, deadline)
    
    # Now try to improve: find paths in remaining edges between odd-degree vertices
    # to create more cycles
    # Actually, let's just return what we have for now
    return cycles, remaining


def _validate(adj, n, pieces):
    """Check pieces are edge-disjoint and cover all edges exactly."""
    seen = set()
    for p in pieces:
        k = len(p)
        if k == 2:
            e = tuple(sorted(p))
        elif k >= 3:
            for i in range(k):
                e = tuple(sorted((p[i], p[(i + 1) % k])))
                if e in seen:
                    return False
                seen.add(e)
            continue
        else:
            return False
        if e in seen:
            return False
        seen.add(e)
    all_edges = set()
    for v in range(n):
        for w in adj[v]:
            if v < w:
                all_edges.add((v, w))
    return seen == all_edges


def _solve_one(n, p, seed, time_budget):
    edges = _instance(n, p, seed)
    adj = _adj_from_edges(edges, n)
    m = len(edges)

    # Baseline: all single edges
    best_pieces = [[u, v] for (u, v) in edges]
    best_count = m

    deadline = _T0 + _DEADLINE - 2  # Safety margin

    # Strategy: Decompose into cycles greedily
    try:
        cycles, remaining = _decompose_into_cycles(adj, n, deadline)
        
        pieces = []
        for c in cycles:
            pieces.append(c)
        for (u, v) in remaining:
            pieces.append([u, v])

        if _validate(adj, n, pieces) and len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
    except Exception:
        pass

    # Try a second pass with different starting points
    if time.time() < deadline - 5:
        try:
            work_adj = [set(s) for s in adj]
            pieces2 = []
            remaining2 = set()
            
            # Use a different strategy: prioritize finding cycles through high-degree vertices
            # by trying to build cycles incrementally
            
            while True:
                if time.time() > deadline:
                    break
                has_edge = False
                for v in range(n):
                    if work_adj[v]:
                        has_edge = True
                        break
                if not has_edge:
                    break
                
                # Find cycle starting from highest degree vertex
                best_start = max(range(n), key=lambda v: len(work_adj[v]))
                if len(work_adj[best_start]) < 2:
                    # No more cycles possible
                    for v in range(n):
                        for w in list(work_adj[v]):
                            if v < w:
                                remaining2.add((v, w))
                                work_adj[v].discard(w)
                                work_adj[w].discard(v)
                    break
                
                cycle = _find_long_cycle_dfs(work_adj, n, best_start, 30, min(deadline, time.time() + 0.8))
                if cycle is None:
                    cycle = _find_short_cycle(work_adj, n)
                if cycle is None:
                    for v in range(n):
                        for w in list(work_adj[v]):
                            if v < w:
                                remaining2.add((v, w))
                                work_adj[v].discard(w)
                                work_adj[w].discard(v)
                    break
                
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    work_adj[u].discard(v)
                    work_adj[v].discard(u)
                pieces2.append(cycle)
            
            for (u, v) in remaining2:
                pieces2.append([u, v])

            if _validate(adj, n, pieces2) and len(pieces2) < best_count:
                best_count = len(pieces2)
                best_pieces = pieces2
        except Exception:
            pass

    return best_pieces


def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() - _T0 > _DEADLINE - 3:
            # Fallback: all single edges
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([[u, v] for (u, v) in edges])
            continue
        try:
            pieces = _solve_one(spec[0], spec[1], spec[2], time_budget=40.0)
        except Exception:
            edges = _instance(spec[0], spec[1], spec[2])
            pieces = [[u, v] for (u, v) in edges]
        results.append(pieces)
    return results
