"""Decompose three fixed graphs into cycles and edges, minimizing piece count.

Strategy:
  1. Build the bipartite double cover of the graph.
  2. Find a maximum 2-matching (each vertex has degree <= 2 in the matching).
     This decomposes into edge-disjoint cycles and paths.
  3. Cycles become cycle-pieces. Paths of length >= 2 are kept as edge-pieces
     (or we try to close them if possible, but since they're paths they can't
     be closed without adding edges).
  4. For remaining edges not in the 2-matching, try to find additional cycles
     greedily, then output remaining edges as single-edge pieces.
  5. Post-process: try to find cycles among the single-edge pieces to merge them.

Actually, a cleaner approach:
  - Use the maximum 2-matching to get a set of edge-disjoint cycles and paths.
  - The cycles are our cycle pieces.
  - The paths contribute edges; we output each edge in a path as a single edge piece,
    OR we try to find cycles in the subgraph of path edges.
  - Remaining edges (not in 2-matching) are handled similarly.

But actually, the maximum 2-matching gives us the maximum number of edges covered by
cycles and paths where each vertex has degree <= 2. The cycles in this decomposition
are edge-disjoint. The paths are also edge-disjoint from each other and from cycles.

Key insight: In a 2-matching, the connected components are cycles and paths.
Each cycle component becomes one piece. Each path component with k edges becomes k edge-pieces
(unless we can do better).

Wait - actually we want to MINIMIZE the number of pieces. A cycle of length k is 1 piece
covering k edges. A path of length k (k edges, k+1 vertices) must be broken into k edge-pieces
since we can't form a cycle from a path.

So the total piece count = (number of cycle components) + (number of edges in path components)
+ (number of edges not in the 2-matching that can't be part of any cycle we find)

To minimize pieces, we want to maximize the number of edges that are in cycles.
The maximum 2-matching maximizes the total edges in cycles+paths, but we specifically want cycles.

Alternative: Find a maximum set of edge-disjoint cycles directly. This is NP-hard in general,
but for small graphs (n<=30) we can use integer programming or clever heuristics.

Let me use a different approach:
  1. Repeatedly find a long cycle, remove it, repeat.
  2. Use a smarter cycle-finding: try to find the longest cycle via DFS with backtracking
     for small graphs, or use a heuristic for larger ones.
  3. After all cycles are found, the remaining edges form a graph with max degree <= 1
     (matching) if we're lucky, or we try to find more cycles.

Actually, for n<=30 graphs, let me try:
  - Use a greedy approach: repeatedly find the longest cycle (using a heuristic), remove it.
  - For finding "longest cycle", use: start from each vertex, do DFS trying to find the
    longest path back to start. For n<=30 this is feasible with good pruning.

Let me try yet another approach:
  - Formulate as: find maximum number of edge-disjoint cycles.
  - Use a flow-based or matching-based approach.

The cleanest formulation: A set of edge-disjoint cycles corresponds to a 2-factor in a
subgraph. We want to maximize the number of edges covered by cycles.

For small graphs, let me just use a good greedy:
  1. While there's a cycle in the remaining graph, find a "good" cycle (long) and remove it.
  2. The key is finding LONG cycles to minimize piece count.

For finding long cycles in graphs with n<=30:
  - For each vertex v, try to find the longest simple path from v back to v (i.e., longest cycle through v).
  - This is the longest cycle problem, NP-hard, but for n=30 we can use DFS with memoization
    or branch-and-bound.

Actually, let me use a practical approach:
  - Use a "maximum cycle packing" via repeated removal of cycles found by a good heuristic.
  - The heuristic: for each start vertex, do a DFS that tries to find the longest cycle.
    Use a priority queue / best-first search.

For n<=30, a DFS-based longest cycle search should be fast enough if we limit the search.

Let me implement:
  1. _find_longest_cycle(adj, deg, n, start): DFS from start, try to find longest cycle back to start.
     Use iterative deepening or just DFS with a global best.
  2. Main loop: find longest cycle, remove it, repeat.
  3. Remaining edges: try to find any cycles, then output as edge pieces.

For the DFS longest cycle: since n<=30, the path length is at most 30. We can do a DFS
that tracks visited vertices and tries to extend the path. At each step, if we're at a
neighbor of start and path length >= 3, we have a cycle. We want the longest one.

To make this efficient, I'll use a recursive DFS with pruning:
  - If current path length + (n - len(visited)) <= best_found, prune.
  - Try neighbors in order of decreasing degree.

Let me code this up.
"""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 110.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(edges, n):
    """Build adjacency list and degree counts."""
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg


def _find_longest_cycle_dfs(adj, deg, n, start, time_limit):
    """
    Find the longest cycle through `start` using DFS with pruning.
    Returns the cycle as a list of vertices, or None.
    """
    best_cycle = [None]
    start_time = time.time()

    # We'll do a DFS from start. Path is a list of vertices.
    # visited is a set of vertices in the current path.
    # We want to find the longest path from start back to start with length >= 3.

    def dfs(node, path, visited):
        if time.time() - start_time > time_limit:
            return
        # Pruning: even if we visit all remaining unvisited vertices, can we beat best?
        remaining_unvisited = n - len(visited)
        if len(path) + remaining_unvisited <= (len(best_cycle[0]) if best_cycle[0] else 0):
            return

        neighbors = sorted(adj[node], key=lambda x: -deg[x])
        for nb in neighbors:
            if time.time() - start_time > time_limit:
                return
            if nb == start and len(path) >= 3:
                # Found a cycle
                if best_cycle[0] is None or len(path) > len(best_cycle[0]):
                    best_cycle[0] = list(path)
                # Don't continue from here (we closed the cycle)
            elif nb not in visited:
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.discard(nb)

    dfs(start, [start], {start})
    return best_cycle[0]


def _find_cycle_greedy(adj, deg, n, time_limit):
    """
    Find a long cycle in the graph. Try each high-degree vertex as start.
    Return the best cycle found, or None.
    """
    best_cycle = None
    start_time = time.time()

    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None
    candidates.sort(key=lambda v: -deg[v])

    per_start_time = max(0.01, time_limit / max(len(candidates), 1))

    for start in candidates:
        if time.time() - start_time > time_limit:
            break
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break
        cycle = _find_longest_cycle_dfs(adj, deg, n, start, remaining_time)
        if cycle is not None:
            if best_cycle is None or len(cycle) > len(best_cycle):
                best_cycle = cycle
            # If we found a very long cycle, stop early
            if len(cycle) >= n:
                break

    if best_cycle is None:
        # Fallback: find any cycle via BFS
        for u in range(n):
            if deg[u] < 2:
                continue
            for v in adj[u]:
                if u == v:
                    continue
                # BFS from v to u avoiding direct edge
                queue = deque([(v, [v])])
                visited = {v}
                while queue:
                    node, path = queue.popleft()
                    for nb in adj[node]:
                        if nb == u and len(path) >= 2:
                            return path + [u]
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
    return best_cycle


def _remove_edges_of_cycle(cycle, adj, deg):
    """Remove all edges of the cycle from the graph."""
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1


def _remove_single_edge(u, v, adj, deg):
    """Remove a single edge."""
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1


def _decompose_graph(edges, n, time_limit):
    """
    Decompose graph into cycles and edges, minimizing piece count.
    Strategy: repeatedly find the longest cycle, remove it.
    """
    adj, deg = _build_adjacency(edges, n)
    pieces = []
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        # Count remaining edges
        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break

        # Time for one cycle-finding attempt
        cycle_time = min(0.5, (time_limit - elapsed) * 0.3)
        if cycle_time <= 0:
            break

        cycle = _find_cycle_greedy(adj, deg, n, cycle_time)
        if cycle is not None and len(cycle) >= 3:
            pieces.append(cycle)
            _remove_edges_of_cycle(cycle, adj, deg)
        else:
            # No cycle found; take remaining edges as single edge pieces
            break

    # Add remaining edges as single-edge pieces
    remaining_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                remaining_edges.add((u, v))

    # Try to find cycles among remaining edges
    if remaining_edges:
        rem_adj = defaultdict(set)
        rem_deg = [0] * n
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            rem_deg[u] += 1
            rem_deg[v] += 1

        # Try to find cycles in remaining subgraph
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
            rem_count = sum(len(rem_adj[u]) for u in range(n)) // 2
            if rem_count == 0:
                break
            cycle_time = min(0.3, (time_limit - elapsed) * 0.3)
            if cycle_time <= 0:
                break
            cycle = _find_cycle_greedy(rem_adj, rem_deg, n, cycle_time)
            if cycle is not None and len(cycle) >= 3:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                    remaining_edges.discard(tuple(sorted((u, v))))
            else:
                break

    # Add all remaining edges as single-edge pieces
    for u, v in sorted(remaining_edges):
        pieces.append([u, v])

    return pieces


def _improve_pieces(pieces, edges, n, time_limit):
    """
    Try to improve the decomposition by finding cycles among single-edge pieces.
    """
    start_time = time.time()

    # Identify single-edge pieces
    edge_pieces = {}
    cycle_pieces = []
    for i, p in enumerate(pieces):
        if len(p) == 2:
            e = tuple(sorted(p))
            edge_pieces[e] = i
        else:
            cycle_pieces.append(p)

    if len(edge_pieces) < 3:
        return pieces

    # Build subgraph of edge pieces
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edge_pieces:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    # Try to find cycles in this subgraph
    improved = False
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        rem_count = sum(len(adj[u]) for u in range(n)) // 2
        if rem_count < 3:
            break

        cycle_time = min(0.2, (time_limit - elapsed) * 0.3)
        if cycle_time <= 0:
            break

        cycle = _find_cycle_greedy(adj, deg, n, cycle_time)
        if cycle is None or len(cycle) < 3:
            break

        # Verify all edges in cycle are in edge_pieces
        cycle_edges = []
        ok = True
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e not in edge_pieces:
                ok = False
                break
            cycle_edges.append(e)
        if not ok:
            break

        # Replace these edge pieces with a cycle
        cycle_pieces.append(cycle)
        for e in cycle_edges:
            del edge_pieces[e]
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
        improved = True

    if not improved:
        return pieces

    # Rebuild pieces list
    new_pieces = list(cycle_pieces)
    for e in sorted(edge_pieces.keys()):
        new_pieces.append(list(e))
    return new_pieces


def _decompose_with_restarts(edges, n, time_limit, num_restarts=3):
    """Try multiple decompositions and keep the best."""
    best_pieces = None
    best_count = float('inf')
    start_time = time.time()

    for r in range(num_restarts):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 1:
            break
        per_time = remaining / max(num_restarts - r, 1)

        pieces = _decompose_graph(edges, n, per_time)
        count = len(pieces)

        if count < best_count:
            best_count = count
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]

    # Final improvement pass
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 0.5:
        best_pieces = _improve_pieces(best_pieces, edges, n, remaining)

    return best_pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)

        graphs_left = len(specs) - i
        graph_time = remaining * 0.85 / max(graphs_left, 1)

        # More restarts for smaller graphs
        num_restarts = max(2, min(4, 60 // max(n, 1) + 1))

        pieces = _decompose_with_restarts(edges, n, graph_time, num_restarts=num_restarts)
        results.append(pieces)

    return results
