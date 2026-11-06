"""Cycle decomposition via maximum matching on line graph (bipartite)."""
import time
import random
from collections import deque

_DEADLINE = time.time() + 140.0


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hopcroft_karp(left_nodes, right_nodes, adj):
    """Hopcroft-Karp maximum bipartite matching.
    adj: dict mapping left node -> list of right nodes
    Returns list of (left, right) matched pairs.
    """
    pair_u = {u: None for u in left_nodes}
    pair_v = {v: None for v in right_nodes}
    dist = {}

    def bfs():
        q = deque()
        for u in left_nodes:
            if pair_u[u] is None:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                pu = pair_v[v]
                if pu is None:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj.get(u, []):
            pu = pair_v[v]
            if pu is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in left_nodes:
            if pair_u[u] is None:
                if dfs(u):
                    matching += 1

    return [(u, pair_u[u]) for u in left_nodes if pair_u[u] is not None]


def _max_cycles_line_graph(edges, adj):
    """Find maximum set of edge-disjoint cycles using bipartite matching on line graph.
    edges: list of (u,v) tuples
    adj: dict vertex -> set of neighbors
    Returns list of cycles (each cycle is a list of vertices).
    """
    m = len(edges)
    if m < 3:
        return []

    # Index edges
    edge_idx = {}
    for i, e in enumerate(edges):
        edge_idx[e] = i
        edge_idx[(e[1], e[0])] = i

    # Build bipartite graph: left[i] -> right[j] if edge i can precede edge j in a cycle
    # edge i=(a,b), edge j=(c,d): i can precede j if b==c or b==d
    # But we need to track orientation. Let's use a different approach.
    # Actually for the line graph matching approach:
    # Left node i represents edge i. Right node j represents edge j.
    # Left i connects to right j if edge i and edge j share a vertex and can be consecutive in a cycle.
    # This is tricky because orientation matters.

    # Simpler approach: just do greedy cycle finding with good heuristics
    # The matching approach for cycle packing in general graphs is complex.
    # Let's use a better greedy approach instead.
    return []


def _find_cycle_greedy(adj, start, used_edges):
    """Find a cycle starting from vertex using greedy DFS avoiding used edges."""
    visited = {start: None}
    stack = [(start, None)]
    while stack:
        v, parent = stack.pop()
        for w in adj.get(v, []):
            e = (min(v, w), max(v, w))
            if e in used_edges:
                continue
            if w == start and len(visited) >= 3:
                # Found cycle
                cycle = [w]
                cur = v
                while cur != start:
                    cycle.append(cur)
                    cur = visited[cur]
                cycle.append(start)
                return cycle
            if w not in visited:
                visited[w] = v
                stack.append((w, v))
    return None


def _decompose_greedy(adj, edges):
    """Decompose graph into cycles and single edges using greedy longest cycle finding."""
    used_edges = set()
    pieces = []
    remaining_edges = list(edges)

    # Sort edges by degree of endpoints to find "rich" cycles first
    vertex_degree = {v: 0 for e in edges for v in e}
    for e in edges:
        vertex_degree[e[0]] += 1
        vertex_degree[e[1]] += 1

    while remaining_edges:
        # Try to find a cycle
        best_cycle = None
        # Try starting from each unused edge's endpoints
        for e in remaining_edges:
            u, v = e
            if e in used_edges:
                continue
            # Try DFS from u avoiding used edges
            cycle = _find_cycle_greedy(adj, u, used_edges)
            if cycle and len(cycle) > 3:
                if best_cycle is None or len(cycle) > len(best_cycle):
                    best_cycle = cycle
                if len(cycle) >= 8:  # Good enough
                    break

        if best_cycle:
            # Mark all edges in cycle as used
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = (min(a, b), max(a, b))
                used_edges.add(e)
            pieces.append(best_cycle)
        else:
            # No more cycles, take remaining edges as single edges
            for e in remaining_edges:
                if e not in used_edges:
                    pieces.append([e[0], e[1]])
                    used_edges.add(e)
            break

    return pieces


def _decompose_with_matching(adj, edges):
    """Use bipartite matching on the line graph to find maximum cycle packing.
    
    For each edge (u,v), create a left node and right node.
    Left (u,v) connects to right (v,w) and right (w,v) if (v,w) is an edge.
    A matching in this bipartite graph corresponds to a collection of cycles and paths.
    Cycles in the matching correspond to cycles in the original graph.
    """
    m = len(edges)
    if m < 3:
        return [[e[0], e[1]] for e in edges]

    # Map edges to indices
    edge_to_idx = {}
    for i, e in enumerate(edges):
        edge_to_idx[e] = i

    # Build adjacency for bipartite graph
    # Left i (edge i = (a,b)) -> Right j (edge j = (c,d)) if b==c or b==d
    # i.e., edge i ends at a vertex that is the start of edge j
    adj_left = {i: [] for i in range(m)}
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if i == j:
                continue
            if b == c or b == d:
                adj_left[i].append(j)

    left_nodes = list(range(m))
    right_nodes = list(range(m))
    matching = _hopcroft_karp(left_nodes, right_nodes, adj_left)

    # Reconstruct cycles from matching
    # matched[i] = j means edge i is followed by edge j
    next_edge = {}
    prev_edge = {}
    for (i, j) in matching:
        next_edge[i] = j
        prev_edge[j] = i

    used = set()
    cycles = []
    for i in range(m):
        if i in used:
            continue
        # Follow the chain
        chain = [i]
        cur = i
        while cur in next_edge:
            cur = next_edge[cur]
            chain.append(cur)
            if cur in used:
                break
        # Check if it forms a cycle
        if len(chain) >= 3 and chain[0] in chain[-1:]:
            pass
        # Actually need to check if the chain forms a cycle
        # A cycle means: following next from i eventually returns to i
        # Let's check properly
        if i in next_edge:
            # Follow from i
            path = [i]
            cur = i
            while cur in next_edge and cur not in used:
                cur = next_edge[cur]
                path.append(cur)
                if cur == i:
                    break
            if path[-1] == i and len(path) >= 3:
                # Extract the cycle
                cycle_edges = path[:-1]  # Remove the repeated i
                # Convert to vertex cycle
                vertices = [edges[cycle_edges[0]][0]]
                for idx in cycle_edges:
                    e = edges[idx]
                    vertices.append(e[1])
                # Check if it's valid
                if len(vertices) >= 3 and vertices[0] != vertices[-1]:
                    # Verify all edges exist
                    valid = True
                    for k in range(len(vertices)):
                        a, b = vertices[k], vertices[(k + 1) % len(vertices)]
                        e = (min(a, b), max(a, b))
                        if e not in edge_to_idx:
                            valid = False
                            break
                    if valid:
                        for idx in cycle_edges:
                            used.add(idx)
                        cycles.append(vertices)

    # Remaining edges as single edges
    pieces = []
    for c in cycles:
        pieces.append(c)
    for i in range(m):
        if i not in used:
            e = edges[i]
            pieces.append([e[0], e[1]])

    return pieces


def _validate(adj, edges, pieces):
    """Check pieces are edge-disjoint and cover all edges exactly."""
    seen = set()
    all_edges = set(edges)
    for p in pieces:
        k = len(p)
        if k == 2:
            e = (min(p[0], p[1]), max(p[0], p[1]))
            if e in seen or e not in all_edges:
                return False
            seen.add(e)
        elif k >= 3:
            for i in range(k):
                a, b = p[i], p[(i + 1) % k]
                e = (min(a, b), max(a, b))
                if e in seen or e not in all_edges:
                    return False
                seen.add(e)
        else:
            return False
    return seen == all_edges


def _solve_one(edges):
    """Solve one instance. Returns list of pieces."""
    m = len(edges)
    if m == 0:
        return []

    # Build adjacency
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    # Method 1: Greedy cycle finding
    pieces1 = _decompose_greedy(adj, edges)
    count1 = len(pieces1)

    # Method 2: Matching-based
    pieces2 = None
    try:
        if time.time() < _DEADLINE - 10:
            pieces2 = _decompose_with_matching(adj, edges)
            if _validate(adj, edges, pieces2):
                pass
            else:
                pieces2 = None
    except Exception:
        pieces2 = None

    # Choose best
    if pieces2 is not None:
        count2 = len(pieces2)
        if count2 < count1:
            return pieces2
    return pieces1


def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        edges = _instance(*spec)
        if time.time() > _DEADLINE - 5:
            # Fallback
            results.append([[u, v] for u, v in edges])
            continue
        pieces = _solve_one(edges)
        # Validate
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        if not _validate(adj, edges, pieces):
            pieces = [[u, v] for u, v in edges]
        results.append(pieces)
    return results
