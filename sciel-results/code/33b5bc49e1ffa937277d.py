"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {
    'n': 24,
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
              [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
              [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21],
              [21, 22], [22, 23]],
    'diameter': 4
}


def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _bfs_distances(adj, src):
    n = P['n']
    dist = [-1] * n
    dist[src] = 0
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _is_connected(adj):
    n = P['n']
    if n == 0:
        return True
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == n


def _is_triangle_free(adj):
    n = P['n']
    for u in range(n):
        for i in range(len(adj[u])):
            for j in range(i + 1, len(adj[u])):
                if adj[u][i] in adj[adj[u][j]]:
                    return False
    return True


def _max_diameter(adj):
    n = P['n']
    maxd = 0
    for src in range(n):
        dist = _bfs_distances(adj, src)
        if -1 in dist:
            return float('inf')
        maxd = max(maxd, max(dist))
    return maxd


def _validate_solution(added_edges):
    all_edges = P['edges'] + added_edges
    adj = _build_adjacency(all_edges)
    if not _is_connected(adj):
        return False
    if not _is_triangle_free(adj):
        return False
    d = _max_diameter(adj)
    return d <= 4


def _edge_distance_in_path(u, v):
    # Distance in the original path
    return abs(u - v)


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s to be safe within 90s limit

    # Precompute all candidate edges (distance >= 3 in path, not already in path)
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))

    candidate_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if abs(u - v) >= 3 and (u, v) not in base_edges:
                candidate_edges.append((u, v))

    # Precompute for each candidate edge, which pairs (a,b) it helps reduce distance for
    # Actually, let's think differently. We need to find a small set of edges such that
    # the diameter is <= 4.

    # Let's try a greedy approach:
    # 1. Start with no added edges
    # 2. Find the pair with maximum distance
    # 3. Add an edge that reduces the most distances or specifically helps the max pair
    # 4. Repeat until diameter <= 4

    # But first, let's try a known good solution as a seed
    # The current program returns all edges with odd difference. Let's count them.
    # Actually, let's just do a greedy set cover-like approach.

    # State: set of added edges
    # We want to minimize |added_edges| such that diameter <= 4 and triangle-free and connected.

    # Greedy:
    added = []
    all_edges_set = set(P['edges'])

    for _ in range(100):  # Max 100 iterations
        if time.time() > deadline:
            break

        # Compute current diameter
        current_edges = P['edges'] + added
        adj = _build_adjacency(current_edges)

        if not _is_connected(adj):
            # Add an edge to connect components - but path is already connected, so this shouldn't happen
            # unless we remove something. We only add, so always connected.
            pass

        maxd = _max_diameter(adj)
        if maxd <= 4:
            # Check triangle-free
            if _is_triangle_free(adj):
                break

        # Find the pair with max distance
        max_pair = None
        max_dist = -1
        for src in range(n):
            dist = _bfs_distances(adj, src)
            for dst in range(src + 1, n):
                if dist[dst] > max_dist:
                    max_dist = dist[dst]
                    max_pair = (src, dst)

        if max_pair is None:
            break

        # Try to find a candidate edge that, when added, reduces the diameter
        # or at least reduces the distance of max_pair
        best_edge = None
        best_score = -1

        for (u, v) in candidate_edges:
            if (u, v) in all_edges_set:
                continue
            if (v, u) in all_edges_set:
                continue

            # Check triangle-freeness if we add this edge
            # u-v edge: check if any w is adjacent to both u and v in current graph
            tri_free = True
            for w in adj[u]:
                if w in adj[v]:
                    tri_free = False
                    break

            if not tri_free:
                continue

            # Tentatively add the edge and compute new diameter
            new_edges = current_edges + [(u, v)]
            new_adj = _build_adjacency(new_edges)
            new_d = _max_diameter(new_adj)

            if new_d < maxd:
                # Score by reduction in diameter
                score = maxd - new_d
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)

        if best_edge is None:
            # No single edge reduces diameter. Try to find an edge that makes progress
            # by reducing the max_pair distance
            for (u, v) in candidate_edges:
                if (u, v) in all_edges_set:
                    continue
                if (v, u) in all_edges_set:
                    continue

                tri_free = True
                for w in adj[u]:
                    if w in adj[v]:
                        tri_free = False
                        break

                if not tri_free:
                    continue

                new_edges = current_edges + [(u, v)]
                new_adj = _build_adjacency(new_edges)
                new_dist = _bfs_distances(new_adj, max_pair[0])
                new_d_pair = new_dist[max_pair[1]]

                if new_d_pair < max_dist:
                    score = max_dist - new_d_pair
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)

        if best_edge is None:
            # Stuck. Try a random edge that maintains triangle-freeness
            shuffled = candidate_edges[:]
            random.shuffle(shuffled)
            for (u, v) in shuffled:
                if (u, v) in all_edges_set:
                    continue
                if (v, u) in all_edges_set:
                    continue

                tri_free = True
                for w in adj[u]:
                    if w in adj[v]:
                        tri_free = False
                        break

                if tri_free:
                    best_edge = (u, v)
                    break

        if best_edge is None:
            break

        added.append(best_edge)
        all_edges_set.add((best_edge[0], best_edge[1]))
        all_edges_set.add((best_edge[1], best_edge[0]))

    # Verify the solution
    if not _validate_solution(added):
        # Fallback: try a simpler known structure
        # Add edges to create a hub-and-spoke pattern
        added = []
        # Let's try adding (0, 12), (12, 23), and some intermediate edges
        # Actually, let's just return the greedy result even if invalid, but that scores 0.
        # Let's try a hardcoded small solution.
        # Path has 24 vertices. We need diameter 4.
        # If we add (0,6), (6,12), (12,18), (18,23), (0,12), (6,18), (12,23)...
        # Let's try: (0, 12), (12, 23) - diameter is still large.
        # We need a better fallback. Let's just return the greedy result.
        pass

    return [[u, v] for u, v in added]


# EVOLVE-BLOCK-END
