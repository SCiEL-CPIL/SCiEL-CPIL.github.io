"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
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


def _decompose_graph(edges, n, time_limit):
    """
    Decompose a graph into cycles and edges.
    Returns a list of pieces.
    """
    if not edges:
        return []

    # Build adjacency
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    pieces = []
    start_time = time.time()

    # First pass: find cycles using DFS-based approach
    # Use a priority queue of vertices sorted by degree
    while True:
        if time.time() - start_time > time_limit * 0.8:
            break

        # Check remaining edges
        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break

        # Find a cycle
        cycle = None

        # Try vertices with highest degree first
        candidates = [v for v in range(n) if deg[v] >= 2]
        candidates.sort(key=lambda v: -deg[v])

        for start in candidates:
            if deg[start] < 2:
                continue

            # DFS to find a cycle back to start
            # Use iterative DFS with path tracking
            stack = [(start, [start])]
            visited = {start}

            while stack:
                node, path = stack.pop()
                if node == start and len(path) >= 3:
                    cycle = path
                    break
                if cycle:
                    break

                # Try neighbors
                neighbors = list(adj[node])
                neighbors.sort(key=lambda x: -deg[x])

                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        cycle = path
                        break
                    if cycle:
                        break
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))

            if cycle:
                break

        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No cycle found, take an edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1

    # Add remaining edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    return pieces


def _merge_into_cycles(pieces, edges_set, n, time_limit):
    """
    Try to merge edge pieces into cycles.
    """
    start_time = time.time()

    # Build a map from edge to piece index
    edge_to_piece = {}
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            edge = (min(u, v), max(u, v))
            edge_to_piece[edge] = i

    # Find connected components of edge pieces
    # Two edge pieces are connected if they share a vertex
    # Build a graph where nodes are edge pieces, edges connect pieces that share a vertex
    piece_vertices = {}  # piece_index -> set of vertices
    vertex_to_pieces = defaultdict(set)  # vertex -> set of piece indices

    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            piece_vertices[i] = {u, v}
            vertex_to_pieces[u].add(i)
            vertex_to_pieces[v].add(i)

    # Try to find cycles in the edge-piece graph
    # Use DFS to find cycles
    used = set()
    new_pieces = []

    # First, keep all non-edge pieces
    for i, piece in enumerate(pieces):
        if len(piece) > 2:
            new_pieces.append(piece)

    # Then process edge pieces
    # Try to find triangles (3 edges forming a cycle)
    # For each vertex, look at all pairs of edge pieces incident to it
    # If the other endpoints are connected by an edge piece, we have a triangle

    # Build adjacency: for each vertex, list of (piece_idx, other_vertex)
    vert_adj = defaultdict(list)
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            vert_adj[u].append((i, v))
            vert_adj[v].append((i, u))

    # Try to find 3-cycles in the edge-piece graph
    # For each edge piece, check if its endpoints can form a triangle
    # This is O(E^2) worst case but should be fine for small graphs

    # Actually, let's try a different approach:
    # Repeatedly look for sets of edge pieces that form a cycle
    # Start with triangles (most common)

    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False

        # Find all triangles in the edge-piece graph
        # A triangle is 3 edge pieces (a,b), (b,c), (c,a) where a,b,c are distinct vertices
        # We can find these by: for each vertex v, for each pair of edge pieces (v,u) and (v,w),
        # check if there's an edge piece (u,w)

        used_edges = set()  # piece indices already used in a cycle

        # Rebuild current edge pieces
        current_edge_pieces = {}
        for i, piece in enumerate(pieces):
            if len(piece) == 2 and i not in used:
                u, v = piece
                current_edge_pieces[(min(u, v), max(u, v))] = i

        # Build vert_adj for current edge pieces only
        vert_adj_current = defaultdict(list)
        for (u, v), i in current_edge_pieces.items():
            vert_adj_current[u].append((i, v))
            vert_adj_current[v].append((i, u))

        # Find triangles
        for v in list(vert_adj_current.keys()):
            edges_at_v = vert_adj_current[v]
            if len(edges_at_v) < 2:
                continue
            for i in range(len(edges_at_v)):
                e1, u1 = edges_at_v[i]
                for j in range(i + 1, len(edges_at_v)):
                    e2, u2 = edges_at_v[j]
                    # Check if (u1, u2) is an edge piece
                    edge_u1_u2 = (min(u1, u2), max(u1, u2))
                    if edge_u1_u2 in current_edge_pieces:
                        e3 = current_edge_pieces[edge_u1_u2]
                        if e1 not in used_edges and e2 not in used_edges and e3 not in used_edges:
                            # Found a triangle: v, u1, u2
                            cycle = [v, u1, u2]
                            # Verify all edges exist
                            if (min(v, u1), max(v, u1)) in current_edge_pieces and \
                               (min(v, u2), max(v, u2)) in current_edge_pieces and \
                               (min(u1, u2), max(u1, u2)) in current_edge_pieces:
                                # Replace the three edge pieces with the cycle
                                used_edges.add(e1)
                                used_edges.add(e2)
                                used_edges.add(e3)
                                used.add(e1)
                                used.add(e2)
                                used.add(e3)
                                new_pieces.append(cycle)
                                changed = True
                                break
                if changed:
                    break
            if changed:
                break

        if changed:
            # Remove used edge pieces from consideration
            # We'll rebuild at the next iteration
            pass

    if not changed:
        # No triangles found, just return original pieces
        return pieces

    # Rebuild the piece list
    final_pieces = []
    for i, piece in enumerate(pieces):
        if len(piece) > 2:
            final_pieces.append(piece)
        elif len(piece) == 2 and i not in used:
            final_pieces.append(piece)
    final_pieces.extend(new_pieces)

    return final_pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Allocate time for this graph
        graph_time = remaining * 0.85  # Leave 15% buffer

        # Try multiple decompositions with different strategies
        best_pieces = None
        best_count = float('inf')

        # Strategy 1: Standard DFS-based cycle finding
        pieces1 = _decompose_graph(edges, n, graph_time * 0.4)
        count1 = len(pieces1)

        # Try to merge edge pieces into cycles
        pieces1_merged = _merge_into_cycles(pieces1, edges_set, n, graph_time * 0.1)
        count1_merged = len(pieces1_merged)

        if count1_merged < best_count:
            best_count = count1_merged
            best_pieces = pieces1_merged

        if count1 < best_count:
            best_count = count1
            best_pieces = pieces1

        # Strategy 2: Try with different vertex orderings
        if time.time() - start_time < _DEADLINE_SECONDS - 10:
            # Shuffle and try again
            pieces2 = _decompose_graph(edges, n, graph_time * 0.3)
            count2 = len(pieces2)
            pieces2_merged = _merge_into_cycles(pieces2, edges_set, n, graph_time * 0.1)
            count2_merged = len(pieces2_merged)

            if count2_merged < best_count:
                best_count = count2_merged
                best_pieces = pieces2_merged

            if count2 < best_count:
                best_count = count2
                best_pieces = pieces2

        results.append(best_pieces)

    return results
