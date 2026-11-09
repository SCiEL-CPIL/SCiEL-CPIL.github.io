import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 16

    # Precompute all edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    m = len(all_edges)  # 120

    # For each edge, precompute which other edges it conflicts with (forms triangle)
    # Edge (a,b) conflicts with (c,d) if {a,b,c,d} has exactly 3 distinct vertices
    # i.e., they share exactly one vertex
    # If we include both (a,b) and (c,d) where c=a, then we must NOT include (b,d)
    # So the conflict is: including (a,b) and (a,d) forbids (b,d)

    # Build conflict structure
    # For each edge e, list of edges that form a triangle with e
    # Actually, for triangle-freeness: if we have edges (u,v) and (v,w), we cannot have (u,w)
    # So for each pair of edges sharing a vertex, the third edge is forbidden

    # Let's use a different representation:
    # For each vertex v, let N(v) be the set of neighbors of v
    # Triangle-free means: for each v, N(v) is an independent set
    # i.e., no two neighbors of v are connected by an edge

    # Use bitmask for adjacency
    # adj[v] is a bitmask of neighbors of v

    def make_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def is_triangle_free(adj):
        for i in range(n):
            # For each pair of neighbors of i, check they're not connected
            ni = adj[i]
            # Get list of neighbors
            nbrs = []
            for j in range(n):
                if ni & (1 << j):
                    nbrs.append(j)
            for idx_a in range(len(nbrs)):
                for idx_b in range(idx_a + 1, len(nbrs)):
                    a, b = nbrs[idx_a], nbrs[idx_b]
                    if adj[a] & (1 << b):
                        return False
        return True

    def is_3_colorable(adj):
        """Check if graph is 3-colorable using backtracking."""
        colors = [-1] * n

        def backtrack(v):
            if v == n:
                return True
            # Try colors 0, 1, 2
            used = set()
            for nb in range(n):
                if adj[v] & (1 << nb) and colors[nb] != -1:
                    used.add(colors[nb])
            for c in range(3):
                if c not in used:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False

        return backtrack(0)

    def is_not_3_colorable(adj):
        return not is_3_colorable(adj)

    # Start with a known 4-chromatic triangle-free graph
    # Use Mycielski of C5 (Grötzsch graph) on vertices 0-10
    def make_groetzsch():
        adj = [0] * n
        # C5: 0-1-2-3-4-0
        c5_edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        for u, v in c5_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Mycielski: copies 5-9, top vertex 10
        for i in range(5):
            neighbors = [(i+1) % 5, (i-1) % 5]
            for nb in neighbors:
                adj[5+i] |= (1 << nb)
                adj[nb] |= (1 << (5+i))
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
        return adj

    best_adj = make_groetzsch()
    best_count = sum(bin(adj).count('1') for adj in best_adj) // 2

    # Now try to add edges to increase density while maintaining triangle-free and 4-chromatic
    # Use local search: try adding edges, check constraints

    # Candidate edges that can be added
    def get_addable_edges(adj):
        addable = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    # Check if adding this edge creates a triangle
                    # Triangle exists if there's w such that (u,w) and (v,w) are both edges
                    common = adj[u] & adj[v]
                    if common == 0:
                        addable.append((u, v))
        return addable

    # Local search with restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break

        adj = list(best_adj)

        # Try to add edges greedily
        for _ in range(100):
            if time.time() > deadline - 5:
                break

            addable = get_addable_edges(adj)
            if not addable:
                break

            # Try each addable edge
            improved = False
            random.shuffle(addable)
            for u, v in addable:
                if time.time() > deadline - 5:
                    break

                # Try adding edge (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

                # Check if still 4-chromatic
                if is_not_3_colorable(adj):
                    # Keep the edge
                    best_count = sum(bin(a).count('1') for a in adj) // 2
                    if best_count > sum(bin(a).count('1') for a in best_adj) // 2:
                        best_adj = list(adj)
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

            if not improved:
                break

        # Local search: try swapping edges
        for _ in range(50):
            if time.time() > deadline - 5:
                break

            # Get current edges
            current_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        current_edges.append((u, v))

            if not current_edges:
                break

            # Try removing one edge and adding another
            random.shuffle(current_edges)
            for u, v in current_edges:
                if time.time() > deadline - 5:
                    break

                # Remove edge (u,v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

                # Try adding a different edge
                addable = get_addable_edges(adj)
                random.shuffle(addable)
                for u2, v2 in addable:
                    if (u2, v2) == (u, v) or (u2, v2) == (v, u):
                        continue
                    if (u2, v2) == (v, u):
                        continue

                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)

                    if is_not_3_colorable(adj):
                        best_count = sum(bin(a).count('1') for a in adj) // 2
                        if best_count > sum(bin(a).count('1') for a in best_adj) // 2:
                            best_adj = list(adj)
                        break
                    else:
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)

                # Restore removed edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Final verification
    adj = best_adj
    if not is_triangle_free(adj):
        # Fallback to Grötzsch
        adj = make_groetzsch()

    if is_3_colorable(adj):
        # Shouldn't happen, but fallback
        adj = make_groetzsch()

    edges = make_edge_list(adj)
    return edges
