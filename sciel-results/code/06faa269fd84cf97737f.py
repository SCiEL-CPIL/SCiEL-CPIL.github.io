import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 31
    target_edges = 91

    # --- 1. Build the C4-free polarity graph of PG(2,5) with 90 edges ---
    # Use the standard construction: vertices are (x,y) in F_5^2 plus 6 points at infinity.
    # Edge condition: (x1-x2)^2 + (y1-y2)^2 = 0 in F_5 for affine points.
    # Points at infinity connect based on slope conditions.
    #
    # The known result: PG(2,5) polarity graph has 90 edges and 0 C4s.
    # We'll construct it explicitly.

    def build_polarity_graph():
        """Construct the C4-free polarity graph of PG(2,5) with exactly 90 edges."""
        edges = set()

        # Vertices 0-24: affine points (x, y) where x = idx % 5, y = idx // 5
        # Vertices 25-30: 6 points at infinity (representing slopes + one extra)
        # For PG(2,5), we have q^2 + q + 1 = 31 points.
        # The 6 points at infinity correspond to the 5 slopes m=0..4 plus the vertical direction.

        # Affine-Affine edges: (x1,y1) ~ (x2,y2) if (x1-x2)^2 + (y1-y2)^2 ≡ 0 (mod 5)
        for i in range(25):
            x1, y1 = i % 5, i // 5
            for j in range(i + 1, 25):
                x2, y2 = j % 5, j // 5
                dx = (x1 - x2) % 5
                dy = (y1 - y2) % 5
                if (dx * dx + dy * dy) % 5 == 0:
                    edges.add((i, j))

        # Affine-Infinity edges:
        # Vertex 25 (slope 0): connects to (x,y) if y ≡ 0, i.e., y=0
        # Vertex 26 (slope 1): connects to (x,y) if y ≡ x
        # Vertex 27 (slope 2): connects to (x,y) if y ≡ 2x
        # Vertex 28 (slope 3): connects to (x,y) if y ≡ 3x
        # Vertex 29 (slope 4): connects to (x,y) if y ≡ 4x
        # Vertex 30 (vertical/infinity): connects to (x,y) if x ≡ 0

        for i in range(25):
            x, y = i % 5, i // 5
            # Check each infinity vertex
            if y == 0:
                edges.add((i, 25))
            if y == x:
                edges.add((i, 26))
            if y == (2 * x) % 5:
                edges.add((i, 27))
            if y == (3 * x) % 5:
                edges.add((i, 28))
            if y == (4 * x) % 5:
                edges.add((i, 29))
            if x == 0:
                edges.add((i, 30))

        # Infinity-Infinity edges: In the polarity graph, no two infinity points are adjacent
        # (they represent parallel classes, which are mutually non-orthogonal in this construction)

        # Verify edge count
        assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"
        return edges

    init_edges = build_polarity_graph()

    # --- 2. Add the 91st edge: choose the non-edge with minimum common neighbors ---
    # This minimizes the number of new C4s created.

    def count_common_neighbors(edges_set, u, v):
        """Count common neighbors of u and v given the edge set."""
        neighbors_u = set()
        neighbors_v = set()
        for a, b in edges_set:
            if a == u:
                neighbors_u.add(b)
            elif b == u:
                neighbors_u.add(a)
            if a == v:
                neighbors_v.add(b)
            elif b == v:
                neighbors_v.add(a)
        return len(neighbors_u & neighbors_v)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(init_edges)
    non_edges = [e for e in all_pairs if e not in edge_set]

    # Find the non-edge with minimum common neighbors
    best_non_edge = None
    best_common = float('inf')
    for u, v in non_edges:
        cn = count_common_neighbors(edge_set, u, v)
        if cn < best_common:
            best_common = cn
            best_non_edge = (u, v)

    init_edges.add(best_non_edge)
    edge_set.add(best_non_edge)

    # --- 3. Local search with incremental C4 delta ---

    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in edge_set:
        adj[u, v] = 1
        adj[v, u] = 1

    def count_c4_full():
        """Count all C4s in the graph."""
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = np.dot(adj[u], adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    def delta_c4_remove(u, v):
        """Compute change in C4 count when removing edge (u,v). Negative means C4s decrease."""
        # Removing (u,v) removes C4s that use this edge.
        # A C4 using edge (u,v) has form u-w-x-v-u or u-x-w-v-u.
        # The number of C4s containing edge (u,v) is:
        # For each common neighbor w of u and v (w != u,v), the number of C4s
        # through (u,v) and w is the number of common neighbors of w and... 
        # Actually, C4s containing edge (u,v): pick two vertices w, x such that
        # w~u, w~v, x~u, x~v, w!=x, and the path u-w-v-x-u forms a C4.
        # Wait, that's not quite right. Let me think more carefully.
        #
        # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} with edges ab, bc, cd, da.
        # Edge (u,v) is in a C4 if there exist w, x such that {u,v,w,x} form a C4
        # and (u,v) is one of the edges.
        # The C4s containing (u,v) are: u-w-v-x-u where w~u, w~v, x~u, x~v, w!=x, w!=u,v, x!=u,v.
        # But we also need w~v and x~u... wait, in u-w-v-x-u:
        #   u~w, w~v, v~x, x~u
        # So w is a common neighbor of u and v, and x is a common neighbor of u and v.
        # But we also need w!=x, and the cycle to be valid (no chords for counting as C4 subgraph).
        # Actually for counting C4 subgraphs, we just need 4 distinct vertices forming a cycle.
        # The number of C4s containing edge (u,v) = C(k, 2) where k = number of common neighbors of u and v?
        # No! That would count C4s where both w and x are common neighbors, but the C4 is u-w-v-x-u.
        # For this to be a C4, we need u~w, w~v, v~x, x~u. Since w is common neighbor of u,v: u~w and w~v. ✓
        # Since x is common neighbor of u,v: u~x and x~v. ✓
        # So yes, each pair of distinct common neighbors {w,x} gives a C4 u-w-v-x-u.
        # But wait, we need to make sure w and x are not adjacent to each other? No, for C4 subgraph counting,
        # we count the subgraph, so even if w~x, the 4-cycle u-w-v-x-u still exists as a subgraph.
        # Actually no - a C4 subgraph requires exactly 4 vertices and 4 edges forming a cycle.
        # If w~x also exists, then {u,v,w,x} might have more edges, but the 4-cycle u-w-v-x-u is still a C4 subgraph.
        # So the count is C(k,2) where k = common_neighbors(u,v).
        
        common = np.dot(adj[u], adj[v])
        # But we need to exclude u and v themselves from common neighbors
        # adj[u,v] = 1, so v is a "common neighbor" of u and u... no.
        # Common neighbors of u and v are vertices w such that adj[u,w]=1 and adj[v,w]=1, w != u, w != v.
        # np.dot(adj[u], adj[v]) counts all w where adj[u,w] and adj[v,w] are both 1.
        # This includes w=u (adj[u,u]=0, so no) and w=v (adj[u,v]=1, adj[v,v]=0, so no).
        # So it correctly counts common neighbors excluding u and v.
        
        return -common * (common - 1) // 2

    def delta_c4_add(u, v):
        """Compute change in C4 count when adding edge (u,v) to a graph where (u,v) is not an edge."""
        # Adding (u,v) creates new C4s that use this edge.
        # Same logic: C(k,2) where k = common neighbors of u and v in the current graph.
        common = np.dot(adj[u], adj[v])
        return common * (common - 1) // 2

    current_score = count_c4_full()
    best_score = current_score
    best_edges = set(edge_set)

    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        current_edges = set(edge_set)
        non_edges = [e for e in all_edges_list if e not in current_edges]
        random.shuffle(non_edges)

        for e_rm in list(current_edges):
            if (time.time() - start_time) > deadline - 3:
                break

            u_rm, v_rm = e_rm

            # Calculate delta for removing this edge
            delta_rm = delta_c4_remove(u_rm, v_rm)

            best_delta = 0
            best_e_add = None

            # Try all non-edges (n=31, so at most ~479 non-edges)
            for e_add in non_edges:
                u_add, v_add = e_add
                if (u_add == u_rm and v_add == v_rm) or (u_add == v_rm and v_add == u_rm):
                    continue
                # Skip if adding creates a duplicate
                if e_add in current_edges:
                    continue

                delta_add = delta_c4_add(u_add, v_add)
                delta = delta_rm + delta_add

                if delta < best_delta:
                    best_delta = delta
                    best_e_add = e_add

            if best_e_add is not None and best_delta < 0:
                # Apply the swap
                u_add, v_add = best_e_add
                edge_set.discard((u_rm, v_rm))
                edge_set.add((u_add, v_add))
                adj[u_rm, v_rm] = 0
                adj[v_rm, u_rm] = 0
                adj[u_add, v_add] = 1
                adj[v_add, u_add] = 1

                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edge_set)
                improved = True
                break  # Restart with updated graph

    # Return the best result
    return [[u, v] for u, v in best_edges]
