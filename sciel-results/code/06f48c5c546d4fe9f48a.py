"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s budget, 15s safety margin

    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))

    # Adjacency for base
    adj_base = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)

    # Candidate edges: all non-base, non-self edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))

    def is_triangle_free_add(adj, u, v):
        common = adj[u] & adj[v]
        return len(common) == 0

    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nbr in adj[node]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist

    # Best solution found so far
    best_edges = None
    best_count = float('inf')

    # --- Phase 0: Direct hub-and-spoke (known 11-edge solution) ---
    # Connect vertex 12 to all odd vertices
    hub_added = []
    for v in range(1, n, 2):
        if (12, v) not in base_edges:
            hub_added.append((12, v))

    adj_hub = [set(s) for s in adj_base]
    valid = True
    for u, v in hub_added:
        if not is_triangle_free_add(adj_hub, u, v):
            valid = False
            break
        adj_hub[u].add(v)
        adj_hub[v].add(u)

    if valid:
        diam = compute_diameter(adj_hub)
        if diam <= 4:
            best_count = len(hub_added)
            best_edges = [[u, v] for u, v in hub_added]

    # --- Phase 1: Try other single-hub configurations ---
    # For each even vertex as hub, connect to all odd vertices
    for hub in range(0, n, 2):
        if time.time() > deadline:
            break
        h_added = []
        adj_h = [set(s) for s in adj_base]
        h_valid = True
        for v in range(1, n, 2):
            if (hub, v) not in base_edges:
                if not is_triangle_free_add(adj_h, hub, v):
                    h_valid = False
                    break
                adj_h[hub].add(v)
                adj_h[v].add(hub)
                h_added.append((hub, v))
        if h_valid:
            diam = compute_diameter(adj_h)
            if diam <= 4 and len(h_added) < best_count:
                best_count = len(h_added)
                best_edges = [[u, v] for u, v in h_added]

    # --- Phase 2: Structured jump patterns (from inspiration 1) ---
    for k in range(2, n):
        if time.time() > deadline:
            break
        for offset in range(k):
            if time.time() > deadline:
                break
            edges_to_add = []
            adj_test = [set(s) for s in adj_base]
            valid = True
            for i in range(0, n, k):
                j = i + k
                if j < n and (i, j) not in base_edges:
                    if not is_triangle_free_add(adj_test, i, j):
                        valid = False
                        break
                    adj_test[i].add(j)
                    adj_test[j].add(i)
                    edges_to_add.append((i, j))
            if not valid or not edges_to_add:
                continue
            diam = compute_diameter(adj_test)
            if diam <= 4 and len(edges_to_add) < best_count:
                best_count = len(edges_to_add)
                best_edges = [[u, v] for u, v in edges_to_add]

    # --- Phase 3: Greedy refinement starting from best found ---
    if time.time() < deadline - 5:
        # Try to improve best_edges by removing edges
        if best_edges is not None:
            adj_g = [set(s) for s in adj_base]
            for u, v in best_edges:
                adj_g[u].add(v)
                adj_g[v].add(u)

            # Try removing each edge and check if still valid
            for idx in range(len(best_edges) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                u, v = best_edges[idx]
                # Try removing this edge
                adj_g[u].remove(v)
                adj_g[v].remove(u)
                # Check diameter
                diam = compute_diameter(adj_g)
                if diam <= 4:
                    best_edges.pop(idx)
                    best_count = len(best_edges)
                else:
                    # Put it back
                    adj_g[u].add(v)
                    adj_g[v].add(u)

    # --- Phase 4: Greedy addition from scratch (from inspiration 1) ---
    if time.time() < deadline - 10:
        adj_greedy = [set(s) for s in adj_base]
        added = []
        added_set = set()
        for iteration in range(15):
            if time.time() > deadline - 5:
                break
            curr_diam = compute_diameter(adj_greedy)
            if curr_diam <= 4:
                break
            best_edge = None
            best_new_diam = float('inf')
            for u, v in candidates:
                if (u, v) in added_set or (v, u) in added_set:
                    continue
                if not is_triangle_free_add(adj_greedy, u, v):
                    continue
                adj_greedy[u].add(v)
                adj_greedy[v].add(u)
                new_diam = compute_diameter(adj_greedy)
                adj_greedy[u].remove(v)
                adj_greedy[v].remove(u)
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            adj_greedy[u].add(v)
            adj_greedy[v].add(u)
            added.append((u, v))
            added_set.add((u, v))
            added_set.add((v, u))

        curr_diam = compute_diameter(adj_greedy)
        if curr_diam <= 4 and len(added) < best_count:
            best_count = len(added)
            best_edges = [[u, v] for u, v in added]

    # --- Phase 5: Two-hub patterns ---
    if time.time() < deadline - 5:
        # Try pairs of hubs (even vertices) each connecting to nearby odd vertices
        even_verts = list(range(0, n, 2))
        for h1 in even_verts:
            if time.time() > deadline - 3:
                break
            for h2 in even_verts:
                if h2 <= h1:
                    continue
                if time.time() > deadline - 3:
                    break
                adj_2h = [set(s) for s in adj_base]
                edges_2h = []
                valid_2h = True
                # h1 connects to odd vertices closer to h1
                for v in range(1, n, 2):
                    if (h1, v) not in base_edges:
                        if not is_triangle_free_add(adj_2h, h1, v):
                            continue
                        adj_2h[h1].add(v)
                        adj_2h[v].add(h1)
                        edges_2h.append((h1, v))
                # h2 connects to odd vertices closer to h2
                for v in range(1, n, 2):
                    if (h2, v) not in base_edges and (h2, v) not in edges_2h:
                        if not is_triangle_free_add(adj_2h, h2, v):
                            continue
                        adj_2h[h2].add(v)
                        adj_2h[v].add(h2)
                        edges_2h.append((h2, v))
                if edges_2h:
                    diam = compute_diameter(adj_2h)
                    if diam <= 4 and len(edges_2h) < best_count:
                        best_count = len(edges_2h)
                        best_edges = [[u, v] for u, v in edges_2h]

    # --- Final verification and fallback ---
    if best_edges is None:
        # Fallback: seed solution (all odd-difference edges)
        best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
    else:
        # Verify the best solution
        adj_final = [set(s) for s in adj_base]
        for u, v in best_edges:
            adj_final[u].add(v)
            adj_final[v].add(u)

        # Check triangle-free
        for u in range(n):
            for v in adj_final[u]:
                for w in adj_final[v]:
                    if w != u and w in adj_final[u]:
                        best_edges = None
                        break
                if best_edges is None:
                    break
            if best_edges is None:
                break

        if best_edges is not None:
            diam_final = compute_diameter(adj_final)
            if diam_final > 4:
                best_edges = None

        if best_edges is None:
            best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]

    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
