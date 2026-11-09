import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    n = 16

    # Precompute edge list for K16
    all_edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(all_edges)
            all_edges.append((i, j))
    m_edges = len(all_edges)  # 120

    # Line graph adjacency: two edges are adjacent if they share a vertex
    # (i.e., adding both to a triangle-free graph would create a triangle)
    # Actually, two edges (a,b) and (c,d) form a triangle iff they share exactly
    # one vertex AND that vertex's other neighbor is also connected.
    # More precisely: edges (u,v) and (w,x) conflict (can't both be in a
    # triangle-free graph) iff they share a vertex, say u=w, and v-x is also an edge.
    # But for the *line graph* approach: two edges conflict if they share a vertex
    # AND the two other endpoints are connected. This is the "triangle" condition.
    #
    # Actually, the standard result: a set of edges is triangle-free iff no three
    # edges form a triangle. In the line graph L(K16), a clique corresponds to a
    # set of edges where every pair shares a vertex. But that's not quite right.
    #
    # Let me reconsider. The triangle-free constraint: no three vertices u,v,w
    # with all three edges (u,v),(v,w),(w,u) present.
    #
    # For a branch-and-bound over edges, the conflict between two edges e1=(a,b)
    # and e2=(c,d): they conflict if adding both to the current set would create
    # a triangle. This happens if:
    #   - They share a vertex, say a=c, and (b,d) is already in the set, OR
    #   - They share a vertex, say a=d, and (b,c) is already in the set, etc.
    #
    # This is state-dependent, so a pure line graph clique won't directly work.
    # Instead, I'll use a backtracking search over edges with bitset-based
    # triangle-free checking, which is the "line graph" idea adapted.

    # Bitmask for 3-colorability check (fast, from current program)
    def is_3_colorable(adj_mask):
        """Fast 3-colorability check using backtracking with MRV heuristic."""
        color = [-1] * n
        colored = [False] * n
        colored_count = 0

        def backtrack():
            nonlocal colored_count
            if colored_count == n:
                return True

            # MRV: pick uncolored vertex with most colored neighbors
            best_v = -1
            best_count = -1
            for v in range(n):
                if not colored[v]:
                    cnt = 0
                    mask = adj_mask[v]
                    for u in range(n):
                        if colored[u] and (mask >> u) & 1:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v

            if best_v == -1:
                return True

            v = best_v
            used = 0
            mask = adj_mask[v]
            for u in range(n):
                if colored[u] and (mask >> u) & 1:
                    used |= (1 << color[u])

            for c in range(3):
                if not (used >> c) & 1:
                    color[v] = c
                    colored[v] = True
                    colored_count += 1
                    if backtrack():
                        return True
                    colored[v] = False
                    colored_count -= 1
            return False

        return backtrack()

    def edge_count(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2

    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def check_triangle_free_add(adj_mask, u, v):
        """Check if adding edge (u,v) to adj_mask keeps it triangle-free."""
        # u and v must not share a common neighbor
        common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
        return common == 0

    # Build initial Grötzsch graph (11 vertices, triangle-free, 4-chromatic)
    def build_groetzsch():
        adj = [0] * n
        # C5 cycle: 0-1-2-3-4-0
        for i in range(5):
            add_edge(adj, i, (i + 1) % 5)
        # Mycielski: vertices 5-9 connect to neighbors of 0-4
        for i in range(5):
            add_edge(adj, 5 + i, (i + 1) % 5)
            add_edge(adj, 5 + i, (i - 1) % 5)
        # Apex vertex 10 connects to 5-9
        for i in range(5):
            add_edge(adj, 10, 5 + i)
        return adj

    # Initialize best
    best_adj = build_groetzsch()
    best_count = edge_count(best_adj)

    # Phase 1: Branch-and-bound search over edges (line graph inspired)
    # We search for the maximum-size triangle-free edge set that is not 3-colorable.
    # Use a recursive search that adds edges one at a time, with pruning.

    # Precompute: for each vertex, bitmask of all other vertices
    all_mask = (1 << n) - 1

    def branch_and_bound(adj_mask, edge_idx, current_count, time_limit):
        """Branch and bound: try to add more edges from edge_idx onwards."""
        nonlocal best_adj, best_count

        if time.time() > time_limit:
            return

        # Upper bound: current edges + remaining edges that could potentially be added
        # (this is loose, but the triangle-free check will prune)
        if current_count + (m_edges - edge_idx) <= best_count:
            return

        # Try to add edge at edge_idx
        if edge_idx < m_edges:
            u, v = all_edges[edge_idx]
            if not (adj_mask[u] >> v) & 1:  # Edge not already present
                if check_triangle_free_add(adj_mask, u, v):
                    # Tentatively add
                    new_adj = list(adj_mask)
                    add_edge(new_adj, u, v)
                    new_count = current_count + 1

                    # Check if still not 3-colorable
                    if not is_3_colorable(new_adj):
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = list(new_adj)
                        # Recurse: try to add more edges
                        branch_and_bound(new_adj, edge_idx + 1, new_count, time_limit)

            # Also try NOT adding this edge
            branch_and_bound(adj_mask, edge_idx + 1, current_count, time_limit)

    # The full branch-and-bound over 120 edges is too large. Instead, use a
    # smarter approach: greedy + local search, but with the line graph insight
    # to prioritize which edges to add.

    # Phase 1: Greedy edge addition with triangle-free and 3-colorability checks
    def greedy_add(adj_mask, time_limit):
        """Greedy: repeatedly add edges that preserve validity, in priority order."""
        # Priority: edges that connect high-degree vertices first (more likely to
        # create triangles, so we need to be careful, but also more "valuable")
        deg = [bin(adj_mask[i]).count('1') for i in range(n)]

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Sort candidate edges by priority (sum of endpoint degrees, descending)
            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] >> j) & 1:
                        if check_triangle_free_add(adj_mask, i, j):
                            candidates.append((deg[i] + deg[j], i, j))
            candidates.sort(key=lambda x: -x[0])

            for _, u, v in candidates:
                if time.time() > time_limit:
                    break
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if not is_3_colorable(new_adj):
                    add_edge(adj_mask, u, v)
                    improved = True
                    deg[u] += 1
                    deg[v] += 1
                    break  # Restart with updated degrees

        return adj_mask

    # Run greedy addition from Grötzsch
    adj_mask = greedy_add(best_adj, deadline - 40)
    current_count = edge_count(adj_mask)
    if current_count > best_count:
        best_count = current_count
        best_adj = list(adj_mask)

    # Phase 2: Local search with edge swaps
    def local_search(adj_mask, time_limit, iterations=500):
        """Local search: remove one edge, try to add another to improve."""
        nonlocal best_adj, best_count

        current_count = edge_count(adj_mask)

        for iteration in range(iterations):
            if time.time() > time_limit:
                break

            # Get existing edges
            existing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        existing.append((i, j))

            if not existing:
                break

            # Randomly pick an edge to remove
            ui, vi = random.choice(existing)
            remove_edge(adj_mask, ui, vi)

            # Try to add different edges
            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] >> j) & 1 and (i, j) != (ui, vi):
                        if check_triangle_free_add(adj_mask, i, j):
                            candidates.append((i, j))
            random.shuffle(candidates)

            added_any = False
            for u, v in candidates:
                if time.time() > time_limit:
                    break
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if not is_3_colorable(new_adj):
                    add_edge(adj_mask, u, v)
                    added_any = True
                    break

            if not added_any:
                # Try more candidates without the triangle-free pre-check
                for u, v in candidates:
                    if time.time() > time_limit:
                        break
                    if not check_triangle_free_add(adj_mask, u, v):
                        continue
                    new_adj = list(adj_mask)
                    add_edge(new_adj, u, v)
                    if not is_3_colorable(new_adj):
                        add_edge(adj_mask, u, v)
                        added_any = True
                        break

            current_count = edge_count(adj_mask)
            if current_count > best_count:
                best_count = current_count
                best_adj = list(adj_mask)

        return adj_mask

    adj_mask = local_search(adj_mask, deadline - 20, iterations=300)

    # Phase 3: Simulated annealing style
    current_adj = list(best_adj)
    current_count = best_count

    for iteration in range(2000):
        if time.time() > deadline - 5:
            break

        move_type = random.random()

        if move_type < 0.3:
            # Try to add an edge
            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (current_adj[i] >> j) & 1:
                        if check_triangle_free_add(current_adj, i, j):
                            candidates.append((i, j))
            if not candidates:
                continue
            u, v = random.choice(candidates)
            new_adj = list(current_adj)
            add_edge(new_adj, u, v)
            if not is_3_colorable(new_adj):
                new_count = edge_count(new_adj)
                if new_count > current_count or random.random() < 0.1:
                    current_adj = new_adj
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = list(current_adj)

        elif move_type < 0.6:
            # Try to remove an edge
            existing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        existing.append((i, j))
            if not existing:
                continue
            u, v = random.choice(existing)
            new_adj = list(current_adj)
            remove_edge(new_adj, u, v)
            if not is_3_colorable(new_adj):
                new_count = edge_count(new_adj)
                if new_count >= current_count - 2 and random.random() < 0.2:
                    current_adj = new_adj
                    current_count = new_count

        else:
            # Swap: remove one edge, try to add another
            existing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        existing.append((i, j))
            if not existing:
                continue
            u1, v1 = random.choice(existing)
            new_adj = list(current_adj)
            remove_edge(new_adj, u1, v1)

            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (new_adj[i] >> j) & 1:
                        if check_triangle_free_add(new_adj, i, j):
                            candidates.append((i, j))
            random.shuffle(candidates)

            found = False
            for u2, v2 in candidates[:20]:
                if time.time() > deadline - 5:
                    break
                trial_adj = list(new_adj)
                add_edge(trial_adj, u2, v2)
                if not is_3_colorable(trial_adj):
                    new_count = edge_count(trial_adj)
                    if new_count > current_count or (new_count >= current_count - 1 and random.random() < 0.3):
                        current_adj = trial_adj
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = list(current_adj)
                        found = True
                        break

    # Final greedy cleanup
    final_adj = list(best_adj)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (final_adj[i] >> j) & 1:
                if check_triangle_free_add(final_adj, i, j):
                    candidates.append((i, j))
    random.shuffle(candidates)
    for u, v in candidates:
        if time.time() > deadline - 1:
            break
        new_adj = list(final_adj)
        add_edge(new_adj, u, v)
        if not is_3_colorable(new_adj):
            add_edge(final_adj, u, v)

    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj

    # Return result
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])

    return result
