import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n = 16
    adj = [[False] * n for _ in range(n)]

    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True

    # 1. Cycle 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)

    # 2. Shadows
    cycle_edges = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle_edges:
        su = 5 + u
        sv = 5 + v
        add_edge(su, v)
        add_edge(sv, u)

    # 3. Top 10 connected to all shadows
    for i in range(5):
        add_edge(10, 5 + i)

    # Get initial edge list
    initial_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                initial_edges.append([i, j])

    best_edges = initial_edges
    best_count = len(initial_edges)

    # Potential edges
    potential_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                potential_edges.append((i, j))

    def init_masks():
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
        return masks

    base_masks = init_masks()

    def creates_triangle_mask(u, v, masks):
        return (masks[u] & masks[v]) != 0

    def is_3_colorable(masks):
        """Check if graph with given masks is 3-colorable using backtracking."""
        color = [-1] * n
        uncolored = set(range(n))

        def backtrack():
            if not uncolored:
                return True
            # Pick vertex with most constrained color (fewest available colors)
            v = min(uncolored, key=lambda x: sum(1 for c in range(3)
                                                if not any((masks[x] >> y) & 1 and color[y] == c
                                                           for y in range(n) if y != x)))
            uncolored.remove(v)

            for c in range(3):
                # Check if color c is available for v
                ok = True
                for u in range(n):
                    if (masks[v] >> u) & 1 and color[u] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = -1

            uncolored.add(v)
            return False

        return backtrack()

    def greedy_complete(masks, current_edges, potential_edges, order='random'):
        """Greedy add edges in given order."""
        masks = masks[:]
        current_edges = current_edges[:]
        shuffled = potential_edges[:]

        if order == 'random':
            random.shuffle(shuffled)
        elif order == 'degree':
            # Sort by sum of degrees of endpoints (higher first)
            shuffled.sort(key=lambda e: bin(masks[e[0]]).count('1') + bin(masks[e[1]]).count('1'), reverse=True)
        elif order == 'constraint':
            # Sort by how many triangles would be created (fewer first = safer)
            def triangle_count(e):
                u, v = e
                return bin(masks[u] & masks[v]).count('1')
            shuffled.sort(key=triangle_count)

        for u, v in shuffled:
            if not creates_triangle_mask(u, v, masks):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_edges.append([u, v])

        return masks, current_edges

    def local_search(masks, current_edges, potential_edges, max_swaps=20):
        """Try edge swaps to improve."""
        masks = masks[:]
        current_edges = current_edges[:]
        edge_set = set(tuple(sorted(e)) for e in current_edges)
        improved = True
        swaps_made = 0

        while improved and swaps_made < max_swaps and time.time() < deadline:
            improved = False
            # Try removing one edge and adding two
            for u, v in list(current_edges):
                if time.time() > deadline - 5:
                    break
                # Remove edge (u,v)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                current_edges.remove([u, v])
                if [v, u] in current_edges:
                    current_edges.remove([v, u])

                # Try to add two edges that were previously blocked
                added = 0
                new_edges = []
                for a, b in potential_edges:
                    if added >= 2:
                        break
                    if a == u or a == v or b == u or b == v:
                        continue
                    if (a, b) in edge_set or (b, a) in edge_set:
                        continue
                    if not creates_triangle_mask(a, b, masks):
                        masks[a] |= (1 << b)
                        masks[b] |= (1 << a)
                        current_edges.append([a, b])
                        new_edges.append((a, b))
                        added += 1

                if added >= 1:
                    improved = True
                    swaps_made += 1
                    for a, b in new_edges:
                        edge_set.add(tuple(sorted((a, b))))
                else:
                    # Revert
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    current_edges.append([u, v])
                    for a, b in new_edges:
                        masks[a] &= ~(1 << b)
                        masks[b] &= ~(1 << a)
                        current_edges.remove([a, b])
                        if [b, a] in current_edges:
                            current_edges.remove([b, a])

        return masks, current_edges

    num_iterations = 5000
    iter_count = 0
    orders = ['random', 'degree', 'constraint']

    while time.time() < deadline and iter_count < num_iterations:
        iter_count += 1

        order = orders[iter_count % len(orders)]
        masks, current_edges = greedy_complete(base_masks, initial_edges, potential_edges, order)

        # Local search
        masks, current_edges = local_search(masks, current_edges, potential_edges, max_swaps=10)

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]

    return best_edges
