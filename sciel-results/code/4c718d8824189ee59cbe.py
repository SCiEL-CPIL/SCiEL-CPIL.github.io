import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin

    N = 16
    ALL = (1 << N) - 1

    # Precompute all possible edges
    edges_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
    M = len(edges_list)

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(N)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # Check triangle-free: common neighbors of u and v must be empty
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        return common == 0

    def is_3_colorable(adj):
        """Fast 3-colorability check using bitmask backtracking."""
        # Order vertices by degree descending for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(N)]
        order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
        
        # Map vertex to position in order
        pos = {v: i for i, v in enumerate(order)}
        
        # Precompute neighbor masks relative to the order? 
        # Actually, just using the adj masks is fine.
        
        color = [-1] * N
        cc = [0, 0, 0]  # Color class bitmasks

        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            
            # Determine forbidden colors based on neighbors
            # A neighbor w is forbidden if color[w] is set.
            # We can check cc masks directly.
            nb_mask = adj[v]
            
            forb = 0
            if nb_mask & cc[0]:
                forb |= 1
            if nb_mask & cc[1]:
                forb |= 2
            if nb_mask & cc[2]:
                forb |= 4
                
            available = 7 ^ forb
            
            # MRV heuristic: pick the vertex with fewest available colors first?
            # The order is fixed by degree. To implement dynamic MRV, we would need to 
            # pick the next vertex dynamically. For N=16, fixed order is often fast enough.
            
            # Try available colors
            # Heuristic: try colors in order 0, 1, 2
            for c in range(3):
                if available & (1 << c):
                    color[v] = c
                    cc[c] |= (1 << v)
                    if backtrack(idx + 1):
                        return True
                    cc[c] &= ~(1 << v)
                    color[v] = -1
            return False

        return backtrack(0)

    def build_groetzsch():
        adj = [0] * N
        # Grötzsch graph on vertices 0-10
        # 5-cycle: 0-1-2-3-4-0
        for i in range(5):
            add_edge(adj, i, (i + 1) % 5)
        # Chords to vertex 5,6,7,8,9
        add_edge(adj, 5, 1); add_edge(adj, 5, 3)
        add_edge(adj, 6, 2); add_edge(adj, 6, 4)
        add_edge(adj, 7, 3); add_edge(adj, 7, 0)
        add_edge(adj, 8, 4); add_edge(adj, 8, 1)
        add_edge(adj, 9, 0); add_edge(adj, 9, 2)
        # Vertex 10 connects to 5,6,7,8,9
        for v in range(5, 10):
            add_edge(adj, 10, v)
        return adj

    # Initial graph: Grötzsch + 5 isolated vertices
    best_adj = build_groetzsch()
    best_count = count_edges(best_adj)
    
    # Verify initial
    if is_3_colorable(best_adj):
        # This should not happen
        pass

    def greedy_add(adj):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for (u, v) in edges_list:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if can_add_edge(adj, u, v):
                    new_adj = adj[:]
                    add_edge(new_adj, u, v)
                    if not is_3_colorable(new_adj):
                        adj = new_adj
                        changed = True
                        break
        return adj

    best_adj = greedy_add(best_adj)
    best_count = count_edges(best_adj)

    # Local Search
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break

        current = best_adj[:]
        cur_count = best_count

        # Perturb: remove some edges
        if restart > 0:
            present_edges = [(u, v) for (u, v) in edges_list if (current[u] >> v) & 1]
            if not present_edges:
                continue
            n_remove = min(1 + restart % 5, len(present_edges))
            for _ in range(n_remove):
                if not present_edges:
                    break
                idx = rng.randrange(len(present_edges))
                u, v = present_edges[idx]
                remove_edge(current, u, v)
                present_edges.pop(idx)
                cur_count -= 1

        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Add edges
            for (u, v) in edges_list:
                if time.time() > deadline:
                    break
                if (current[u] >> v) & 1:
                    continue
                if can_add_edge(current, u, v):
                    new_adj = current[:]
                    add_edge(new_adj, u, v)
                    if not is_3_colorable(new_adj):
                        current = new_adj
                        cur_count += 1
                        improved = True
                        break

            if time.time() > deadline:
                break

            # Phase 2: Swaps (Remove 1, Add 2)
            if not improved:
                present_edges = [(u, v) for (u, v) in edges_list if (current[u] >> v) & 1]
                rng.shuffle(present_edges)
                # Limit swaps to avoid long loops
                for (ru, rv) in present_edges[:10]:
                    if time.time() > deadline:
                        break
                    test = current[:]
                    remove_edge(test, ru, rv)
                    added = 0
                    for (a, b) in edges_list:
                        if time.time() > deadline:
                            break
                        if (a == ru and b == rv) or (a == rv and b == ru):
                            continue
                        if (test[a] >> b) & 1:
                            continue
                        if not can_add_edge(test, a, b):
                            continue
                        test2 = test[:]
                        add_edge(test2, a, b)
                        if not is_3_colorable(test2):
                            test = test2
                            added += 1
                            if added >= 2:
                                break
                    if added >= 2:
                        current = test
                        cur_count += added - 1
                        improved = True
                        break

        if cur_count > best_count:
            best_adj = current
            best_count = cur_count

    # Final greedy pass
    best_adj = greedy_add(best_adj)
    best_count = count_edges(best_adj)

    # Verification
    if is_3_colorable(best_adj):
        # Fallback to known valid graph
        best_adj = build_groetzsch()
        best_count = count_edges(best_adj)

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])

    return edges
