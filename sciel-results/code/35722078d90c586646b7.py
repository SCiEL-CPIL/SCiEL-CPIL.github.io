import random
import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate the pinned edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    r = random.Random(614)
    r.shuffle(edges)

    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx

    total_edges = len(edges)

    # Precompute: for each vertex, list of (neighbor, edge_idx) sorted by edge_idx
    # This lets us quickly build adj_mask for any prefix K
    vertex_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        vertex_edges[u].append((v, idx))
        vertex_edges[v].append((u, idx))
    for i in range(n):
        vertex_edges[i].sort(key=lambda x: x[1])

    full_mask = (1 << n) - 1

    def build_adj_masks(K):
        """Build adjacency bitmasks for the graph formed by the first K edges."""
        adj = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_hamilton_cycle(adj):
        """Check if graph with adjacency bitmasks adj has a Hamilton cycle using DFS with pruning."""
        # Quick degree check
        for i in range(n):
            if adj[i].bit_count() < 2:
                return None

        # Connectivity check via BFS
        visited = 0
        stack = [0]
        visited |= 1
        while stack:
            u = stack.pop()
            m = adj[u] & ~visited
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                visited |= (1 << v)
                stack.append(v)
        if visited != full_mask:
            return None

        # Symmetry breaking: fix vertex 0 as start, and require vertex 1 to come before vertex n-1
        # Actually, for a cycle we can fix: path starts at 0, and we require the second vertex
        # to be less than the last vertex (breaking reflection symmetry)
        # But simpler: just start DFS from 0

        # Use iterative DFS with Warnsdorff + cut-set pruning
        # State: (current_vertex, visited_mask, path)
        # We'll use a stack-based approach with backtracking

        # Precompute for each vertex its neighbors sorted by their degree in the full graph
        # (static heuristic)
        static_deg = [adj[i].bit_count() for i in range(n)]
        # Sort neighbors of each vertex by static degree (Warnsdorff)
        neighbors_sorted = []
        for i in range(n):
            nbrs = []
            m = adj[i]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                nbrs.append(v)
            nbrs.sort(key=lambda v: static_deg[v])
            neighbors_sorted.append(nbrs)

        # DFS with pruning
        path = [0]
        visited_mask = 1

        def dfs(u, vmask, depth):
            if time.time() > deadline:
                return False

            if depth == n:
                # Check if u connects back to 0
                return (adj[u] & 1) != 0

            # Get unvisited neighbors of u
            candidates = adj[u] & (~vmask)
            if candidates == 0:
                return False

            # Pruning: check for cut vertices / disconnected unvisited subgraph
            # If removing u disconnects the unvisited vertices, fail
            # This is expensive, so do a lighter check:
            # For each unvisited vertex, check if it still has at least 2 neighbors
            # in (unvisited ∪ {u}) -- actually in the remaining graph
            # Lighter: check if any unvisited vertex has 0 neighbors in unvisited ∪ {u}
            # But that's O(n) per step which is fine for n=24

            # Build candidate list with dynamic degree
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Dynamic degree: number of unvisited neighbors of v (including u? no, u is being left)
                # After visiting v, v's unvisited neighbors are adj[v] & ~(vmask | (1<<v))
                # But we also need v to be able to connect to the rest
                # Dynamic degree for ordering: count of unvisited neighbors of v
                deg = (adj[v] & ~(vmask | (1 << v))).bit_count()
                # If this isn't the last vertex, v must have at least 1 unvisited neighbor
                # (or be the last and connect to 0)
                if depth + 1 == n:
                    # v is the last vertex, must connect to 0
                    if not (adj[v] & 1):
                        return False
                else:
                    # v must have at least 1 unvisited neighbor after we visit it
                    # (it needs to go somewhere)
                    # Actually it needs at least 1 neighbor among unvisited vertices
                    # that it can go to next
                    if deg == 0:
                        return False
                cand_list.append((deg, v))

            cand_list.sort()

            for deg, v in cand_list:
                path.append(v)
                if dfs(v, vmask | (1 << v), depth + 1):
                    return True
                path.pop()

            return False

        if dfs(0, 1, 1):
            return path
        return None

    # Strategy: binary search for minimum K where a Hamilton cycle exists
    # Lower bound: need at least n edges for a cycle, and each vertex needs degree >= 2
    # So we need at least 2n/2 = n edges... but actually for a 24-cycle we need 24 edges
    # But the edges must form a connected 2-regular graph on all vertices
    # Lower bound: n (24) edges
    # Upper bound: total_edges

    # First, find a cycle in the full graph as fallback
    adj_full = build_adj_masks(total_edges)
    fallback_cycle = has_hamilton_cycle(adj_full)
    if fallback_cycle is None:
        return list(range(n))

    best_cycle = fallback_cycle
    best_K = total_edges

    # Binary search
    low = n  # minimum edges needed for a cycle
    high = total_edges

    # We know full graph works
    while low < high:
        if time.time() > deadline - 5:
            break
        mid = (low + high) // 2
        adj_mid = build_adj_masks(mid)
        cycle = has_hamilton_cycle(adj_mid)
        if cycle:
            best_cycle = cycle
            best_K = mid
            high = mid
        else:
            low = mid + 1

    # Verify best_cycle
    if best_cycle and len(best_cycle) == n:
        return best_cycle

    return fallback_cycle

if __name__ == "__main__":
    pass
