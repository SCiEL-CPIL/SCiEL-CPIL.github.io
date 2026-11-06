import time

def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin before 120s kill

    n = 16
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m_all = len(all_edges)

    # Build adjacency as bitmasks for fast triangle checks
    # adj[u] is an int bitmask of neighbors of u
    adj = [0] * n

    # Start with complete graph, then remove edges that create triangles
    # to get a maximal triangle-free graph (greedy Turán-like)
    # Actually, start empty and add edges greedily to maximize edges
    # while keeping triangle-free.

    # Greedy: process edges in a fixed order, add if no triangle formed.
    # To get a good (dense) triangle-free graph, use a Turán-partition ordering:
    # vertices 0..5 part A, 6..10 part B, 11..15 part C.
    # Order edges: within-part first (won't be added), then cross-part.
    # But simpler: just iterate all edges; add edge (u,v) if no common neighbor.

    # Use a randomized restart approach to find a dense triangle-free graph,
    # then remove edges to break 3-colorability.

    def is_triangle_free(adj, u, v):
        # Check if u and v share a common neighbor
        return (adj[u] & adj[v]) == 0

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Greedy triangle-free construction
    def greedy_triangle_free(order):
        adj = [0] * n
        edges = []
        for (u, v) in order:
            if is_triangle_free(adj, u, v):
                add_edge(adj, u, v)
                edges.append((u, v))
        return adj, edges

    # 3-colorability check via backtracking
    def is_3_colorable(adj):
        color = [-1] * n
        # Order vertices by degree descending for efficiency
        deg = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -deg[i])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try 3 colors
            for c in range(3):
                # Check if color c is available for v
                ok = True
                nb = adj[v]
                while nb:
                    lsb = nb & (-nb)
                    u = lsb.bit_length() - 1
                    if color[u] == c:
                        ok = False
                        break
                    nb ^= lsb
                if ok:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    # Generate edge orderings for greedy construction
    # Base ordering: Turán-like (cross-part edges first)
    base_order = []
    # Part A: 0-5, Part B: 6-10, Part C: 11-15
    partA = list(range(0, 6))
    partB = list(range(6, 11))
    partC = list(range(11, 16))
    
    # Cross-part edges first (these will be added), within-part last (won't be added)
    for a in partA:
        for b in partB:
            base_order.append((a, b))
    for a in partA:
        for c in partC:
            base_order.append((a, c))
    for b in partB:
        for c in partC:
            base_order.append((b, c))
    # Within-part edges (will be rejected due to triangles)
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in set(base_order):
                base_order.append((i, j))
    
    # Remove duplicates
    seen = set()
    unique_order = []
    for e in base_order:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key)
            unique_order.append(key)
    base_order = unique_order

    best_edges = None
    best_count = 0

    # Try the base greedy ordering
    adj, edges = greedy_triangle_free(base_order)
    count = len(edges)
    if count > best_count:
        best_count = count
        best_edges = edges[:]
        best_adj = adj[:]

    # Now remove edges to break 3-colorability
    # Greedy removal: repeatedly remove an edge whose removal keeps triangle-free
    # and makes the graph 3-colorable, choosing the edge that minimizes loss.
    # Strategy: find a set of edges to remove such that the result is 3-colorable,
    # and the set is minimal. We want to remove as FEW edges as possible.
    
    # Approach: Find a 3-coloring of the complement of some edge set.
    # Equivalently: find 3 independent sets S1,S2,S3 covering all vertices,
    # and remove all edges within each Si. Minimize total removed edges.
    
    # This is: partition vertices into 3 sets to minimize sum of edges within each part.
    # This is the 3-way clustering / min-3-partition problem.
    
    # Greedy: start with empty removal set, check 3-colorability.
    # If not 3-colorable, find a "conflict" and remove an edge.
    
    # Better approach: directly search for the best 3-partition.
    # For each vertex, assign to one of 3 parts. Cost = edges within same part.
    # Minimize cost.
    
    # Use local search: random init, then move vertices to reduce cost.
    
    import random
    random.seed(42)
    
    # Build adjacency list for the triangle-free graph
    adj_list = [[] for _ in range(n)]
    for (u, v) in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Edge set for quick lookup
    edge_set = set()
    for (u, v) in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def partition_cost(partition):
        """Cost = number of edges with both endpoints in same part"""
        cost = 0
        for i in range(n):
            for j in adj_list[i]:
                if j > i and partition[i] == partition[j]:
                    cost += 1
        return cost
    
    def try_remove_edges(adj, edges, edge_set, adj_list):
        """Find a 3-partition minimizing internal edges. Remove those edges."""
        # Start with a random partition
        best_part = None
        best_cost = float('inf')
        
        for trial in range(200):
            if time.time() > DEADLINE:
                break
            part = [random.randint(0, 2) for _ in range(n)]
            # Local search
            improved = True
            while improved:
                improved = False
                # Try moving each vertex to a different part
                for v in range(n):
                    if time.time() > DEADLINE:
                        break
                    old_part = part[v]
                    for new_part in range(3):
                        if new_part == old_part:
                            continue
                        # Compute cost change
                        delta = 0
                        for u in adj_list[v]:
                            if part[u] == old_part:
                                delta -= 1  # was internal, becomes cross
                            if part[u] == new_part:
                                delta += 1  # was cross, becomes internal
                        if delta < 0:
                            part[v] = new_part
                            improved = True
                            break
                # Try swapping pairs
                for v1 in range(n):
                    for v2 in range(v1+1, n):
                        if time.time() > DEADLINE:
                            break
                        if part[v1] == part[v2]:
                            continue
                        # Cost of v1 in part[v1], v2 in part[v2]
                        # vs v1 in part[v2], v2 in part[v1]
                        old_cost = 0
                        new_cost = 0
                        for u in adj_list[v1]:
                            if u == v2:
                                continue
                            if part[u] == part[v1]:
                                old_cost += 1
                            if part[u] == part[v2]:
                                new_cost += 1
                        for u in adj_list[v2]:
                            if u == v1:
                                continue
                            if part[u] == part[v2]:
                                old_cost += 1
                            if part[u] == part[v1]:
                                new_cost += 1
                        # v1-v2 edge: was cross (different parts), stays cross (swapped, still different)
                        if old_cost > new_cost:
                            part[v1], part[v2] = part[v2], part[v1]
                            improved = True
            cost = partition_cost(part)
            if cost < best_cost:
                best_cost = cost
                best_part = part[:]
        
        if best_part is None:
            return None
        
        # Build the edge set to remove
        remove_set = set()
        for i in range(n):
            for j in adj_list[i]:
                if j > i and best_part[i] == best_part[j]:
                    remove_set.add((i, j))
        
        # Verify: the remaining graph should be 3-colorable
        remaining_edges = [e for e in edges if e not in remove_set]
        
        # Check 3-colorability
        adj2 = [0] * n
        for (u, v) in remaining_edges:
            adj2[u] |= (1 << v)
            adj2[v] |= (1 << u)
        
        if not is_3_colorable(adj2):
            # This shouldn't happen if partition is valid, but just in case
            return None
        
        return remaining_edges
    
    # Try to remove edges from the best triangle-free graph
    result = try_remove_edges(best_adj, best_edges, edge_set, adj_list)
    
    if result is not None:
        # Verify triangle-free
        adj_check = [0] * n
        for (u, v) in result:
            adj_check[u] |= (1 << v)
            adj_check[v] |= (1 << u)
        for i in range(n):
            for j in range(i+1, n):
                if (adj_check[i] >> j) & 1:
                    if (adj_check[i] & adj_check[j]) != 0:
                        # Triangle found! Should not happen
                        pass
        
        # Verify not 3-colorable
        if not is_3_colorable(adj_check):
            return [[u, v] for (u, v) in result]
    
    # Fallback: return the Groetzsch graph (known valid)
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges_fallback = [list(e) for e in cycle]
    for u, w in cycle:
        edges_fallback += [[u, 5 + w], [w, 5 + u]]
    edges_fallback += [[5 + u, 10] for u in range(5)]
    return edges_fallback
