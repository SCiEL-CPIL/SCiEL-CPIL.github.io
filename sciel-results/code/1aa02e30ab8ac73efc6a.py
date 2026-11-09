"""Rewritten program: triangle-free diameter-4 augmentation with local search."""
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = P['n']
    base_edges = {tuple(e) for e in P['edges']}
    base_set = set(base_edges)
    # Add reversed
    for (u, v) in base_edges:
        base_set.add((v, u))

    # Candidate edges: |u-v| >= 3 to avoid triangles with path edges
    # Actually, need to check triangle-freedom carefully
    # An edge (u,v) creates a triangle if there exists w such that (u,w) and (w,v) are in the graph
    # With path edges, (u,v) with v-u=2 would create triangle u-(u+1)-v
    # With v-u=1 it's already an edge
    # So candidates are v-u >= 3

    all_candidates = []
    for u in range(n):
        for v in range(u + 3, n):
            all_candidates.append((u, v))

    def is_triangle_free(edge_set, add_edge):
        """Check if adding edge to edge_set keeps graph triangle-free."""
        u, v = add_edge
        # Check all neighbors of u and v in current edge_set
        # Triangle exists if there's w with (u,w) and (w,v) both in edge_set
        neighbors_u = set()
        neighbors_v = set()
        for (a, b) in edge_set:
            if a == u:
                neighbors_u.add(b)
            elif b == u:
                neighbors_u.add(a)
            if a == v:
                neighbors_v.add(b)
            elif b == v:
                neighbors_v.add(a)
        # Intersection of neighbors_u and neighbors_v gives common neighbors
        common = neighbors_u & neighbors_v
        return len(common) == 0

    def get_diameter(edge_set):
        """Compute diameter of graph given by edge_set using BFS from each node."""
        adj = {i: set() for i in range(n)}
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        # Check connectivity
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append(nb)
        if len(visited) != n:
            return float('inf')

        max_dist = 0
        for src in range(n):
            dist = {src: 0}
            queue = [src]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb not in dist:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            if len(dist) < n:
                return float('inf')
            max_dist = max(max_dist, max(dist.values()))
        return max_dist

    def is_valid(edge_set):
        """Check diameter <= 4."""
        return get_diameter(edge_set) <= 4

    # Greedy construction: start with path, add edges to reduce diameter
    # Strategy: iteratively add the edge that most reduces diameter while maintaining triangle-freedom

    def greedy_construct(candidates, seed):
        random.seed(seed)
        edge_set = set(base_edges)
        remaining = list(candidates)
        random.shuffle(remaining)

        # First, compute current diameter
        current_diam = get_diameter(edge_set)
        if current_diam <= 4:
            return list(edge_set)

        # Try to add edges greedily
        for edge in remaining:
            if time.time() > deadline:
                break
            # Check triangle-freedom
            if not is_triangle_free(edge_set, edge):
                continue
            # Try adding
            new_set = edge_set | {edge}
            new_diam = get_diameter(new_set)
            if new_diam < current_diam:
                edge_set = new_set
                current_diam = new_diam
                if current_diam <= 4:
                    break

        if current_diam <= 4:
            return list(edge_set)
        return None

    # Local search: start from greedy solution, try to remove edges
    def local_search(edge_list, seed):
        random.seed(seed)
        edge_set = set(edge_list)
        current_diam = get_diameter(edge_set)
        if current_diam > 4:
            return None

        # Try to remove edges one by one
        removed = True
        while removed:
            removed = False
            edges_to_try = list(edge_set)
            random.shuffle(edges_to_try)
            for edge in edges_to_try:
                if time.time() > deadline:
                    return list(edge_set) if get_diameter(edge_set) <= 4 else None
                new_set = edge_set - {edge}
                new_diam = get_diameter(new_set)
                if new_diam <= 4:
                    edge_set = new_set
                    current_diam = new_diam
                    removed = True
                    break
        return list(edge_set)

    # Multi-start: try different greedy seeds and local searches
    best_solution = None
    best_count = float('inf')

    # Try multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break

        sol = greedy_construct(all_candidates, seed)
        if sol is not None:
            refined = local_search(sol, seed)
            if refined is not None:
                count = len(refined) - n + 1  # number of added edges (base has n-1 edges)
                # Actually, count = len(refined) - (n-1)
                count = len(refined) - (n - 1)
                if count < best_count:
                    best_count = count
                    best_solution = refined
                    if best_count == 0:
                        break

    # If no solution found, return empty (invalid but best effort)
    if best_solution is None:
        return []

    # Convert to list of [u, v] pairs
    result = []
    for (u, v) in best_solution:
        if (u, v) not in base_set:
            result.append([int(u), int(v)])

    return result
