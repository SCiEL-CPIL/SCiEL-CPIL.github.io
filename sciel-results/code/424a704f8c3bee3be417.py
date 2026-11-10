"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # 10s safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def greedy_path_cover(edges):
        """
        Fallback greedy edge-disjoint path cover.
        """
        if not edges:
            return []

        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))

        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)

        used_edges = set()
        paths = []

        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break

            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)

            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break

            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor

            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)

            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])

        return paths

    def improved_path_cover(edges):
        """
        Minimum cost perfect matching on odd-degree vertices.
        1. Compute all-pairs shortest paths (distances) among odd-degree vertices.
        2. Solve min-cost perfect matching on the odd vertices using Hungarian algorithm.
        3. The matched pairs define 'virtual' connections. Adding these to the graph makes it Eulerian.
        4. Find Eulerian circuit in the augmented multigraph.
        5. Extract simple paths by cutting the circuit at the virtual edges.
        6. If extracted paths are not simple (due to shared vertices in virtual paths), fall back to greedy.
           (Note: For sparse random graphs, simple paths are highly likely, but we must verify).
        """
        if not edges:
            return []

        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        k = len(odd_vertices)

        if k == 0:
            # Already Eulerian
            # Find Eulerian circuit
            circuit = find_eulerian_circuit(edges)
            if circuit:
                return [circuit]
            return greedy_path_cover(edges)

        if k == 2:
            # Only one pair needed
            # Find shortest path between the two
            sp = bfs_shortest_path(odd_vertices[0], odd_vertices[1], adj)
            if sp and len(sp) >= 2:
                virtual_edges = []
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)
                
                # Augment and find Eulerian circuit
                paths = extract_paths_from_eulerian(edges, virtual_edges, adj)
                if paths:
                    return paths
            return greedy_path_cover(edges)

        # Compute distances between all pairs of odd vertices
        # Use BFS from each odd vertex
        # dist[i][j] is distance from odd_vertices[i] to odd_vertices[j]
        # If no path, set to a large number (but graph is connected, so should be fine)
        INF = 10**9
        dist_matrix = np.zeros((k, k), dtype=int)
        
        for i, src in enumerate(odd_vertices):
            # BFS from src
            dist = {src: 0}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            
            for j, dst in enumerate(odd_vertices):
                if dst in dist:
                    dist_matrix[i][j] = dist[dst]
                else:
                    dist_matrix[i][j] = INF

        # Solve min-cost perfect matching
        # linear_sum_assignment solves min sum of cost[i][j] for assigned pairs
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # The pairs are (row_ind[i], col_ind[i])
        # Since it's a symmetric matrix and perfect matching, we just need the pairs.
        # Note: linear_sum_assignment returns a permutation. We need to ensure we don't double count or miss.
        # For a perfect matching in a symmetric matrix, the assignment is a set of pairs.
        # Actually, linear_sum_assignment finds a minimum weight matching in a bipartite graph.
        # For symmetric matrix, this corresponds to a perfect matching in the original graph.
        
        pairs = []
        used_in_matching = set()
        for i, j in zip(row_ind, col_ind):
            if i == j:
                # This shouldn't happen for a valid perfect matching on distinct vertices
                # unless k=0, but k is even and >0.
                continue
            if i not in used_in_matching and j not in used_in_matching:
                pairs.append((odd_vertices[i], odd_vertices[j]))
                used_in_matching.add(i)
                used_in_matching.add(j)
        
        # It's possible that the assignment has i->j and j->i as separate entries?
        # No, linear_sum_assignment returns a one-to-one mapping.
        # If i is mapped to j, j is mapped to some l.
        # For a symmetric cost matrix, the optimal assignment will be symmetric in terms of pairs, 
        # but the algorithm treats it as bipartite.
        # We need to be careful. The standard way to get pairs from a symmetric assignment:
        # Iterate through assigned (i, j). If i < j, it's a pair.
        # But wait, if i=0, j=1 and i=1, j=0 are both in the assignment?
        # No, linear_sum_assignment returns a permutation P such that cost[i][P[i]] is minimized.
        # It does not return both (0,1) and (1,0) as separate "selected" cells in the sense of a set of edges.
        # It selects a set of cells (i, P[i]).
        # If the matrix is symmetric, the optimal solution will satisfy P[P[i]] = i for all i?
        # Not necessarily, but the cost is the same.
        # To be safe, we can just take the pairs (i, P[i]) where i < P[i].
        
        pairs = []
        for i, j in zip(row_ind, col_ind):
            if i < j:
                pairs.append((odd_vertices[i], odd_vertices[j]))
        
        # Find shortest paths for each pair
        virtual_edges = []
        for (a, b) in pairs:
            sp = bfs_shortest_path(a, b, adj)
            if sp and len(sp) >= 2:
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)
            else:
                # If no path found (should not happen in connected graph), skip or handle error
                pass

        # Extract paths from Eulerian circuit of augmented graph
        paths = extract_paths_from_eulerian(edges, virtual_edges, adj)
        
        if paths:
            # Verify validity
            edge_set = set()
            for (u, v) in edges:
                edge_set.add((u, v) if u <= v else (v, u))
            
            covered = set()
            valid = True
            for p in paths:
                if len(p) < 2:
                    valid = False
                    break
                if len(set(p)) != len(p):
                    valid = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid = False
                        break
                    if key in covered:
                        valid = False
                        break
                    covered.add(key)
                if not valid:
                    break
            
            if valid and covered == edge_set:
                return paths
        
        return greedy_path_cover(edges)

    def bfs_shortest_path(src, dst, adj):
        if src == dst:
            return []
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    if nb == dst:
                        path = []
                        node = dst
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    q.append(nb)
        return None

    def find_eulerian_circuit(edges):
        """Find an Eulerian circuit in a graph where all vertices have even degree."""
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)

        if not adj:
            return []

        start = next(iter(adj))
        edge_count = defaultdict(int)
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            edge_count[key] += 1

        hier_adj = defaultdict(list)
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            hier_adj[u].append((v, key))
            hier_adj[v].append((u, key))

        used_count = defaultdict(int)
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            found = False
            for (nb, key) in hier_adj[v]:
                if used_count[key] < edge_count[key]:
                    used_count[key] += 1
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        circuit.reverse()
        return circuit

    def extract_paths_from_eulerian(edges, virtual_edges, adj):
        """
        1. Build augmented multigraph (original + virtual edges).
        2. Find Eulerian circuit.
        3. Cut circuit at virtual edges to get paths.
        4. Return list of paths.
        """
        if not edges:
            return []
        
        # Build augmented adjacency with edge tracking
        # We need to distinguish between original and virtual edges when cutting.
        # We'll track edges by (u, v) and a counter for multiplicity if needed, 
        # but here virtual edges are just added once. 
        # However, a virtual edge might coincide with an original edge.
        # So we need to know *which* traversal is virtual.
        
        # Let's create a list of edges for the augmented graph.
        # Each entry: (u, v, is_virtual)
        aug_edges = []
        for (u, v) in edges:
            aug_edges.append((u, v, False))
        for (u, v) in virtual_edges:
            aug_edges.append((u, v, True))
            
        # Build adjacency for Hierholzer
        # adj[u] = list of indices into aug_edges
        aug_adj = defaultdict(list)
        for i, (u, v, is_v) in enumerate(aug_edges):
            aug_adj[u].append(i)
            aug_adj[v].append(i)
            
        used_edge_idx = [False] * len(aug_edges)
        
        # Find start vertex. Any vertex with degree > 0.
        # Since graph is connected and has edges, pick one from original edges.
        start = edges[0][0]
        
        stack = [start]
        circuit_vertices = []
        circuit_edge_flags = [] # True if the edge just traversed was virtual
        
        while stack:
            v = stack[-1]
            found = False
            # Iterate through neighbors
            for idx in aug_adj[v]:
                if not used_edge_idx[idx]:
                    u, w, is_v = aug_edges[idx]
                    # Determine next vertex
                    next_v = w if u == v else u
                    used_edge_idx[idx] = True
                    stack.append(next_v)
                    circuit_edge_flags.append(is_v)
                    found = True
                    break
            if not found:
                circuit_vertices.append(stack.pop())
                
        circuit_vertices.reverse()
        circuit_edge_flags.reverse()
        
        # circuit_vertices[0] is start, circuit_vertices[-1] is start
        # circuit_edge_flags[i] corresponds to edge between circuit_vertices[i] and circuit_vertices[i+1]
        
        paths = []
        current_path = [circuit_vertices[0]]
        
        for i in range(len(circuit_edge_flags)):
            is_v = circuit_edge_flags[i]
            next_v = circuit_vertices[i + 1]
            current_path.append(next_v)
            
            if is_v:
                # Cut here
                if len(current_path) >= 2:
                    # Check if this path is simple
                    if len(set(current_path)) == len(current_path):
                        paths.append(current_path)
                    else:
                        # Path is not simple. This happens if the virtual path and original path share vertices in a way that creates a loop in the "logical" path.
                        # In this case, we cannot simply return this path.
                        # We should signal failure so the caller can fall back to greedy.
                        return []
                current_path = [next_v]
        
        if len(current_path) >= 2:
            if len(set(current_path)) == len(current_path):
                paths.append(current_path)
            else:
                return []
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 5:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        try:
            paths = improved_path_cover(edges)
        except Exception:
            paths = greedy_path_cover(edges)

        # Validate
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                valid = False
                break
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if not valid or covered != edge_set:
            paths = greedy_path_cover(edges)
            # Re-validate greedy
            covered2 = set()
            valid2 = True
            for p in paths:
                if len(p) < 2:
                    valid2 = False
                    break
                if len(set(p)) != len(p):
                    valid2 = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid2 = False
                        break
                    if key in covered2:
                        valid2 = False
                        break
                    covered2.add(key)
                if not valid2:
                    break
            if not valid2 or covered2 != edge_set:
                # Last resort: each edge as its own path
                paths = [[list(e)] for e in edges]

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
