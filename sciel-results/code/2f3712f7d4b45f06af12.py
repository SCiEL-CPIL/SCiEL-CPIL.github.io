"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    _start = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

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

    def bfs_shortest_path(src, dst, adj):
        if src == dst:
            return [src]
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
        aug_edges = []
        for (u, v) in edges:
            aug_edges.append((u, v, False))
        for (u, v) in virtual_edges:
            aug_edges.append((u, v, True))
            
        aug_adj = defaultdict(list)
        for i, (u, v, is_v) in enumerate(aug_edges):
            aug_adj[u].append(i)
            aug_adj[v].append(i)
            
        used_edge_idx = [False] * len(aug_edges)
        
        start = edges[0][0]
        
        stack = [start]
        circuit_vertices = []
        circuit_edge_flags = []
        
        while stack:
            v = stack[-1]
            found = False
            for idx in aug_adj[v]:
                if not used_edge_idx[idx]:
                    u, w, is_v = aug_edges[idx]
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
        
        paths = []
        current_path = [circuit_vertices[0]]
        
        for i in range(len(circuit_edge_flags)):
            is_v = circuit_edge_flags[i]
            next_v = circuit_vertices[i + 1]
            current_path.append(next_v)
            
            if is_v:
                if len(current_path) >= 2:
                    if len(set(current_path)) == len(current_path):
                        paths.append(current_path)
                    else:
                        return []
                current_path = [next_v]
        
        if len(current_path) >= 2:
            if len(set(current_path)) == len(current_path):
                paths.append(current_path)
            else:
                return []
                
        return paths

    def improved_path_cover(edges):
        """
        Minimum cost perfect matching on odd-degree vertices.
        1. Compute all-pairs shortest paths (distances) among odd-degree vertices.
        2. Solve min-cost perfect matching on the odd vertices using Hungarian algorithm.
        3. The matched pairs define 'virtual' connections. Adding these to the graph makes it Eulerian.
        4. Find Eulerian circuit in the augmented multigraph.
        5. Extract simple paths by cutting the circuit at the virtual edges.
        6. If extracted paths are not simple, fall back to greedy.
        """
        if not edges:
            return []

        adj = build_adjacency(edges)
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        k = len(odd_vertices)

        if k == 0:
            circuit = find_eulerian_circuit(edges)
            if circuit:
                return [circuit]
            return greedy_path_cover(edges)

        if k == 2:
            sp = bfs_shortest_path(odd_vertices[0], odd_vertices[1], adj)
            if sp and len(sp) >= 2:
                virtual_edges = []
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)
                
                paths = extract_paths_from_eulerian(edges, virtual_edges, adj)
                if paths:
                    return paths
            return greedy_path_cover(edges)

        INF = 10**9
        dist_matrix = np.zeros((k, k), dtype=int)
        
        for i, src in enumerate(odd_vertices):
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

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        pairs = []
        for i, j in zip(row_ind, col_ind):
            if i < j:
                pairs.append((odd_vertices[i], odd_vertices[j]))
        
        virtual_edges = []
        for (a, b) in pairs:
            sp = bfs_shortest_path(a, b, adj)
            if sp and len(sp) >= 2:
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)
            else:
                pass

        paths = extract_paths_from_eulerian(edges, virtual_edges, adj)
        
        if paths:
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

    def validate(paths, edges):
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        covered = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    return False
                if key in covered:
                    return False
                covered.add(key)
        return covered == edge_set

    def local_search(paths, edges, adj, time_limit=5.0):
        """
        Local search to merge paths where possible.
        1. Represent paths as sets of edges.
        2. Identify potential merges: path A ends at u, path B starts at v, and (u, v) is an edge.
           Wait, merging two paths P1 (v1..vk) and P2 (w1..wm) requires an edge (vk, w1) or (vk, wm).
           If we use edge (vk, w1), the new path is v1..vk..w1..wm.
           This edge (vk, w1) must exist in the graph and NOT be in P1 or P2.
           But wait, if (vk, w1) is not in P1 or P2, it must be in some other path P3.
           So we need to split P3 to free up the edge.
           This is a 3-path move: merge P1 and P2, split P3.
           
           Simpler 2-path move:
           If P1 ends at u and P2 starts at v, and (u, v) is an edge.
           If (u, v) is in P1 or P2, we can't use it.
           If (u, v) is in P3, we must split P3.
           
           Let's implement the 3-path move:
           Find P1, P2, P3.
           P1 ends at u, P2 starts at v. Edge (u, v) exists.
           (u, v) is in P3.
           Split P3 at (u, v) into P3a and P3b.
           Merge P1 and P2 using (u, v) to form P12.
           New set of paths: {P12, P3a, P3b} replacing {P1, P2, P3}.
           Number of paths stays the same? No, 3 paths become 3 paths.
           Wait, merging P1 and P2 reduces count by 1. Splitting P3 increases count by 1. Net change 0.
           This is not useful for minimizing count.
           
           What if (u, v) is NOT in any path? Impossible, all edges are covered.
           
           What if P1 and P2 are the same path? No, distinct paths.
           
           Is there a way to reduce path count?
           Yes, if we can merge P1 and P2 without splitting another path.
           This requires an edge (u, v) that is NOT in any path. Impossible.
           
           So local search to REDUCE path count is impossible if the cover is edge-disjoint and complete?
           Wait. The number of paths in a path cover is related to the number of odd degree vertices.
           Specifically, any path cover has at least (number of odd degree vertices) / 2 paths.
           If we achieve this lower bound, we cannot reduce the count further.
           The improved_path_cover achieves this lower bound (if paths are simple).
           So local search for count reduction is likely futile if we already hit the lower bound.
           
           However, if improved_path_cover failed due to non-simple paths, we fall back to greedy.
           Greedy likely has more paths than the lower bound.
           Can local search reduce greedy's count?
           Yes. If greedy has 10 paths, and lower bound is 5, local search might find a way to merge.
           How?
           If we have a path cover, we can try to "swap" edges to merge paths.
           This is complex.
           
           Given the time limit and complexity, and the fact that improved_path_cover usually works,
           I will skip complex local search and rely on improved_path_cover.
           If improved_path_cover fails, greedy is the fallback.
        """
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 5:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        try:
            paths = improved_path_cover(edges)
        except Exception:
            paths = greedy_path_cover(edges)

        if not validate(paths, edges):
            paths = greedy_path_cover(edges)
            if not validate(paths, edges):
                paths = [[list(e)] for e in edges]

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
