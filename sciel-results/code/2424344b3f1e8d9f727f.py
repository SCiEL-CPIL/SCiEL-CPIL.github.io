import time
import random
import sys
from collections import defaultdict, deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def bfs_shortest_paths(adj, n):
        """Compute shortest path distances from all vertices using BFS."""
        dist = [[-1] * n for _ in range(n)]
        for src in range(n):
            dist[src][src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[src][v] == -1:
                        dist[src][v] = dist[src][u] + 1
                        queue.append(v)
        return dist
    
    def decompose_paths(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        # Compute degrees
        deg = [0] * n
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
        
        # Find odd-degree vertices
        odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
        
        if len(odd_vertices) == 0:
            # All even degrees - Eulerian circuit exists
            # Find Eulerian circuit using Hierholzer's algorithm
            edge_set = set(remaining_edges)
            adj_list = {i: [] for i in range(n)}
            for u, v in edge_set:
                adj_list[u].append(v)
                adj_list[v].append(u)
            
            circuit = []
            stack = []
            start = next(iter(edge_set))[0] if edge_set else 0
            stack.append(start)
            
            while stack:
                v = stack[-1]
                if adj_list[v]:
                    u = adj_list[v].pop()
                    edge_key = (v, u) if v < u else (u, v)
                    if edge_key in edge_set:
                        edge_set.discard(edge_key)
                        stack.append(u)
                else:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            if len(circuit) > 1:
                paths.append(circuit)
            return paths
        
        # Odd number of vertices - need to add edges to make even
        # Use minimum weight matching to pair odd vertices
        # Build distance matrix for odd vertices
        dist = bfs_shortest_paths(adj, n)
        
        m = len(odd_vertices)
        # Create cost matrix for matching
        cost = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    cost[i][j] = 10000  # High cost for self-matching
                else:
                    d = dist[odd_vertices[i]][odd_vertices[j]]
                    cost[i][j] = d if d > 0 else 10000
        
        # Use Hungarian algorithm for minimum weight perfect matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # The matching pairs up odd vertices
        # For each pair, we need to find a path between them
        # and add those edges to create an Eulerian graph
        
        # Build the set of edges to add (paths between matched pairs)
        added_edges = set()
        for i, j in zip(row_ind, col_ind):
            u, v = odd_vertices[i], odd_vertices[j]
            if u != v:
                # Find shortest path between u and v
                path = _find_shortest_path(adj, u, v, dist)
                if path:
                    for k in range(len(path) - 1):
                        a, b = path[k], path[k+1]
                        ek = (a, b) if a < b else (b, a)
                        added_edges.add(ek)
        
        # Now we have a multigraph where all vertices have even degree
        # Find Eulerian circuit in this multigraph
        # We need to track which edges are from original and which are added
        
        # Build multigraph adjacency
        from collections import defaultdict as dd
        multi_adj = dd(list)
        edge_count = dd(int)
        
        # Add original edges
        for u, v in remaining_edges:
            multi_adj[u].append(v)
            multi_adj[v].append(u)
            edge_count[(u, v) if u < v else (v, u)] += 1
        
        # Add the path edges
        for ek in added_edges:
            u, v = ek
            multi_adj[u].append(v)
            multi_adj[v].append(u)
            edge_count[ek] += 1
        
        # Find Eulerian circuit using Hierholzer's algorithm
        circuit = []
        stack = []
        start = next(iter(multi_adj)) if multi_adj else 0
        stack.append(start)
        
        used_edges = dd(int)
        
        while stack:
            v = stack[-1]
            found_next = False
            for u in multi_adj[v]:
                ek = (v, u) if v < u else (u, v)
                if used_edges[ek] < edge_count[ek]:
                    used_edges[ek] += 1
                    stack.append(u)
                    found_next = True
                    break
            if not found_next:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        if len(circuit) > 1:
            paths.append(circuit)
        
        return paths
    
    def _find_shortest_path(adj, src, dst, dist):
        """Find shortest path from src to dst using parent tracking."""
        if src == dst:
            return [src]
        
        parent = {src: None}
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            if u == dst:
                break
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
        
        if dst not in parent:
            return None
        
        # Reconstruct path
        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result (each edge a path) for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        
        paths = decompose_paths(edges)
        
        results.append(paths)

    return results
