import time
import sys
from random import Random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: single edges
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # 1. Find Minimum Weight T-Join to make graph Eulerian
        # Identify odd degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            S_edges = set(edges)
        else:
            # Compute shortest paths between odd vertices
            # Since n is small, Floyd-Warshall or Dijkstra from each odd vertex
            # Dijkstra is better for sparse graphs
            dist = {}
            prev = {}
            odd_indices = {v: i for i, v in enumerate(odd_vertices)}
            m = len(odd_vertices)
            
            # dist_matrix[i][j] is distance between odd_vertices[i] and odd_vertices[j]
            dist_matrix = np.zeros((m, m))
            # prev_matrix to reconstruct paths? 
            # We need the actual paths to get the edges in T.
            # Let's store the path edges for each pair.
            paths = {}
            
            for i, u in enumerate(odd_vertices):
                # Dijkstra from u
                dist_u = {v: float('inf') for v in range(n)}
                dist_u[u] = 0
                prev_u = {v: None for v in range(n)}
                visited = set()
                
                # Simple Dijkstra without heap for n=30
                while len(visited) < n:
                    # Find unvisited node with min dist
                    min_dist = float('inf')
                    min_node = -1
                    for v in range(n):
                        if v not in visited and dist_u[v] < min_dist:
                            min_dist = dist_u[v]
                            min_node = v
                    if min_node == -1:
                        break
                    visited.add(min_node)
                    
                    for w in adj[min_node]:
                        if w not in visited:
                            new_dist = dist_u[min_node] + 1
                            if new_dist < dist_u[w]:
                                dist_u[w] = new_dist
                                prev_u[w] = min_node
                
                for j, v in enumerate(odd_vertices):
                    if i == j:
                        continue
                    dist_matrix[i][j] = dist_u[v]
                    # Reconstruct path
                    if dist_u[v] == float('inf'):
                        continue
                    path_edges = set()
                    curr = v
                    while curr != u:
                        p = prev_u[curr]
                        if p is None:
                            break
                        path_edges.add(tuple(sorted((p, curr))))
                        curr = p
                    paths[(i, j)] = path_edges
            
            # DP for Minimum Weight Perfect Matching on odd_vertices
            # m is even
            if m % 2 != 0:
                # Should not happen in a graph
                S_edges = set(edges)
            else:
                # dp[mask]
                size = 1 << m
                dp = [float('inf')] * size
                dp[0] = 0
                # To reconstruct, store the choice
                choice = [-1] * size
                
                for mask in range(size):
                    if dp[mask] == float('inf'):
                        continue
                    # Find first set bit
                    i = 0
                    while i < m and not (mask & (1 << i)):
                        i += 1
                    if i == m:
                        continue
                    
                    # Try to pair i with j
                    for j in range(i + 1, m):
                        if mask & (1 << j):
                            new_mask = mask ^ (1 << i) ^ (1 << j)
                            cost = dist_matrix[i][j]
                            if dp[new_mask] + cost < dp[mask]:
                                dp[mask] = dp[new_mask] + cost
                                choice[mask] = j
                                
                # Reconstruct T edges
                T_edges = set()
                mask = (1 << m) - 1
                while mask:
                    j = choice[mask]
                    if j == -1:
                        break
                    i = 0
                    while i < m and not (mask & (1 << i)):
                        i += 1
                    # Pair i and j
                    # Add path edges
                    if (i, j) in paths:
                        T_edges.symmetric_difference_update(paths[(i, j)])
                    elif (j, i) in paths:
                        T_edges.symmetric_difference_update(paths[(j, i)])
                    mask ^= (1 << i) ^ (1 << j)
                    
                S_edges = set(edges) - T_edges
        
        # 2. Decompose G[S_edges] into cycles
        # Build adjacency for S
        adj_S = {i: set() for i in range(n)}
        for u, v in S_edges:
            adj_S[u].add(v)
            adj_S[v].add(u)
            
        cycles = []
        remaining_edges = set(S_edges)
        
        # Heuristic: Find longest cycle
        def find_longest_cycle(adj, n, limit_time=0.5):
            # Try to find a cycle
            # Use DFS
            best_cycle = []
            start_time_local = time.time()
            
            # Try starting from each node
            for start in range(n):
                if not adj[start]:
                    continue
                # DFS
                stack = [(start, [start])]
                visited = {start}
                # To find a cycle, we need to detect back edge
                # This simple DFS finds *a* cycle, not necessarily longest
                # For better results, we can try multiple starts or random orders
                pass
            
            # Simpler heuristic: Just find any cycle
            # We'll use a simple DFS to find a cycle
            visited = set()
            for start in range(n):
                if start in visited:
                    continue
                # DFS
                stack = [(start, [start])]
                local_visited = {start}
                while stack:
                    node, path = stack.pop()
                    if node in local_visited and len(path) > 1:
                        # Found cycle? No, this is just path
                        pass
                    # This logic is tricky.
                    # Let's just find a cycle using standard method
                    break
                
                # Standard cycle finding
                # Pick a node, DFS, if we hit a visited node that is not parent, we have a cycle
                # Let's implement a robust find_cycle
                pass
            
            # Fallback: Find any cycle
            for u in range(n):
                if not adj[u]:
                    continue
                visited = {u}
                parent = {u: -1}
                stack = [u]
                found = False
                while stack and not found:
                    node = stack.pop()
                    for nb in adj[node]:
                        if nb == parent[node]:
                            continue
                        if nb in visited:
                            # Cycle found
                            # Reconstruct
                            cycle_nodes = [nb]
                            curr = node
                            while curr != nb:
                                cycle_nodes.append(curr)
                                curr = parent[curr]
                            cycle_nodes.append(nb)
                            # Convert to edges
                            cycle_edges = []
                            for i in range(len(cycle_nodes) - 1):
                                u_c = cycle_nodes[i]
                                v_c = cycle_nodes[i+1]
                                cycle_edges.append(tuple(sorted((u_c, v_c))))
                            # Check distinct vertices
                            if len(set(cycle_nodes)) >= 3:
                                return cycle_edges
                        else:
                            visited.add(nb)
                            parent[nb] = node
                            stack.append(nb)
            return None

        # The above find_cycle is weak. Let's use a better one.
        # Since n is small, we can try to find the longest cycle with a limited search.
        
        def find_cycle(adj, n):
            # Returns list of edges
            for start in range(n):
                if not adj[start]:
                    continue
                visited = {start}
                parent = {start: -1}
                stack = [start]
                while stack:
                    node = stack.pop()
                    for nb in list(adj[node]):
                        if nb == parent[node]:
                            continue
                        if nb in visited:
                            # Cycle
                            cycle_nodes = [nb]
                            curr = node
                            while curr != nb:
                                cycle_nodes.append(curr)
                                curr = parent[curr]
                            if len(cycle_nodes) >= 3:
                                edges = []
                                for i in range(len(cycle_nodes)):
                                    u_c = cycle_nodes[i]
                                    v_c = cycle_nodes[(i+1) % len(cycle_nodes)]
                                    edges.append(tuple(sorted((u_c, v_c))))
                                return edges
                        else:
                            visited.add(nb)
                            parent[nb] = node
                            stack.append(nb)
            return None

        # To improve, we can try to remove edges that are in triangles first? No, we want long cycles.
        # Let's just use the simple find_cycle. It's fast.
        # To get better results, we can try to find the cycle with max length by trying all starts?
        # The simple DFS depends on the order.
        
        # Let's try a slightly better heuristic:
        # Pick the node with highest degree to start.
        
        while remaining_edges:
            if time.time() > deadline:
                break
            # Find a cycle in remaining_edges
            # Build adj for remaining
            adj_rem = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
                
            cyc = find_cycle(adj_rem, n)
            if cyc:
                cycles.append(cyc)
                for e in cyc:
                    remaining_edges.remove(e)
            else:
                # No more cycles
                break
        
        # Remaining edges in S_edges are not in cycles?
        # Wait, if find_cycle fails, it means no cycles exist.
        # But S_edges is Eulerian, so if it has edges, it must have a cycle.
        # So the loop should empty S_edges.
        
        # The edges in T_edges are single pieces
        # The edges in remaining_edges (if any) are also single pieces?
        # No, if the loop breaks due to timeout, we have leftovers.
        
        pieces = []
        for c in cycles:
            # Convert to list of vertices for the piece
            # The piece is a cycle given by vertices
            # We have edges. Reconstruct vertices.
            # Simple: just output the edges? No, "cycle given by at least three distinct vertices"
            # The verifier likely accepts a list of vertices.
            # Let's format the cycle as a list of vertices.
            # We need to order the vertices.
            # The edges are (u,v).
            # We can just output the list of edges? 
            # "A piece is a single edge [u, v] or a cycle given by at least three distinct vertices"
            # So cycle piece = [v1, v2, ..., vk]
            
            # Reconstruct vertex list from edges
            # This is a bit tedious. 
            # Alternatively, we can store the vertex list when we find the cycle.
            
            pass
            
        # This implementation is getting complex and error-prone in a short time.
        # Let's simplify.
        # Just return single edges if the cycle decomposition is too complex to implement correctly in time.
        # But we want to beat 1.0.
        # Even finding triangles helps.
        
        # Let's restart the logic for one graph to be safe and simple.
        
    # Given the complexity, I will implement a simpler version:
    # 1. Find T-Join to get Eulerian subgraph S.
    # 2. Decompose S into cycles using a simple DFS that finds *any* cycle.
    # 3. Return cycles and T edges.
    
    # I will rewrite the loop inside construct to be robust.

    results = []
    for n, p, seed in specs:
        if time.time() > deadline:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        T_edges = set()
        if odd_vertices:
            m = len(odd_vertices)
            if m % 2 == 0:
                # Compute distances
                dist_mat = np.zeros((m, m))
                paths = {}
                for i, u in enumerate(odd_vertices):
                    # Dijkstra
                    dist_u = [float('inf')] * n
                    prev_u = [-1] * n
                    dist_u[u] = 0
                    visited = [False] * n
                    for _ in range(n):
                        min_d = float('inf')
                        min_v = -1
                        for v in range(n):
                            if not visited[v] and dist_u[v] < min_d:
                                min_d = dist_u[v]
                                min_v = v
                        if min_v == -1: break
                        visited[min_v] = True
                        for w in adj[min_v]:
                            if not visited[w] and dist_u[min_v] + 1 < dist_u[w]:
                                dist_u[w] = dist_u[min_v] + 1
                                prev_u[w] = min_v
                    for j, v in enumerate(odd_vertices):
                        if i == j: continue
                        dist_mat[i][j] = dist_u[v]
                        if dist_u[v] < float('inf'):
                            pe = set()
                            curr = v
                            while curr != u:
                                p = prev_u[curr]
                                if p == -1: break
                                pe.add(tuple(sorted((p, curr))))
                                curr = p
                            paths[(i, j)] = pe
                
                # DP
                size = 1 << m
                dp = [float('inf')] * size
                dp[0] = 0
                choice = [-1] * size
                for mask in range(size):
                    if dp[mask] == float('inf'): continue
                    i = 0
                    while i < m and not (mask & (1 << i)):
                        i += 1
                    if i == m: continue
                    for j in range(i + 1, m):
                        if mask & (1 << j):
                            nm = mask ^ (1 << i) ^ (1 << j)
                            if dp[nm] + dist_mat[i][j] < dp[mask]:
                                dp[mask] = dp[nm] + dist_mat[i][j]
                                choice[mask] = j
                                
                mask = (1 << m) - 1
                while mask:
                    j = choice[mask]
                    if j == -1: break
                    i = 0
                    while i < m and not (mask & (1 << i)):
                        i += 1
                    if (i, j) in paths:
                        T_edges.symmetric_difference_update(paths[(i, j)])
                    elif (j, i) in paths:
                        T_edges.symmetric_difference_update(paths[(j, i)])
                    mask ^= (1 << i) ^ (1 << j)
        
        S_edges = set(edges) - T_edges
        
        # Decompose S_edges
        adj_S = {i: set() for i in range(n)}
        for u, v in S_edges:
            adj_S[u].add(v)
            adj_S[v].add(u)
            
        pieces = []
        remaining = set(S_edges)
        
        def find_cycle(adj, n):
            for start in range(n):
                if not adj[start]: continue
                visited = {start}
                parent = {start: -1}
                stack = [start]
                while stack:
                    node = stack.pop()
                    for nb in list(adj[node]):
                        if nb == parent[node]: continue
                        if nb in visited:
                            cyc_nodes = [nb]
                            curr = node
                            while curr != nb:
                                cyc_nodes.append(curr)
                                curr = parent[curr]
                            if len(cyc_nodes) >= 3:
                                return cyc_nodes
                        else:
                            visited.add(nb)
                            parent[nb] = node
                            stack.append(nb)
            return None

        while remaining:
            if time.time() > deadline: break
            # Rebuild adj for remaining
            adj_rem = {i: set() for i in range(n)}
            for u, v in remaining:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            cyc_nodes = find_cycle(adj_rem, n)
            if cyc_nodes:
                pieces.append(cyc_nodes)
                # Remove edges
                for i in range(len(cyc_nodes)):
                    u = cyc_nodes[i]
                    v = cyc_nodes[(i+1) % len(cyc_nodes)]
                    e = tuple(sorted((u, v)))
                    remaining.discard(e)
            else:
                break
        
        # Add remaining edges and T_edges as single pieces
        for e in T_edges:
            pieces.append(list(e))
        for e in remaining:
            pieces.append(list(e))
            
        results.append(pieces)
        
    return results

construct()
