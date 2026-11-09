import sys
from collections import defaultdict
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges):
        if not edges:
            return []
        
        # Map edge index to tuple
        edge_list = list(edges)
        E = len(edge_list)
        
        # Build adjacency for edges (share vertex)
        # adj[i] = set of j such that edge i and edge j share a vertex
        adj = [set() for _ in range(E)]
        for i in range(E):
            u, v = edge_list[i]
            for j in range(E):
                if i == j: continue
                x, y = edge_list[j]
                if u == x or u == y or v == x or v == y:
                    adj[i].add(j)
        
        # Build bipartite graph for matching
        # Left: 0..E-1, Right: 0..E-1
        # Use scipy or manual Hopcroft-Karp
        # Given small E, manual or simple BFS matching is fine.
        # Let's use a simple BFS-based Hopcroft-Karp implementation
        
        # pair_u[i] = matched right node for left node i
        # pair_v[j] = matched left node for right node j
        pair_u = [-1] * E
        pair_v = [-1] * E
        
        def bfs():
            from collections import deque
            dist = [-1] * E
            queue = deque()
            for u in range(E):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found, dist

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching_size = 0
        while time.time() < deadline:
            found, dist = bfs()
            if not found:
                break
            for u in range(E):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        # Reconstruct paths
        # next_edge[i] = j if i is matched to j (i -> j)
        next_edge = [-1] * E
        for i in range(E):
            if pair_u[i] != -1:
                next_edge[i] = pair_u[i]
        
        visited = [False] * E
        paths = []
        
        for i in range(E):
            if not visited[i]:
                # Start a new path
                path_edges = []
                cur = i
                while cur != -1 and not visited[cur]:
                    visited[cur] = True
                    path_edges.append(cur)
                    cur = next_edge[cur]
                
                # Reconstruct vertex path
                if not path_edges:
                    continue
                    
                # Determine vertex sequence
                verts = []
                prev_end = None
                valid = True
                for idx, e_idx in enumerate(path_edges):
                    u, v = edge_list[e_idx]
                    if prev_end is None:
                        verts.append(u)
                        prev_end = v
                    else:
                        if u == prev_end:
                            verts.append(v)
                            prev_end = v
                        elif v == prev_end:
                            verts.append(u)
                            prev_end = u
                        else:
                            valid = False
                            break
                
                if not valid:
                    # Should not happen if matching is correct, but handle
                    # Fallback: single edge paths
                    for e_idx in path_edges:
                        paths.append(list(edge_list[e_idx]))
                    continue
                
                # Check if simple (no repeated vertices)
                if len(set(verts)) == len(verts):
                    paths.append(verts)
                else:
                    # Cycle or repeated vertex. Split into 2 paths.
                    # Find the point where the vertex repeats.
                    # For a cycle, the first vertex repeats at the end.
                    # We can split the edge list.
                    # Simple strategy: Split the vertex list in half?
                    # No, we must respect edge boundaries.
                    # If it's a cycle, we can just take the first k-1 edges as one path, and the last edge as another?
                    # Or better: find the first repeated vertex index.
                    seen = {}
                    split_idx = -1
                    for k, vert in enumerate(verts):
                        if vert in seen:
                            split_idx = k
                            break
                        seen[vert] = k
                    
                    if split_idx != -1:
                        # Split vertex path at split_idx
                        # Path 1: verts[0:split_idx]
                        # Path 2: verts[split_idx:]
                        # But we need to ensure edges are covered.
                        # This splitting logic is tricky with edges.
                        # Alternative: Just output the path as is if we allow cycles?
                        # Task says "simple path".
                        # If we can't easily split, we can output single edges for this component.
                        # This increases path count but is safe.
                        for e_idx in path_edges:
                            paths.append(list(edge_list[e_idx]))
                    else:
                        paths.append(verts)
                        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = solve_graph(edges)
        results.append(paths)
        
    return results
