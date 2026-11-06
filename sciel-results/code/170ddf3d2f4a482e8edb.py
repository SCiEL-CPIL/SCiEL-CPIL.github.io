import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[u] contains v in right set
        n_left, n_right: number of nodes in left and right sets
        Returns matching: dict left_node -> right_node
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u

    def decompose_with_matching(edges):
        if not edges:
            return []
        
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # Build adjacency for bipartite graph
        # Left: edge indices, Right: edge indices
        # Edge (u_L, v_R) exists if edge u and edge v share a vertex
        adj = [[] for _ in range(m)]
        
        # Group edges by vertex
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)
        
        for v, e_list in vertex_edges.items():
            # For every pair of edges incident to v, we can connect them
            # e1 can be followed by e2
            for i in range(len(e_list)):
                for j in range(len(e_list)):
                    if i != j:
                        adj[e_list[i]].append(e_list[j])
        
        # Find maximum matching
        pair_u = hopcroft_karp(adj, m, m)
        
        # Reconstruct paths
        # next_edge[i] = j means edge i is followed by edge j
        next_edge = [-1] * m
        prev_edge = [-1] * m
        for u in range(m):
            if pair_u[u] != -1:
                v = pair_u[u]
                next_edge[u] = v
                prev_edge[v] = u
        
        paths = []
        visited_edges = [False] * m
        
        for i in range(m):
            if prev_edge[i] == -1:
                # Start of a path
                path = []
                current = i
                while current != -1:
                    if visited_edges[current]:
                        break # Should not happen if matching is valid
                    visited_edges[current] = True
                    u, v = edges[current]
                    # Determine direction
                    # If this is the first edge, we just add both
                    # If we came from prev, we need to maintain order
                    if not path:
                        path.append(u)
                        path.append(v)
                    else:
                        # We need to figure out which end of the current edge connects to the last vertex of the path
                        last_v = path[-1]
                        # The current edge is (u, v). One of u or v must be last_v
                        if u == last_v:
                            path.append(v)
                        elif v == last_v:
                            path.append(u)
                        else:
                            # This indicates an error in logic or non-simple trail structure
                            # Break to avoid infinite loop
                            break
                    current = next_edge[current]
                
                if path:
                    paths.append(path)
        
        # Check if all edges are covered
        if not all(visited_edges):
            return None # Fallback needed
        
        # Verify simplicity
        for p in paths:
            if len(p) != len(set(p)):
                return None # Not simple, fallback needed
                
        return paths

    def greedy_decompose(edges):
        # Fallback greedy
        remaining = set(edges)
        paths = []
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        while remaining:
            # Pick a start vertex
            # Prefer odd degree
            degs = defaultdict(int)
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            odd = [v for v, d in degs.items() if d % 2 == 1]
            if odd:
                start = random.choice(odd)
            else:
                start = random.choice(list(degs.keys()))
                
            path = [start]
            current = start
            visited = {start}
            
            while True:
                avail = [nb for nb in adj[current] if tuple(sorted((current, nb))) in remaining and nb not in visited]
                if not avail:
                    break
                # Heuristic: pick neighbor with highest degree in remaining
                avail.sort(key=lambda x: sum(1 for e in remaining if x in e), reverse=True)
                next_v = avail[0]
                e = tuple(sorted((current, next_v)))
                remaining.remove(e)
                adj[current].remove(next_v)
                adj[next_v].remove(current)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
            paths.append(path)
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > deadline:
            break
        edges = instance(*spec)
        
        # Try matching first
        paths = decompose_with_matching(edges)
        
        if paths is None:
            # Fallback to greedy
            paths = greedy_decompose(edges)
            
        results.append(paths)
        
    return results
