"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper to run Max Flow with vertex splitting
    def max_flow_disjoint_paths(source, targets, blocked):
        """
        Finds vertex-disjoint paths from source to each target in targets.
        Returns list of paths (list of vertices) if successful, else None.
        blocked: set of vertices that cannot be used as interiors.
        """
        k = len(targets)
        if k == 0:
            return []
        
        # Node IDs in flow graph:
        # v_in = 2*v, v_out = 2*v + 1
        # Super Source = 2*n, Super Sink = 2*n + 1
        N = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        
        # Adjacency list for flow graph: [to, cap, rev_index]
        graph = [[] for _ in range(N)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
        
        # Add vertex splitting edges
        for v in range(n):
            if v == source or v in targets:
                cap = k # Effectively infinite for our small k
            elif v in blocked:
                cap = 0
            else:
                cap = 1
            
            add_edge(2*v, 2*v+1, cap)
        
        # Add graph edges
        for u in range(n):
            for v in adj[u]:
                # Edge from u_out to v_in
                add_edge(2*u+1, 2*v, 1)
        
        # Connect Source
        add_edge(S, 2*source+1, k)
        
        # Connect Targets to Sink
        for t in targets:
            add_edge(2*t, T, 1)
            
        # Dinic's Algorithm
        flow = 0
        level = [0] * N
        it = [0] * N
        
        def bfs():
            for i in range(N):
                level[i] = -1
            q = deque([S])
            level[S] = 0
            while q:
                v = q.popleft()
                for u, cap, rev in graph[v]:
                    if cap > 0 and level[u] < 0:
                        level[u] = level[v] + 1
                        if u == T:
                            return True
                        q.append(u)
            return level[T] >= 0
        
        def dfs(v, f):
            if v == T:
                return f
            for i in range(it[v], len(graph[v])):
                it[v] = i
                u, cap, rev = graph[v][i]
                if cap > 0 and level[v] < level[u]:
                    d = dfs(u, min(f, cap))
                    if d > 0:
                        graph[v][i][1] -= d
                        graph[u][rev][1] += d
                        return d
            return 0
        
        while bfs():
            for i in range(N):
                it[i] = 0
            while True:
                f = dfs(S, float('inf'))
                if f == 0:
                    break
                flow += f
                if flow == k:
                    break
        
        if flow < k:
            return None
        
        # Extract paths
        # We need to find the paths in the residual graph where flow was sent.
        # Actually, we can just look at the edges with flow > 0.
        # Since capacities are 1 (except source/sinks/branches), the paths are simple.
        
        paths = []
        # We need to trace from S to T.
        # Since S connects only to source_out, and targets_in connect to T.
        # We can find paths by DFS on the edges with flow > 0.
        
        # Reconstruct flow graph edges with flow
        # Edge u->v has flow if graph[u][i][1] < original_cap?
        # Easier: Just look for edges where graph[v][rev][1] > 0 (reverse flow exists)
        # Or simply, since we know the structure, we can trace back from T.
        
        # Let's create a list of edges that carry flow.
        # An edge u->v carries flow if graph[u][i][1] < cap_original.
        # We didn't store cap_original. But we know cap is 1 or k.
        # Let's just re-run a simpler extraction.
        
        # For each target t, find path from source to t in the "used" subgraph.
        # Used edges are those where graph[u][i][1] < cap_init.
        # Wait, we modified the graph.
        # Let's identify used edges:
        # For each u, for each edge index i in graph[u]:
        #   if graph[u][i][1] < cap_init(u, i):
        #       flow is present.
        
        # We can just do a BFS/DFS from source to each target, only traversing edges with flow > 0.
        # How to know if flow > 0?
        # The reverse edge graph[v][rev][1] will be > 0 if flow was sent.
        
        used_edges = set()
        for u in range(N):
            for i in range(len(graph[u])):
                v, cap, rev = graph[u][i]
                # If reverse edge has positive capacity, flow was sent
                if graph[v][rev][1] > 0:
                    used_edges.add((u, v))
        
        # Now find paths from source to each target
        # Map flow node id to original vertex
        def node_to_v(node):
            if node == S or node == T:
                return None
            return node // 2
        
        # Build adjacency for used edges
        used_adj = {i: [] for i in range(N)}
        for u, v in used_edges:
            used_adj[u].append(v)
        
        found_paths = []
        for t in targets:
            # BFS from source (2*source+1) to t (2*t)
            start = 2*source+1
            end = 2*t
            q = deque([start])
            prev = {start: None}
            while q:
                curr = q.popleft()
                if curr == end:
                    # Reconstruct path
                    path_nodes = []
                    curr_node = end
                    while curr_node is not None:
                        path_nodes.append(curr_node)
                        curr_node = prev[curr_node]
                    path_nodes.reverse()
                    
                    # Convert to vertex IDs
                    # Path is S -> source_out -> ... -> t_in -> T
                    # We want source -> ... -> t
                    verts = []
                    for node in path_nodes:
                        if node == S or node == T:
                            continue
                        v = node // 2
                        # Avoid duplicates if path goes v_in -> v_out
                        if not verts or verts[-1] != v:
                            verts.append(v)
                    found_paths.append(verts)
                    break
                for next_node in used_adj.get(curr, []):
                    if next_node not in prev:
                        prev[next_node] = curr
                        q.append(next_node)
            
            if len(found_paths) < len(targets):
                return None # Should not happen if flow was k
        
        return found_paths

    # Greedy construction
    # Start with an edge
    best_result = None
    best_size = 0
    
    # Try all edges as starting pair
    for u, v in P['edges']:
        branches = [u, v]
        paths_dict = {(u, v): [u, v], (v, u): [v, u]}
        blocked = set()
        
        # Try to add more vertices
        while True:
            added = False
            # Candidates: all vertices not in branches and not blocked
            candidates = [x for x in range(n) if x not in branches and x not in blocked]
            
            # Sort candidates by degree descending (heuristic)
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for w in candidates:
                # Try to add w
                # Targets are the current branches
                new_paths = max_flow_disjoint_paths(w, branches, blocked)
                if new_paths is not None:
                    # Success
                    # Add w to branches
                    branches.append(w)
                    
                    # Update paths_dict
                    for b, path in zip(branches[:-1], new_paths):
                        # path is from w to b
                        # Store as unordered pair key
                        key1 = (w, b)
                        key2 = (b, w)
                        paths_dict[key1] = path
                        paths_dict[key2] = list(reversed(path))
                        
                        # Update blocked set with interiors
                        # Interior is path[1:-1]
                        for node in path[1:-1]:
                            blocked.add(node)
                    
                    added = True
                    break
            
            if not added:
                break
        
        if len(branches) > best_size:
            best_size = len(branches)
            # Prepare result format
            result_branches = [int(x) for x in branches]
            result_paths = []
            for i in range(len(result_branches)):
                for j in range(i+1, len(result_branches)):
                    u, v = result_branches[i], result_branches[j]
                    # Find path in paths_dict
                    p = paths_dict.get((u, v))
                    if p is None:
                        p = paths_dict.get((v, u))
                    if p:
                        result_paths.append([int(x) for x in p])
                    else:
                        # Fallback (should not happen)
                        result_paths.append([u, v])
                        
            best_result = {'branches': result_branches, 'paths': result_paths}
            
    if best_result is None:
        # Fallback to seed-like behavior
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    return best_result

# EVOLVE-BLOCK-END
