# EVOLVE-BLOCK-START
import math
import sys
from collections import deque

def construct():
    """
    Find a maximum subset of [1,256] such that a+b is never a perfect square
    for any a<=b in the set (including a=b).
    
    Approach:
    1. Exclude numbers i where 2*i is a perfect square (self-conflict).
    2. Build conflict graph on remaining numbers.
    3. Check if bipartite. If so, solve MWIS exactly via min-cut (max-flow).
    4. If not bipartite, use branch-and-bound or heuristic.
    """
    N = 256
    
    def is_square(x):
        if x < 0:
            return False
        r = math.isqrt(x)
        return r * r == x

    # Identify self-conflicts (2*i is a square)
    valid_nodes = []
    for i in range(1, N + 1):
        if not is_square(2 * i):
            valid_nodes.append(i)
            
    M = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    values = valid_nodes
    
    # Build adjacency list
    adj = [[] for _ in range(M)]
    for i in range(M):
        u = values[i]
        for j in range(i + 1, M):
            v = values[j]
            if is_square(u + v):
                adj[i].append(j)
                adj[j].append(i)
                
    # Check bipartiteness
    color = [-1] * M
    is_bipartite = True
    for i in range(M):
        if color[i] == -1:
            color[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    else:
                        if color[v] == color[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break

    if is_bipartite:
        # Solve MWIS on bipartite graph using Min-Cut / Max-Flow
        # Standard reduction: 
        #   - Source S, Sink T
        #   - For each node v:
        #       if color[v] == 0: edge S -> v with capacity 1
        #       if color[v] == 1: edge v -> T with capacity 1
        #   - For each edge (u,v) in conflict graph:
        #       if color[u] == 0 and color[v] == 1: edge u -> v with capacity INF
        #       if color[v] == 0 and color[u] == 1: edge v -> u with capacity INF
        #   - Min cut gives minimum vertex cover
        #   - Maximum independent set = V \ min_vertex_cover
        
        INF = 10**9
        total_nodes = M + 2
        S = M
        T = M + 1
        
        # flow_adj[u] = list of [v, capacity, rev_index]
        flow_adj = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
            
        for i in range(M):
            if color[i] == 0:
                add_edge(S, i, 1)
            else:
                add_edge(i, T, 1)
                
        for i in range(M):
            for j in adj[i]:
                if color[i] == 0 and color[j] == 1:
                    add_edge(i, j, INF)
                elif color[i] == 1 and color[j] == 0:
                    add_edge(j, i, INF)
                    
        # Dinic's algorithm
        def dinic():
            flow = 0
            level = [-1] * total_nodes
            ptr = [0] * total_nodes
            
            def bfs():
                for i in range(total_nodes):
                    level[i] = -1
                level[S] = 0
                q = deque([S])
                while q:
                    u = q.popleft()
                    for e in flow_adj[u]:
                        v, cap, rev = e
                        if cap > 0 and level[v] == -1:
                            level[v] = level[u] + 1
                            q.append(v)
                return level[T] != -1
                
            def dfs(u, pushed):
                if pushed == 0:
                    return 0
                if u == T:
                    return pushed
                while ptr[u] < len(flow_adj[u]):
                    e = flow_adj[u][ptr[u]]
                    v, cap, rev = e
                    if level[v] == level[u] + 1 and cap > 0:
                        tr = dfs(v, min(pushed, cap))
                        if tr > 0:
                            e[1] -= tr
                            flow_adj[v][rev][1] += tr
                            return tr
                    ptr[u] += 1
                return 0
                
            while bfs():
                for i in range(total_nodes):
                    ptr[i] = 0
                while True:
                    pushed = dfs(S, INF)
                    if pushed == 0:
                        break
                    flow += pushed
            return flow
            
        dinic()
        
        # Find reachable nodes from S in residual graph
        reachable = [False] * total_nodes
        q = deque([S])
        reachable[S] = True
        while q:
            u = q.popleft()
            for e in flow_adj[u]:
                v, cap, rev = e
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)
                    
        # Min vertex cover:
        #   - Nodes in color 0 that are NOT reachable (on sink side)
        #   - Nodes in color 1 that ARE reachable (on source side)
        # Max independent set = all nodes minus min vertex cover
        
        in_cover = [False] * M
        for i in range(M):
            if color[i] == 0:
                if not reachable[i]:
                    in_cover[i] = True
            else:
                if reachable[i]:
                    in_cover[i] = True
                    
        result = [values[i] for i in range(M) if not in_cover[i]]
        return sorted(result)
        
    else:
        # Fallback for non-bipartite: use greedy heuristic
        # Start with all valid nodes, remove conflicts greedily
        selected = set(valid_nodes)
        
        changed = True
        while changed:
            changed = False
            # Find node with highest degree in induced subgraph
            deg = {v: 0 for v in selected}
            for u in selected:
                for v in adj[node_index[u]]:
                    if values[v] in selected:
                        deg[u] += 1
            if deg:
                max_node = max(deg, key=deg.get)
                if deg[max_node] > 0:
                    selected.remove(max_node)
                    changed = True
                    
        # Try to add back removed nodes
        improved = True
        while improved:
            improved = False
            for v in valid_nodes:
                if v in selected:
                    continue
                ok = True
                for nb in adj[node_index[v]]:
                    if values[nb] in selected:
                        ok = False
                        break
                if ok:
                    selected.add(v)
                    improved = True
                    
        return sorted(selected)

# EVOLVE-BLOCK-END
