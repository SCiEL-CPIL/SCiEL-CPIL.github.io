import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    
    # Precompute squares up to 2*N
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter nodes that violate the diagonal condition (a=a => 2a is square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict graph: edge between i,j if valid_nodes[i] + valid_nodes[j] is a square
    # Only consider i < j to avoid duplicates
    adj = [set() for _ in range(M)]
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Check if graph is bipartite and get 2-coloring
    color = [-1] * M  # -1 unvisited, 0 or 1 for colors
    is_bipartite = True
    for start in range(M):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # 4. Solve MWIS via min-cut (all weights = 1)
        # Left nodes: color 0, Right nodes: color 1
        # Source -> left (cap 1), left -> right (cap INF) for edges, right -> sink (cap 1)
        
        INF = 10**9
        # Nodes: 0=source, 1..M = graph nodes, M+1=sink
        num_flow_nodes = M + 2
        source = 0
        sink = M + 1
        
        # Build residual graph for Edmonds-Karp
        # Using adjacency list with (to, capacity)
        residual = [[] for _ in range(num_flow_nodes)]
        # parent[node] = (prev_node, capacity) for augmenting path
        
        def add_edge(u, v, cap):
            residual[u].append([v, cap, len(residual[v])])
            residual[v].append([u, 0, len(residual[u]) - 1])
        
        # Add edges
        for i in range(M):
            if color[i] == 0:  # left
                add_edge(source, i + 1, 1)
            else:  # right
                add_edge(i + 1, sink, 1)
        
        for i in range(M):
            for j in adj[i]:
                if color[i] == 0 and color[j] == 1:
                    add_edge(i + 1, j + 1, INF)
        
        # Edmonds-Karp max flow
        def bfs_flow():
            # Find augmenting path via BFS
            parent = [-1] * num_flow_nodes
            parent[source] = source
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for edge in residual[u]:
                    v, cap, _ = edge
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == sink:
                            # Found path, augment
                            flow = INF
                            cur = sink
                            while cur != source:
                                prev = parent[cur]
                                for e in residual[prev]:
                                    if e[0] == cur:
                                        flow = min(flow, e[1])
                                        break
                                cur = prev
                            # Update residual
                            cur = sink
                            while cur != source:
                                prev = parent[cur]
                                for e in residual[prev]:
                                    if e[0] == cur:
                                        e[1] -= flow
                                        residual[cur][e[2]][1] += flow
                                        break
                                cur = prev
                            return flow
                        queue.append(v)
            return 0
        
        max_flow = 0
        while True:
            f = bfs_flow()
            if f == 0:
                break
            max_flow += f
            if time.time() > deadline - 5:
                break
        
        min_vertex_cover_size = max_flow
        # Find the independent set: 
        # After max flow, do BFS from source in residual graph
        # S = reachable from source, T = not reachable
        # Independent set = (left \ S) ∪ (right ∩ S)
        # i.e., left nodes not reachable + right nodes reachable
        
        reachable = [False] * num_flow_nodes
        reachable[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in residual[u]:
                v, cap, _ = edge
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    queue.append(v)
        
        independent_set = []
        for i in range(M):
            node_in_flow = i + 1
            if color[i] == 0:  # left
                if not reachable[node_in_flow]:
                    independent_set.append(valid_nodes[i])
            else:  # right
                if reachable[node_in_flow]:
                    independent_set.append(valid_nodes[i])
        
        if len(independent_set) > 0:
            return independent_set
    
    # Fallback: greedy + local search if not bipartite or timeout
    # Greedy: pick nodes with fewest conflicts first
    if time.time() > deadline - 2:
        # Return a simple valid set
        result = []
        for v in valid_nodes:
            valid = True
            for u in result:
                if u + v in squares:
                    valid = False
                    break
            if valid:
                result.append(v)
        if result:
            return result
        return [1]
    
    # Local search: start with greedy, then try swaps
    # Greedy initialization
    result = set()
    remaining = list(valid_nodes)
    # Sort by degree (number of conflicts)
    degrees = [len(adj[i]) for i in range(M)]
    remaining.sort(key=lambda x: degrees[node_to_idx[x]])
    
    for v in remaining:
        vi = node_to_idx[v]
        can_add = True
        for u in result:
            if vi in adj[node_to_idx[u]]:
                can_add = False
                break
        if can_add:
            result.add(v)
    
    # Local search: try to add nodes, then try swaps
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        # Try adding
        for v in valid_nodes:
            if v not in result:
                vi = node_to_idx[v]
                can_add = True
                for u in result:
                    if vi in adj[node_to_idx[u]]:
                        can_add = False
                        break
                if can_add:
                    result.add(v)
                    improved = True
                    break
        # Try swaps: remove one, add two
        if not improved:
            for remove in list(result):
                ri = node_to_idx[remove]
                temp_result = result - {remove}
                for v1 in valid_nodes:
                    if v1 in temp_result:
                        continue
                    v1i = node_to_idx[v1]
                    can_add_v1 = True
                    for u in temp_result:
                        if v1i in adj[node_to_idx[u]]:
                            can_add_v1 = False
                            break
                    if not can_add_v1:
                        continue
                    temp_result2 = temp_result | {v1}
                    for v2 in valid_nodes:
                        if v2 in temp_result2:
                            continue
                        v2i = node_to_idx[v2]
                        can_add_v2 = True
                        for u in temp_result2:
                            if v2i in adj[node_to_idx[u]]:
                                can_add_v2 = False
                                break
                        if not can_add_v2:
                            continue
                        if len(temp_result2) > len(result):
                            result = temp_result2 | {v2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return list(result)
# EVOLVE-BLOCK-END
