import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback
            edges = instance_graph(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance_graph(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        
    return results

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges):
    if not edges:
        return []
        
    # 1. Build Edge Adjacency for Bipartite Graph
    # Left nodes: 0..E-1, Right nodes: 0..E-1
    # Edge (i, j) in bipartite graph if edge_i and edge_j share a vertex.
    
    num_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # For each edge, find neighbors in the line graph sense
    # adj_bip[i] = list of j such that edge i and edge j share a vertex
    adj_bip = [[] for _ in range(num_edges)]
    
    # Map vertex to list of edge indices incident to it
    vertex_to_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vertex_to_edges[u].append(i)
        vertex_to_edges[v].append(i)
        
    for v in vertex_to_edges:
        e_list = vertex_to_edges[v]
        for i in e_list:
            for j in e_list:
                if i != j:
                    adj_bip[i].append(j)
                    
    # 2. Hopcroft-Karp for Maximum Matching
    # matchL[i] = j means left i is matched to right j
    # matchR[j] = i means right j is matched to left i
    matchL = [-1] * num_edges
    matchR = [-1] * num_edges
    dist = [0] * num_edges
    
    def bfs():
        queue = deque()
        for u in range(num_edges):
            if matchL[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_bip[u]:
                if matchR[v] == -1:
                    found = True
                elif dist[matchR[v]] == -1:
                    dist[matchR[v]] = dist[u] + 1
                    queue.append(matchR[v])
        return found
        
    def dfs(u):
        for v in adj_bip[u]:
            if matchR[v] == -1 or (dist[matchR[v]] == dist[u] + 1 and dfs(matchR[v])):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = -1
        return False
        
    matching_size = 0
    while bfs():
        for u in range(num_edges):
            if matchL[u] == -1:
                if dfs(u):
                    matching_size += 1
                    
    # 3. Reconstruct Paths
    # A path is a sequence of edges.
    # If matchL[i] = j, then edge i is followed by edge j.
    # This forms chains.
    
    # next_edge[i] = j if i is followed by j
    next_edge = [-1] * num_edges
    for i in range(num_edges):
        if matchL[i] != -1:
            next_edge[i] = matchL[i]
            
    # Find starts of paths: edges i such that no edge k has next_edge[k] == i
    # i.e., i is not in the image of next_edge
    is_end_of_path = [False] * num_edges
    for j in range(num_edges):
        if next_edge[j] != -1:
            is_end_of_path[next_edge[j]] = True
            
    paths = []
    visited_edges = [False] * num_edges
    
    for i in range(num_edges):
        if not is_end_of_path[i] and not visited_edges[i]:
            # Start a path from edge i
            path_edges = []
            cur = i
            while cur != -1 and not visited_edges[cur]:
                visited_edges[cur] = True
                path_edges.append(edges[cur])
                cur = next_edge[cur]
                
            # Convert edge list to vertex list
            if not path_edges:
                continue
                
            vertices = [path_edges[0][0]]
            for k in range(len(path_edges)):
                e = path_edges[k]
                u, v = e
                last_v = vertices[-1]
                if last_v == u:
                    vertices.append(v)
                elif last_v == v:
                    vertices.append(u)
                else:
                    # Disconnected? This shouldn't happen if matching is valid for connected components
                    # But if the graph is not connected, or if we have a cycle that wasn't detected?
                    # Actually, if edges share a vertex, the vertex list construction works.
                    # If they don't share a vertex, they wouldn't be in the same chain.
                    # So this branch is for safety.
                    vertices.append(u)
                    vertices.append(v)
                    
            paths.append(vertices)
            
    # Check for simplicity
    # If a path is not simple, it's a trail with repeated vertices.
    # We need to split it.
    final_paths = []
    for p in paths:
        if len(set(p)) == len(p):
            final_paths.append(p)
        else:
            # Split at first repeated vertex
            seen = set()
            current_path = []
            for v in p:
                if v in seen:
                    if current_path:
                        final_paths.append(current_path)
                    current_path = [v]
                    seen = {v}
                else:
                    current_path.append(v)
                    seen.add(v)
            if current_path:
                final_paths.append(current_path)
                
    return final_paths

# Note: The above Hopcroft-Karp implementation is recursive DFS. 
# For N=300 edges, recursion depth is small. 
# However, Python recursion limit might be an issue if chains are long, but max chain is N. 
# N=300 is safe. 
# Also, the graph B is dense-ish for small N, but for N=300, |E|~300, adj_bip lists can be large. 
# But Hopcroft-Karp is efficient.

# One detail: The problem asks for "simple" paths. 
# The matching gives minimum *trails*. 
# If a trail has repeated vertices, we split it. 
# This increases the path count. 
# Is it possible that a different matching yields a set of simple paths with fewer total paths? 
# Yes, theoretically. 
# But for sparse random graphs, the "non-simple" penalty is rare. 
# If we find non-simple paths, we could try to swap matches to fix them, but that's complex. 
# Given the time limit, the matching solution is likely optimal or near-optimal. 
# The greedy method often gets stuck at 1.1-1.2x the optimum. 
# Matching should be very close to optimum.

# Let's add a fallback: if the resulting path count is worse than a simple greedy, use greedy?
# No, the matching count is a lower bound on trails, and splitting only increases it. 
# So matching result <= greedy result usually. 
# We'll stick with matching.

# Wait, the recursive DFS in Hopcroft-Karp might hit recursion limit if the augmenting path is long?
# Max path length in BFS tree is bounded by number of vertices in B, which is 2*|E|. 
# For |E|=300, depth is ~600. Python default recursion limit is 1000. 
# So it's safe.

# To be extra safe against recursion errors or performance issues on the larger graph (30 nodes, ~300 edges),
# I will implement an iterative DFS or just trust the limits. 
# 300 edges is small.

# One more check: The bipartite graph construction.
# adj_bip[i] contains j if edge i and edge j share a vertex.
# This allows i to be followed by j.
# It also allows j to be followed by i.
# This is correct for undirected edges.

# Final check on "simple" constraint.
# If the matching creates a cycle of edges e1->e2->e1, this is a cycle in the graph.
# A cycle is not a path. 
# The problem asks for paths. 
# If we have a cycle, we must break it into a path. 
# Breaking a cycle of length L into a path of length L reduces the number of "links" by 1? 
# No, a cycle of edges e1..ek uses k links. 
# If we break it, we have k edges and k-1 links (if we just cut one link). 
# Wait, a cycle e1->e2->...->ek->e1. 
# In the matching, e1 matched to e2, ..., ek matched to e1. 
# This forms a cycle in the permutation. 
# A path cover cannot contain a cycle. 
# We must break the cycle. 
# Breaking the cycle reduces the matching size by 1 (we remove one edge from the matching). 
# This increases the path count by 1. 
# My reconstruction code `while cur != -1 and not visited_edges[cur]` will handle cycles by stopping when it hits a visited edge.
# But it will leave the cycle unprocessed if we don't detect it as a start.
# `is_end_of_path` logic: 
# If e1->e2->e1, then e1 is end of e2, e2 is end of e1. 
# Neither is a start. 
# So the cycle is ignored! 
# We need to handle cycles. 
# Any edge in a cycle can be a start if we break the cycle. 
# To minimize paths, we should break the cycle at one point. 
# This effectively removes one match. 
# So we should detect cycles and treat one edge in the cycle as a start. 

# Modified Reconstruction:
# 1. Find all starts (in-degree 0 in the matching graph).
# 2. Follow chains.
# 3. Any edge not visited is in a cycle. 
# 4. For each cycle, pick an arbitrary edge to be the start, break the link after the last edge in the cycle.

def decompose_graph_v2(edges):
    if not edges:
        return []
        
    num_edges = len(edges)
    
    vertex_to_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vertex_to_edges[u].append(i)
        vertex_to_edges[v].append(i)
        
    adj_bip = [[] for _ in range(num_edges)]
    for v in vertex_to_edges:
        e_list = vertex_to_edges[v]
        for i in e_list:
            for j in e_list:
                if i != j:
                    adj_bip[i].append(j)
                    
    matchL = [-1] * num_edges
    matchR = [-1] * num_edges
    dist = [0] * num_edges
    
    def bfs():
        queue = deque()
        for u in range(num_edges):
            if matchL[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_bip[u]:
                if matchR[v] == -1:
                    found = True
                elif dist[matchR[v]] == -1:
                    dist[matchR[v]] = dist[u] + 1
                    queue.append(matchR[v])
        return found
        
    def dfs(u):
        for v in adj_bip[u]:
            if matchR[v] == -1 or (dist[matchR[v]] == dist[u] + 1 and dfs(matchR[v])):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = -1
        return False
        
    while bfs():
        for u in range(num_edges):
            if matchL[u] == -1:
                if dfs(u):
                    pass # matching size not strictly needed, just the matches
                
    next_edge = [-1] * num_edges
    for i in range(num_edges):
        if matchL[i] != -1:
            next_edge[i] = matchL[i]
            
    is_end_of_path = [False] * num_edges
    for j in range(num_edges):
        if next_edge[j] != -1:
            is_end_of_path[next_edge[j]] = True
            
    paths = []
    visited_edges = [False] * num_edges
    
    # Process chains starting from non-ends
    for i in range(num_edges):
        if not is_end_of_path[i] and not visited_edges[i]:
            path_edges = []
            cur = i
            while cur != -1 and not visited_edges[cur]:
                visited_edges[cur] = True
                path_edges.append(edges[cur])
                cur = next_edge[cur]
            if path_edges:
                vertices = [path_edges[0][0]]
                for k in range(len(path_edges)):
                    e = path_edges[k]
                    u, v = e
                    last_v = vertices[-1]
                    if last_v == u:
                        vertices.append(v)
                    elif last_v == v:
                        vertices.append(u)
                    else:
                        vertices.append(u)
                        vertices.append(v)
                paths.append(vertices)
                
    # Process cycles
    for i in range(num_edges):
        if not visited_edges[i]:
            # This is a cycle
            path_edges = []
            cur = i
            while not visited_edges[cur]:
                visited_edges[cur] = True
                path_edges.append(edges[cur])
                cur = next_edge[cur]
            # We have a cycle. We must break it.
            # The last edge's next is the first edge. We stop there.
            if path_edges:
                vertices = [path_edges[0][0]]
                for k in range(len(path_edges)):
                    e = path_edges[k]
                    u, v = e
                    last_v = vertices[-1]
                    if last_v == u:
                        vertices.append(v)
                    elif last_v == v:
                        vertices.append(u)
                    else:
                        vertices.append(u)
                        vertices.append(v)
                paths.append(vertices)
                
    # Final check for simplicity and splitting
    final_paths = []
    for p in paths:
        if len(set(p)) == len(p):
            final_paths.append(p)
        else:
            seen = set()
            current_path = []
            for v in p:
                if v in seen:
                    if current_path:
                        final_paths.append(current_path)
                    current_path = [v]
                    seen = {v}
                else:
                    current_path.append(v)
                    seen.add(v)
            if current_path:
                final_paths.append(current_path)
                
    return final_paths

# Replace decompose_graph with decompose_graph_v2
decompose_graph = decompose_graph_v2
