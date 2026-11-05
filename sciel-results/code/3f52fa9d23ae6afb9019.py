import time
import math
import random
import heapq
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Identify vertices that cannot be in the set (self-conflict: 2*v is square)
    forbidden = set()
    for v in vertices:
        if 2 * v in squares:
            forbidden.add(v)
            
    # The remaining vertices form a graph where edges exist if u+v is a square.
    # This graph is bipartite with respect to the "square sum" property?
    # Actually, the graph of "sum is square" on integers is NOT necessarily bipartite globally,
    # but let's check the structure.
    # Wait, the standard trick for "sum is square" is that it relates to parity?
    # 1+3=4 (sq), 3+6=9 (sq), 6+10=16 (sq). 1,3,6,10...
    # Let's just build the graph on the allowed vertices and check if it's bipartite.
    # If it is, we can solve Max Independent Set exactly via Max Flow (Min Cut).
    # If not, we fall back to a heuristic or decompose.
    
    allowed = [v for v in vertices if v not in forbidden]
    
    # Build adjacency list for allowed vertices
    # Map vertex value to index in allowed list for faster lookup?
    # Or just use sets.
    adj = {v: set() for v in allowed}
    
    for i, u in enumerate(allowed):
        for v in allowed[i+1:]:
            if u + v in squares:
                adj[u].add(v)
                adj[v].add(u)
                
    # Check if the graph is bipartite
    # Use BFS/DFS coloring
    color = {}
    is_bipartite = True
    for v in allowed:
        if v in color:
            continue
        color[v] = 0
        queue = deque([v])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                else:
                    if color[neighbor] == color[node]:
                        is_bipartite = False
                        break
            if not is_bipartite:
                break
        if not is_bipartite:
            break
            
    best_set = set()
    
    if is_bipartite:
        # The graph is bipartite. Max Independent Set in a bipartite graph
        # can be found via Max Flow / Min Cut.
        # Specifically, MIS = V - Min Vertex Cover.
        # Min Vertex Cover in bipartite graph = Max Matching (Konig's Theorem).
        # We can compute Max Matching using Hopcroft-Karp or just Edmonds-Karp (Max Flow).
        # Given N=256, Edmonds-Karp is O(VE^2) which is fine.
        
        # Let's implement a simple Max Flow (Edmonds-Karp) to find Max Matching.
        # Left set: color 0
        # Right set: color 1
        
        left_nodes = [v for v in allowed if color[v] == 0]
        right_nodes = [v for v in allowed if color[v] == 1]
        
        # Map to indices for flow network
        # Source s, Sink t
        # Nodes: s, left_indices, right_indices, t
        
        # We'll build the flow graph dynamically or use a matrix.
        # Since N is small, a matrix is fine.
        
        num_left = len(left_nodes)
        num_right = len(right_nodes)
        
        # Index mapping
        left_idx = {v: i for i, v in enumerate(left_nodes)}
        right_idx = {v: i for i, v in enumerate(right_nodes)}
        
        # Flow graph nodes:
        # 0: Source
        # 1..num_left: Left nodes
        # num_left+1..num_left+num_right: Right nodes
        # num_left+num_right+1: Sink
        
        V_flow = num_left + num_right + 2
        s = 0
        t = V_flow - 1
        
        # Capacity matrix
        cap = [[0] * V_flow for _ in range(V_flow)]
        
        # Edges from Source to Left
        for i in range(num_left):
            cap[s][1 + i] = 1
            
        # Edges from Right to Sink
        for j in range(num_right):
            cap[1 + num_left + j][t] = 1
            
        # Edges from Left to Right (if edge exists in original graph)
        for u in left_nodes:
            i = left_idx[u]
            for v in adj[u]:
                if v in right_idx:
                    j = right_idx[v]
                    cap[1 + i][1 + num_left + j] = 1
                    
        # Edmonds-Karp Algorithm
        def bfs_flow():
            # Returns path if found, else None
            parent = [-1] * V_flow
            visited = [False] * V_flow
            queue = deque([s])
            visited[s] = True
            
            while queue:
                u = queue.popleft()
                if u == t:
                    break
                for v in range(V_flow):
                    if not visited[v] and cap[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        queue.append(v)
                        
            if not visited[t]:
                return None
            
            # Reconstruct path and find bottleneck
            path = []
            v = t
            while v != s:
                u = parent[v]
                path.append((u, v))
                v = u
                
            bottleneck = float('inf')
            for u, v in path:
                if cap[u][v] < bottleneck:
                    bottleneck = cap[u][v]
                    
            return path, bottleneck
            
        flow = 0
        while True:
            result = bfs_flow()
            if result is None:
                break
            path, bottleneck = result
            for u, v in path:
                cap[u][v] -= bottleneck
                cap[v][u] += bottleneck
            flow += bottleneck
            
        # Now we have the Max Flow (which equals Max Matching size).
        # The Min Cut partitions the graph into S (reachable from s in residual) and T.
        # The Min Vertex Cover is (Left \ S) U (Right \cap S) ??
        # Let's recall the construction:
        # Left nodes in S, Right nodes in T -> Edge is cut? No.
        # Standard construction:
        # Edge (u, v) in matching if u in Left\S and v in Right\S? 
        # Actually, let's just find the Min Vertex Cover directly from the residual graph.
        
        # Find reachable nodes from s in residual graph
        visited = [False] * V_flow
        queue = deque([s])
        visited[s] = True
        while queue:
            u = queue.popleft()
            for v in range(V_flow):
                if not visited[v] and cap[u][v] > 0:
                    visited[v] = True
                    queue.append(v)
                    
        # Min Vertex Cover C:
        # C contains Left nodes NOT in S (i.e., not visited)
        # C contains Right nodes IN S (i.e., visited)
        
        min_vertex_cover = set()
        
        # Left nodes not visited
        for i in range(num_left):
            if not visited[1 + i]:
                min_vertex_cover.add(left_nodes[i])
                
        # Right nodes visited
        for j in range(num_right):
            if visited[1 + num_left + j]:
                min_vertex_cover.add(right_nodes[j])
                
        # Max Independent Set = V - Min Vertex Cover
        best_set = set(allowed) - min_vertex_cover
        
    else:
        # If not bipartite, we fall back to a heuristic approach.
        # Given the structure of square sums, it's likely bipartite or close to it.
        # If not, we can use a greedy approach or simulated annealing on the allowed set.
        # For now, let's use a simple greedy with random restarts to ensure we return something valid.
        
        # Greedy: Pick vertices with lowest degree first? Or highest?
        # Usually, for MIS, picking low degree first is good.
        
        def greedy_mis():
            remaining = set(allowed)
            result = set()
            # Sort by degree in induced subgraph? 
            # To be fast, just pick random or lowest degree.
            # Let's try lowest degree.
            while remaining:
                if time.time() - start_time > DEADLINE:
                    break
                # Find vertex with min degree in remaining
                min_deg = float('inf')
                min_v = None
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                if min_v is None:
                    break
                result.add(min_v)
                # Remove min_v and its neighbors
                remaining.discard(min_v)
                for u in adj[min_v]:
                    if u in remaining:
                        remaining.discard(u)
            return result
            
        best_set = greedy_mis()
        
        # Try to improve with 1-swap?
        # Simple local search
        current_set = set(best_set)
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            # Try to add
            candidates = [v for v in allowed if v not in current_set]
            random.shuffle(candidates)
            for c in candidates:
                if all((c + v) not in squares for v in current_set):
                    current_set.add(c)
                    improved = True
                    break
            if improved:
                continue
            # Try to swap: remove one, add two?
            # This is complex. Let's just try adding for now.
            # If no add, try removing one to allow two adds?
            # For simplicity, if we are stuck, we accept current.
            
        best_set = current_set
        
    # Final verification
    best_list = list(best_set)
    valid = True
    for i in range(len(best_list)):
        a = best_list[i]
        for j in range(i, len(best_list)):
            b = best_list[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set: v % 3 == 1
        # 1, 4, 7...
        # 1+1=2 (no), 1+4=5 (no), 1+7=8 (no), 4+4=8 (no), 4+7=11 (no)...
        # Let's verify v%3==1 is valid.
        # Squares mod 3 are 0, 1.
        # If a,b are 1 mod 3, a+b is 2 mod 3.
        # 2 is not a square mod 3.
        # So any set of numbers congruent to 1 mod 3 is valid.
        # Size: 256 // 3 = 85.
        return list(range(1, 257, 3))
        
    return best_list
