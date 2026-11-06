"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    
    # Find the largest component
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    if comp_size <= 2:
        return largest_comp

    # Check if the component is a tree
    edge_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                edge_count += 1
                
    is_tree = (edge_count == comp_size - 1)
    
    if is_tree:
        # Find diameter of the tree
        def bfs(far_from):
            dist = {far_from: 0}
            parent = {far_from: None}
            queue = deque([far_from])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor in comp_set and neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
            u = max(dist, key=dist.get)
            return u, dist, parent
            
        # First BFS
        u, dist1, parent1 = bfs(largest_comp[0])
        # Second BFS from u
        v, dist2, parent2 = bfs(u)
        
        # Reconstruct path from u to v
        path = [v]
        while path[-1] != u:
            path.append(parent2[path[-1]])
        path.reverse()
        
        return path

    # If not a tree, use heuristic DFS
    # Identify leaves
    leaves = [v for v in largest_comp if len(adj[v] & comp_set) == 1]
    if not leaves:
        leaves = largest_comp
        
    best_path = []
    rng = random.Random(748)
    
    def greedy_dfs(start):
        path = [start]
        used = {start}
        cur = start
        while True:
            candidates = [v for v in adj[cur] if v in comp_set and v not in used]
            if not candidates:
                break
            # Warnsdorff: prefer nodes with fewest unvisited neighbors
            candidates.sort(key=lambda v: (len(adj[v] & comp_set - used), rng.random()))
            nxt = candidates[0]
            path.append(nxt)
            used.add(nxt)
            cur = nxt
        return path

    # Try from each leaf
    for leaf in leaves:
        p = greedy_dfs(leaf)
        if len(p) > len(best_path):
            best_path = p
            if len(best_path) == comp_size:
                return best_path
                
    # If leaves didn't give a full path, try random nodes
    for _ in range(20):
        start = rng.choice(largest_comp)
        p = greedy_dfs(start)
        if len(p) > len(best_path):
            best_path = p
            if len(best_path) == comp_size:
                return best_path

    # Verify validity
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
             return [best_path[0]]
             
    return best_path
# EVOLVE-BLOCK-END
