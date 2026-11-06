import sys
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Ensure all nodes are in adj
    for i in range(n):
        if i not in adj:
            adj[i] = set()

    # Find connected components
    visited = set()
    components = []
    
    for i in range(n):
        if i not in visited:
            comp = []
            stack = [i]
            visited.add(i)
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(comp)
            
    best_path = []
    deadline = time.time() + 50  # Safety margin
    
    for comp in components:
        if time.time() > deadline:
            break
            
        if len(comp) == 1:
            if len(comp) > len(best_path):
                best_path = comp
            continue
            
        # Check if it's a tree
        edge_count = 0
        for u in comp:
            edge_count += len(adj[u])
        edge_count //= 2
        
        if edge_count == len(comp) - 1:
            # It's a tree. Find diameter.
            # 1. BFS from arbitrary node to find farthest
            start = comp[0]
            def bfs_farthest(start_node):
                dist = {start_node: 0}
                parent = {start_node: None}
                q = deque([start_node])
                farthest = start_node
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                            if dist[v] > dist[farthest]:
                                farthest = v
                return farthest, dist, parent
                
            end1, _, _ = bfs_farthest(start)
            end2, dist, parent = bfs_farthest(end1)
            
            # Reconstruct path from end1 to end2
            path = [end2]
            curr = end2
            while curr != end1:
                curr = parent[curr]
                path.append(curr)
            
            if len(path) > len(best_path):
                best_path = path
        else:
            # It's not a tree (unicyclic or more). 
            # Given the graph parameters, these are small.
            # We can use a simple DFS to find the longest path in this small component.
            # Since components are small (likely < 20 nodes), this is fast.
            
            comp_set = set(comp)
            
            def dfs_longest(node, visited_in_path, current_path):
                nonlocal best_path, deadline
                if time.time() > deadline:
                    return
                if len(current_path) > len(best_path):
                    best_path = list(current_path)
                    
                for neighbor in adj[node]:
                    if neighbor in comp_set and neighbor not in visited_in_path:
                        visited_in_path.add(neighbor)
                        current_path.append(neighbor)
                        dfs_longest(neighbor, visited_in_path, current_path)
                        current_path.pop()
                        visited_in_path.remove(neighbor)
            
            # Start DFS from each node in the component to ensure we find the global max
            # Optimization: Only need to start from nodes that could be endpoints?
            # For small graphs, just try all starts.
            for start_node in comp:
                dfs_longest(start_node, {start_node}, [start_node])
                
    return best_path
# EVOLVE-BLOCK-END
