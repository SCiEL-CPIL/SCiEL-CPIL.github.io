import random
import time

def construct():
    # Pinned graph data from the prompt
    P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    best_path = []
    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    for comp in components:
        if time.time() - start_time > time_limit:
            break
            
        # Check if component is a tree
        # A tree with V vertices has V-1 edges
        edge_count = sum(len(adj[u]) for u in comp) // 2
        is_tree = (edge_count == len(comp) - 1)
        
        if is_tree:
            # Calculate diameter
            if len(comp) == 1:
                path = [comp[0]]
            else:
                # BFS from arbitrary node to find farthest node A
                def bfs_farthest(start, comp_set):
                    dist = {start: 0}
                    queue = [start]
                    farthest = start
                    max_dist = 0
                    while queue:
                        u = queue.pop(0)
                        for v in adj[u]:
                            if v in comp_set and v not in dist:
                                dist[v] = dist[u] + 1
                                queue.append(v)
                                if dist[v] > max_dist:
                                    max_dist = dist[v]
                                    farthest = v
                    return farthest, dist
                
                comp_set = set(comp)
                # Pick arbitrary start
                start_node = comp[0]
                # Find farthest node A from start
                node_a, dist_a = bfs_farthest(start_node, comp_set)
                # Find farthest node B from A, and reconstruct path
                node_b, dist_b = bfs_farthest(node_a, comp_set)
                
                # Reconstruct path from node_a to node_b
                # We need parent pointers. Let's do a BFS that stores parents.
                def bfs_path(start, target, comp_set):
                    if start == target:
                        return [start]
                    parent = {start: None}
                    queue = [start]
                    while queue:
                        u = queue.pop(0)
                        if u == target:
                            break
                        for v in adj[u]:
                            if v in comp_set and v not in parent:
                                parent[v] = u
                                queue.append(v)
                    
                    # Reconstruct
                    path = []
                    curr = target
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    return path
                
                path = bfs_path(node_a, node_b, comp_set)
            
            if len(path) > len(best_path):
                best_path = path
        else:
            # Heuristic search for non-tree component
            # Use randomized greedy (Warnsdorff's rule)
            # Run multiple iterations
            iterations = 1000
            for _ in range(iterations):
                if time.time() - start_time > time_limit:
                    break
                
                # Random start
                start_node = random.choice(comp)
                path = [start_node]
                visited_nodes = {start_node}
                
                while True:
                    current = path[-1]
                    unvisited_neighbors = [v for v in adj[current] if v not in visited_nodes and v in set(comp)]
                    
                    if not unvisited_neighbors:
                        break
                    
                    # Warnsdorff's rule: pick neighbor with fewest unvisited neighbors
                    # Add randomness to tie-breaking
                    scored_neighbors = []
                    for v in unvisited_neighbors:
                        # Count unvisited neighbors of v
                        unvisited_count = sum(1 for w in adj[v] if w not in visited_nodes and w in set(comp))
                        # Add small random noise to break ties
                        noise = random.random() * 0.1
                        scored_neighbors.append((unvisited_count + noise, v))
                    
                    scored_neighbors.sort(key=lambda x: x[0])
                    next_node = scored_neighbors[0][1]
                    
                    path.append(next_node)
                    visited_nodes.add(next_node)
                
                if len(path) > len(best_path):
                    best_path = path
                    
    return best_path
