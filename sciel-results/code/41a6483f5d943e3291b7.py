"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

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
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    best_path = []
    
    for comp in components:
        if not comp:
            continue
            
        # Check if it's a tree
        num_nodes = len(comp)
        num_edges = sum(len(adj[u] & set(comp)) for u in comp) // 2
        
        if num_edges == num_nodes - 1:
            # It's a tree. Find diameter.
            # BFS from arbitrary node to find farthest
            def bfs_farthest(start):
                dist = {start: 0}
                q = [start]
                farthest = start
                max_d = 0
                while q:
                    u = q.pop(0)
                    if dist[u] > max_d:
                        max_d = dist[u]
                        farthest = u
                    for v in adj[u]:
                        if v in set(comp) and v not in dist:
                            dist[v] = dist[u] + 1
                            q.append(v)
                return farthest, dist
                
            start = comp[0]
            far1, _ = bfs_farthest(start)
            far2, dist2 = bfs_farthest(far1)
            
            # Reconstruct path
            # We need the actual path, not just the length.
            # Let's do a BFS from far2 to get parent pointers or just DFS back.
            # Since it's a tree, we can just do a DFS from far1 to far2.
            path = []
            visited_tree = set()
            def dfs_tree(u, target):
                if u == target:
                    return [u]
                visited_tree.add(u)
                for v in adj[u]:
                    if v in set(comp) and v not in visited_tree:
                        res = dfs_tree(v, target)
                        if res:
                            return [u] + res
                return None
                
            path = dfs_tree(far1, far2)
            if path and len(path) > len(best_path):
                best_path = path
        else:
            # It has cycles. Use DFS to find longest path.
            # Since components are small, this is feasible.
            # We can try starting from every node in the component.
            comp_set = set(comp)
            
            def dfs_longest(current, visited_nodes, path):
                nonlocal best_path
                if len(path) > len(best_path):
                    best_path = path[:]
                
                for nei in adj[current]:
                    if nei in comp_set and nei not in visited_nodes:
                        visited_nodes.add(nei)
                        path.append(nei)
                        dfs_longest(nei, visited_nodes, path)
                        path.pop()
                        visited_nodes.remove(nei)
                        
            # Optimization: Only start DFS from nodes that might lead to long paths?
            # Or just run from all nodes.
            # To avoid redundant work, we can just run from all nodes.
            # Given small size, it's fine.
            for start_node in comp:
                dfs_longest(start_node, {start_node}, [start_node])
                
    return best_path
# EVOLVE-BLOCK-END
