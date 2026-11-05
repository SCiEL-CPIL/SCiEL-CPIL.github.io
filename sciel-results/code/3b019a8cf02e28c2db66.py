import sys
import time
import random
import collections

# EVOLVE-BLOCK-START
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
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 80.0  # Safety margin
    
    # Function to find longest path in a component using DFS
    def find_longest_path_in_component(comp):
        nonlocal best_path
        if not comp:
            return
        
        comp_set = set(comp)
        
        # Optimization: If the component is a tree, compute diameter
        # Count edges in component
        edge_count = 0
        for u in comp:
            edge_count += len(adj[u] & comp_set)
        edge_count //= 2
        
        if edge_count == len(comp) - 1:
            # It's a tree. Find diameter.
            # BFS from arbitrary node to find farthest node u
            def bfs_farthest(start):
                dist = {start: 0}
                q = collections.deque([start])
                farthest = start
                while q:
                    node = q.popleft()
                    for nei in adj[node]:
                        if nei in comp_set and nei not in dist:
                            dist[nei] = dist[node] + 1
                            q.append(nei)
                            if dist[nei] > dist[farthest]:
                                farthest = nei
                return farthest, dist
                
            u, _ = bfs_farthest(comp[0])
            v, dist_v = bfs_farthest(u)
            # Reconstruct path from u to v
            # We need parent pointers
            def bfs_path(start, target):
                prev = {start: None}
                q = collections.deque([start])
                while q:
                    node = q.popleft()
                    if node == target:
                        break
                    for nei in adj[node]:
                        if nei in comp_set and nei not in prev:
                            prev[nei] = node
                            q.append(nei)
                path = []
                curr = target
                while curr is not None:
                    path.append(curr)
                    curr = prev[curr]
                return path[::-1]
            
            path = bfs_path(u, v)
            if len(path) > len(best_path):
                best_path = path
            return

        # If not a tree, use DFS with backtracking
        # Use iterative DFS with explicit stack to avoid recursion limits and allow better control
        
        def dfs_iterative(start):
            nonlocal best_path
            if time.time() > deadline:
                return
            
            # State: (node, path, visited_set, next_neighbor_index)
            # We'll use a stack of states
            # For each node, we need to track which neighbors we've tried
            
            # Initial state
            path = [start]
            visited_set = {start}
            
            # Stack entries: (node, path_copy, visited_copy, neighbor_list, neighbor_idx)
            # To save memory, we can use references, but for safety with backtracking, copies are needed
            # However, N=100, so copies are fine.
            
            # Pre-compute sorted neighbor lists for each node in comp
            # Sort by degree descending to find long paths faster
            sorted_neighbors = {}
            for node in comp:
                neis = [nei for nei in adj[node] if nei in comp_set]
                neis.sort(key=lambda x: len(adj[x]), reverse=True)
                sorted_neighbors[node] = neis
            
            stack = [(start, path, visited_set, sorted_neighbors[start], 0)]
            
            while stack:
                if time.time() > deadline:
                    return
                
                node, path, visited_set, neis, idx = stack[-1]
                
                # Check if current path is better
                if len(path) > len(best_path):
                    best_path = path[:]
                
                # Pruning: if current length + number of unvisited nodes in comp <= best_len, return
                if len(path) + (len(comp) - len(visited_set)) <= len(best_path):
                    stack.pop()
                    continue
                
                # Find next unvisited neighbor
                found_next = False
                while idx < len(neis):
                    nei = neis[idx]
                    idx += 1
                    if nei not in visited_set:
                        found_next = True
                        break
                
                if found_next:
                    # Update stack top with new idx
                    stack[-1] = (node, path, visited_set, neis, idx)
                    # Push new state
                    new_path = path + [nei]
                    new_visited = visited_set | {nei}
                    stack.append((nei, new_path, new_visited, sorted_neighbors[nei], 0))
                else:
                    # No more neighbors, backtrack
                    stack.pop()
            
        # Run DFS from each node in the component
        for start in comp:
            if time.time() > deadline:
                break
            dfs_iterative(start)
            
    # Process components
    for comp in components:
        if time.time() > deadline:
            break
        find_longest_path_in_component(comp)
        
    # If best_path is empty (shouldn't happen), return any edge
    if not best_path:
        if edges:
            return [edges[0][0], edges[0][1]]
        else:
            return [0]
            
    return best_path
# EVOLVE-BLOCK-END
