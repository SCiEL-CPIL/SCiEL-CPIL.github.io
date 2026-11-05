"""Improved program using DFS with enhanced branch-and-bound for longest path."""

def construct():
    import time
    import sys
    sys.setrecursionlimit(10000)
    
    n = 100
    edges = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    
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
        if len(comp) <= len(best_path):
            continue
        
        comp_set = set(comp)
        comp_adj = {v: adj[v] & comp_set for v in comp}
        
        # Check if it's a tree (edges = vertices - 1)
        edge_count = sum(len(comp_adj[v]) for v in comp) // 2
        is_tree = (edge_count == len(comp) - 1)
        
        if is_tree:
            # For a tree, longest path = diameter
            from collections import deque
            
            # BFS from arbitrary node to find farthest
            start = comp[0]
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([start])
            farthest = start
            while queue:
                node = queue.popleft()
                if dist[node] > dist[farthest]:
                    farthest = node
                for nei in comp_adj[node]:
                    if nei not in dist:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
            
            # BFS from farthest to find the other end and reconstruct path
            dist2 = {farthest: 0}
            parent2 = {farthest: -1}
            queue = deque([farthest])
            farthest2 = farthest
            while queue:
                node = queue.popleft()
                if dist2[node] > dist2[farthest2]:
                    farthest2 = node
                for nei in comp_adj[node]:
                    if nei not in dist2:
                        dist2[nei] = dist[node] + 1
                        parent2[nei] = node
                        queue.append(nei)
            
            # Reconstruct path from farthest2 back to farthest
            path = []
            cur = farthest2
            while cur != -1:
                path.append(cur)
                cur = parent2[cur]
            path.reverse()
            
            if len(path) > len(best_path):
                best_path = path
        else:
            # Not a tree - use enhanced DFS with pruning
            # Precompute neighbors as sorted lists (by degree for better pruning)
            comp_neis = {}
            for v in comp:
                neighbors = sorted(comp_adj[v], key=lambda x: len(comp_adj[x]))
                comp_neis[v] = neighbors
            
            # Multiple greedy strategies to get a good initial lower bound
            def greedy_from(start):
                path = [start]
                seen = {start}
                while True:
                    candidates = [v for v in comp_neis[path[-1]] if v not in seen]
                    if not candidates:
                        break
                    # Pick candidate with fewest unvisited neighbors (most constrained)
                    v = min(candidates, key=lambda w: len([x for x in comp_neis[w] if x not in seen]))
                    path.append(v)
                    seen.add(v)
                return path
            
            def greedy_degree_from(start):
                path = [start]
                seen = {start}
                while True:
                    candidates = [v for v in comp_neis[path[-1]] if v not in seen]
                    if not candidates:
                        break
                    # Pick candidate with highest degree among unvisited
                    v = max(candidates, key=lambda w: len(comp_adj[w]))
                    path.append(v)
                    seen.add(v)
                return path
            
            greedy_paths = []
            for start in comp:
                greedy_paths.append(greedy_from(start))
                greedy_paths.append(greedy_degree_from(start))
            
            best_greedy = max(greedy_paths, key=len)
            best_local = best_greedy[:]
            
            # DFS with enhanced pruning
            # Key improvement: compute the size of the largest connected component
            # in the remaining graph to get a tighter upper bound
            def get_largest_remaining_component(last, mask):
                """Find the largest connected component reachable from 'last' in unvisited nodes."""
                # BFS from last through unvisited neighbors
                visited_local = set()
                queue = [last]
                visited_local.add(last)
                max_size = 0
                while queue:
                    node = queue.pop(0)
                    for nei in comp_neis[node]:
                        bit = 1 << nei
                        if not (mask & bit) and nei not in visited_local:
                            visited_local.add(nei)
                            queue.append(nei)
                return len(visited_local) - 1  # exclude last itself
            
            deadline = time.time() + 55  # Leave 5 seconds for other components
            
            def dfs(path, mask):
                nonlocal best_local
                curr_len = len(path)
                
                # Check time
                if time.time() > deadline:
                    return
                
                if curr_len > len(best_local):
                    best_local = path[:]
                
                last = path[-1]
                
                # Enhanced pruning: find largest reachable component
                # Only do this if we're close to the best (to save time)
                if curr_len + (len(comp) - mask.bit_count()) > len(best_local):
                    # Simple bound first
                    remaining = len(comp) - mask.bit_count()
                    if curr_len + remaining <= len(best_local):
                        return
                    
                    # Try to get tighter bound using connected component analysis
                    # But only if it's worth it (small remaining set)
                    if remaining <= 30:
                        max_reachable = get_largest_remaining_component(last, mask)
                        if curr_len + max_reachable <= len(best_local):
                            return
                
                for nei in comp_neis[last]:
                    bit = 1 << nei
                    if not (mask & bit):
                        path.append(nei)
                        dfs(path, mask | bit)
                        path.pop()
            
            # Try from each starting node, prioritized by degree
            starts = sorted(comp, key=lambda x: len(comp_adj[x]), reverse=True)
            for start in starts:
                if time.time() > deadline:
                    break
                dfs([start], 1 << start)
            
            if len(best_local) > len(best_path):
                best_path = best_local
    
    return best_path
