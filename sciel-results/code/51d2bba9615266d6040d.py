import sys
import time
import random

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
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # For each component, find the longest path
    for comp in components:
        if time.time() - start_time > time_limit:
            break
            
        if len(comp) == 0:
            continue
            
        # Exact search with connectivity pruning for components up to size 30
        # This is a structural change: using connectivity pruning to avoid local optima
        if len(comp) <= 30:
            comp_set = set(comp)
            best = []
            
            # Precompute adjacency within component
            comp_adj = {node: [nei for nei in adj[node] if nei in comp_set] for node in comp}
            
            # Warnsdorff's rule: order neighbors by degree (fewest options first)
            def warnsdorff_order(node, visited_set):
                neighbors = [nei for nei in comp_adj[node] if nei not in visited_set]
                # Sort by number of available neighbors (fewest first)
                neighbors.sort(key=lambda x: len([n for n in comp_adj[x] if n not in visited_set]))
                return neighbors
            
            # DFS with connectivity pruning
            def dfs(current, path, visited_set):
                nonlocal best
                if time.time() - start_time > time_limit:
                    return
                if len(path) > len(best):
                    best = path[:]
                
                # Connectivity pruning: check if remaining unvisited nodes are connected
                # to current node's unvisited neighbors
                unvisited = comp_set - visited_set
                if not unvisited:
                    return
                
                # Check if any unvisited node is still reachable from current
                # through unvisited nodes
                reachable = set()
                stack = [current]
                visited_reach = {current}
                while stack:
                    node = stack.pop()
                    for nei in comp_adj[node]:
                        if nei in unvisited and nei not in visited_reach:
                            visited_reach.add(nei)
                            stack.append(nei)
                
                # If no unvisited nodes are reachable, we can't extend further
                if not visited_reach & unvisited:
                    return
                
                # Try each neighbor in Warnsdorff order
                for nei in warnsdorff_order(current, visited_set):
                    dfs(nei, path + [nei], visited_set | {nei})
                    if time.time() - start_time > time_limit:
                        return
            
            # Try each node as start, prioritizing nodes with higher degree
            starts = sorted(comp, key=lambda x: len(comp_adj[x]), reverse=True)
            for start in starts:
                if time.time() - start_time > time_limit:
                    break
                dfs(start, [start], {start})
            
            if len(best) > len(best_path):
                best_path = best
        else:
            # For larger components, use Warnsdorff-guided greedy with random restarts
            rng = random.Random(42)
            best_local = []
            
            for restart in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Random start
                start = rng.choice(comp)
                path = [start]
                seen = {start}
                
                while True:
                    current = path[-1]
                    unvisited_neighbors = [nei for nei in adj[current] if nei not in seen]
                    if not unvisited_neighbors:
                        break
                    
                    # Warnsdorff: choose neighbor with fewest unvisited options
                    unvisited_neighbors.sort(key=lambda x: len([n for n in adj[x] if n not in seen]))
                    
                    # With some probability, choose randomly to escape local optima
                    if rng.random() < 0.2:
                        next_node = rng.choice(unvisited_neighbors)
                    else:
                        next_node = unvisited_neighbors[0]
                    
                    path.append(next_node)
                    seen.add(next_node)
                
                if len(path) > len(best_local):
                    best_local = path[:]
            
            if len(best_local) > len(best_path):
                best_path = best_local

    return best_path
