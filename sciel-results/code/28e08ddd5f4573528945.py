import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Identify largest connected component
    visited_comp = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            comp = []
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > len(largest_comp):
                largest_comp = comp
            
    if not largest_comp:
        return [0]
    
    best_path = []
    best_len = 0
    
    # Multiple greedy strategies to get a good initial lower bound
    def greedy_warnsdorff(start_node, strategy='warnsdorff'):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            if strategy == 'warnsdorff':
                next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            elif strategy == 'max_degree':
                next_node = max(neighbors, key=lambda x: (len(adj[x] - visited), -x))
            elif strategy == 'random':
                next_node = random.choice(neighbors)
            else:
                next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    rng = random.Random(12345)
    
    # Try multiple greedy strategies from all start nodes
    for i in largest_comp:
        for strategy in ['warnsdorff', 'max_degree']:
            path = greedy_warnsdorff(i, strategy)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
        
        # Random restarts
        for _ in range(3):
            path = greedy_warnsdorff(i, 'random')
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    if time.time() > deadline:
        return best_path
    
    # Forced-move greedy: always take the unique unvisited neighbor if available
    def forced_greedy(start_node):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            # Check for forced moves: neighbors with only 1 unvisited neighbor (excluding last)
            forced = []
            for x in neighbors:
                # Count unvisited neighbors of x, excluding last (which is visited)
                unvisited_neighs = adj[x] - visited
                if len(unvisited_neighs) == 1:
                    forced.append(x)
            if forced:
                # Take the forced move
                next_node = forced[0]
            else:
                # Warnsdorff: pick neighbor with fewest unvisited neighbors
                next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    # Try forced greedy from all start nodes
    for i in largest_comp:
        path = forced_greedy(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Extend both ends of the path
    def extend_both_ends(path):
        visited = set(path)
        path = list(path)
        changed = True
        while changed:
            changed = False
            # Try extending the left end
            if path:
                left = path[0]
                left_neighbors = [x for x in adj[left] if x not in visited]
                if left_neighbors:
                    # Pick the one with fewest unvisited neighbors
                    next_left = min(left_neighbors, key=lambda x: (len(adj[x] - visited), x))
                    path.insert(0, next_left)
                    visited.add(next_left)
                    changed = True
            # Try extending the right end
            if path:
                right = path[-1]
                right_neighbors = [x for x in adj[right] if x not in visited]
                if right_neighbors:
                    next_right = min(right_neighbors, key=lambda x: (len(adj[x] - visited), x))
                    path.append(next_right)
                    visited.add(next_right)
                    changed = True
        return path
    
    best_path = extend_both_ends(best_path)
    best_len = len(best_path)
    
    if time.time() > deadline:
        return best_path
    
    # Backtracking search to find longer paths
    def backtrack_search(target_len, start_node, time_limit):
        """
        Try to find a path of length target_len starting from start_node.
        Uses DFS with pruning.
        """
        deadline_local = start_time + time_limit
        
        def dfs(node, path, visited):
            if time.time() > deadline_local:
                return None
            if len(path) == target_len:
                return list(path)
            
            neighbors = [x for x in adj[node] if x not in visited]
            if not neighbors:
                return None
            
            # Sort by remaining degree (Warnsdorff)
            neighbors.sort(key=lambda x: len(adj[x] - visited))
            
            # Forced move check
            forced = []
            for x in neighbors:
                unvisited_neighs = adj[x] - visited
                if len(unvisited_neighs) == 1:
                    forced.append(x)
            
            # Try forced moves first, then others
            search_order = forced + [x for x in neighbors if x not in forced]
            
            for x in search_order:
                path.append(x)
                visited.add(x)
                res = dfs(x, path, visited)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)
                
            return None
        
        path = [start_node]
        visited = {start_node}
        return dfs(start_node, path, visited)
    
    # Try to find longer paths using backtracking
    max_possible = len(largest_comp)
    current_target = best_len + 1
    
    # Time budget for each target length
    while current_target <= max_possible and time.time() < deadline - 10.0:
        found = False
        search_time_left = deadline - 10.0 - time.time()
        
        # Try multiple start nodes
        start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))
        
        for start in start_nodes:
            if time.time() > deadline - 10.0:
                break
            
            # Time limit for this start node
            time_per_start = min(5.0, search_time_left / max(1, len(start_nodes)))
            res = backtrack_search(current_target, start, time_per_start)
            if res is not None:
                best_len = current_target
                best_path = res
                found = True
                break
        
        if found:
            current_target += 1
        else:
            # If we couldn't find a path of this length, try a slightly longer one
            # with a different heuristic, or break
            # Let's try one more time with a longer time budget
            search_time_left = deadline - 10.0 - time.time()
            if search_time_left > 5.0:
                for start in start_nodes[:10]:  # Only try first 10 start nodes
                    if time.time() > deadline - 10.0:
                        break
                    res = backtrack_search(current_target, start, search_time_left / 10)
                    if res is not None:
                        best_len = current_target
                        best_path = res
                        found = True
                        break
            
            if not found:
                break
    
    # Final polish: extend both ends again
    best_path = extend_both_ends(best_path)
    best_len = len(best_path)
    
    return best_path
# EVOLVE-BLOCK-END
