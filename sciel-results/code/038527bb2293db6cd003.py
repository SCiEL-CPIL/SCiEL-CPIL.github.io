"""Rewritten program: Multi-strategy longest path search with local search improvements."""
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best_path = []
    
    def is_valid_path(path, adj):
        """Check if path is valid: distinct vertices, consecutive edges."""
        if not path:
            return False
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    def try_extend_ends(path, adj, visited):
        """Try to extend path at both ends greedily."""
        path = list(path)
        visited = set(visited)
        changed = True
        while changed:
            changed = False
            # Try to extend at the front
            front = path[0]
            for v in sorted(adj[front]):
                if v not in visited:
                    path.insert(0, v)
                    visited.add(v)
                    changed = True
                    break
            # Try to extend at the back
            if not changed:
                back = path[-1]
                for v in sorted(adj[back]):
                    if v not in visited:
                        path.append(v)
                        visited.add(v)
                        changed = True
                        break
        return path, visited
    
    def insert_vertex(path, adj, visited, i, v):
        """Insert vertex v at position i in path, returning new path and visited."""
        new_path = list(path)
        new_visited = set(visited)
        new_path.insert(i, v)
        new_visited.add(v)
        return new_path, new_visited
    
    def local_search(path, adj, deadline):
        """Improve path via insertion and extension."""
        path = list(path)
        visited = set(path)
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to extend at ends
            new_path, new_visited = try_extend_ends(path, adj, visited)
            if len(new_path) > len(path):
                path = new_path
                visited = new_visited
                improved = True
                continue
            
            # 2. Try to insert unvisited vertices into the path
            found_insertion = False
            for i in range(len(path) - 1):
                u = path[i]
                w = path[i + 1]
                # Check if any unvisited vertex v is connected to both u and w
                common_neighbors = adj[u] & adj[w]
                candidates = [v for v in common_neighbors if v not in visited]
                if candidates:
                    v = candidates[0]
                    path, visited = insert_vertex(path, adj, visited, i + 1, v)
                    found_insertion = True
                    improved = True
                    break
            
            if found_insertion:
                continue
            
            # If no improvement, break
            break
            
        return path
    
    def warnsdorff_dfs(start, adj, deadline, max_depth=200):
        """DFS with Warnsdorff heuristic and backtracking."""
        best = [start]
        path = [start]
        visited = {start}
        
        def dfs():
            nonlocal best, path, visited
            if time.time() > deadline:
                return
            if len(path) > len(best):
                best = list(path)
            
            current = path[-1]
            candidates = [v for v in adj[current] if v not in visited]
            if not candidates:
                return
            
            # Warnsdorff: prefer vertices with fewest unvisited neighbors
            def key(v):
                return (len(adj[v] - visited), v)
            candidates.sort(key=key)
            
            for v in candidates:
                if time.time() > deadline:
                    return
                path.append(v)
                visited.add(v)
                dfs()
                path.pop()
                visited.discard(v)
        
        dfs()
        return best
    
    def greedy_path(start, adj, rng, deadline):
        """Greedy path construction with random choices."""
        path = [start]
        visited = {start}
        for _ in range(n):
            if time.time() > deadline:
                break
            current = path[-1]
            candidates = [v for v in adj[current] if v not in visited]
            if not candidates:
                break
            v = rng.choice(candidates)
            path.append(v)
            visited.add(v)
        return path
    
    def two_opt_improve(path, adj, deadline):
        """Try 2-opt style moves to improve path."""
        path = list(path)
        visited = set(path)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            # Try reversing segments
            for i in range(len(path) - 1):
                for j in range(i + 1, len(path)):
                    # Check if reversing path[i:j+1] creates valid edges
                    if i > 0 and path[i-1] in adj[path[j]]:
                        if j < len(path) - 1 and path[j+1] in adj[path[i]]:
                            # Reverse the segment
                            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                            if is_valid_path(new_path, adj):
                                path = new_path
                                improved = True
                                break
                if improved:
                    break
        
        return path
    
    # Strategy:
    # 1. Generate initial paths using multiple methods.
    # 2. Improve them using local search.
    # 3. Keep the best path found.
    
    rng = random.Random(42)
    starts = list(range(n))
    rng.shuffle(starts)
    
    # Try all starts with Warnsdorff DFS
    for start in starts:
        if time.time() > deadline:
            break
        if len(adj[start]) == 0:
            continue
        
        # Get initial path using Warnsdorff DFS
        path = warnsdorff_dfs(start, adj, deadline)
        
        # Improve it
        path = local_search(path, adj, deadline)
        
        if len(path) > len(best_path):
            best_path = path
    
    # If we still have time, try random restarts with greedy + local search
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 100:
        restarts += 1
        start = rng.randint(0, n - 1)
        if len(adj[start]) == 0:
            continue
        
        # Randomized greedy start
        path = greedy_path(start, adj, rng, deadline)
        
        # Improve
        path = local_search(path, adj, deadline)
        
        if len(path) > len(best_path):
            best_path = path
    
    # Final polish: try to improve best_path further
    if best_path:
        best_path = local_search(best_path, adj, deadline)
    
    if not best_path:
        best_path = [0]
        
    return best_path
