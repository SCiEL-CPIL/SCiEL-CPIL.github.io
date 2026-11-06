"""Rewritten program: Warnsdorff-guided DFS + backtracking + path extension."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best_path = []
    
    def try_extend(path, adj):
        """Try to extend path at both ends."""
        path = list(path)
        changed = True
        while changed:
            changed = False
            # Try to extend at the front
            front = path[0]
            for v in sorted(adj[front]):
                if v not in set(path):
                    path.insert(0, v)
                    changed = True
                    break
            # Try to extend at the back
            if not changed:
                back = path[-1]
                for v in sorted(adj[back]):
                    if v not in set(path):
                        path.append(v)
                        changed = True
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

    def improve_path(path, adj, deadline):
        """Improve a path by extension and insertion."""
        path = list(path)
        visited = set(path)
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to extend at ends
            new_path = try_extend(path, adj)
            if len(new_path) > len(path):
                path = new_path
                visited = set(path)
                improved = True
                continue
            
            # 2. Try to insert unvisited vertices into the path
            # Check all edges in the path
            found_insertion = False
            for i in range(len(path) - 1):
                u = path[i]
                w = path[i + 1]
                # Check if any unvisited vertex v is connected to both u and w
                common_neighbors = adj[u] & adj[w]
                candidates = [v for v in common_neighbors if v not in visited]
                if candidates:
                    # Pick one (random or first)
                    v = candidates[0]
                    path.insert(i + 1, v)
                    visited.add(v)
                    found_insertion = True
                    improved = True
                    break
            
            if found_insertion:
                continue
            
            # If no improvement, we are stuck.
            # We could try to restart, but this function only improves a single path.
            # The caller will handle restarts.
            break
            
        return path

    # Strategy:
    # 1. Generate an initial path using DFS from a few starts.
    # 2. Improve it using the insertion/extension loop.
    # 3. If stuck, try a new random start and repeat.
    
    starts = list(range(n))
    rng = random.Random(42)
    rng.shuffle(starts)
    
    # We will keep a set of "tried" starts to avoid redundant work if we restart
    # But since we just pick random starts, we might repeat. 
    # Better: iterate through starts, but allow "restart" logic to pick new ones.
    
    # Let's just try all starts once with the improvement loop.
    # If time permits, we can do multiple passes.
    
    for start in starts:
        if time.time() > deadline:
            break
        if len(adj[start]) == 0:
            continue
            
        # Get initial path
        path = warnsdorff_dfs(start, adj, deadline)
        
        # Improve it
        path = improve_path(path, adj, deadline)
        
        if len(path) > len(best_path):
            best_path = path
            
    # If we still have time, try to improve the best_path further with random restarts
    # or just run improve_path again? It will break immediately if no improvement.
    # So we need to perturb.
    
    # Let's try a few random restarts if time is left
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 50:
        restarts += 1
        start = rng.randint(0, n - 1)
        if len(adj[start]) == 0:
            continue
        
        # Randomized greedy start
        path = [start]
        visited = {start}
        # Greedy extension
        for _ in range(n):
            current = path[-1]
            candidates = [v for v in adj[current] if v not in visited]
            if not candidates:
                break
            # Random choice
            v = rng.choice(candidates)
            path.append(v)
            visited.add(v)
            
        # Improve
        path = improve_path(path, adj, deadline)
        
        if len(path) > len(best_path):
            best_path = path
            
    if not best_path:
        best_path = [0]
        
    return best_path
