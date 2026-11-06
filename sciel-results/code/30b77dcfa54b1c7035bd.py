"""Longest simple path in pinned G(100, 1.8/100) graph."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees
    deg = [len(adj[i]) for i in range(n)]
    
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
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Focus on the largest component
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    best_path = []
    best_len = 0
    
    def try_extend(path, seen):
        """Try to extend path from both ends."""
        nonlocal best_path, best_len
        changed = True
        while changed:
            changed = False
            # Extend from front
            if path:
                front = path[0]
                for nb in sorted(adj[front], key=lambda x: (deg[x], x)):
                    if nb not in seen:
                        path.insert(0, nb)
                        seen.add(nb)
                        changed = True
                        break
            # Extend from back
            if path:
                back = path[-1]
                for nb in sorted(adj[back], key=lambda x: (deg[x], x)):
                    if nb not in seen:
                        path.append(nb)
                        seen.add(nb)
                        changed = True
                        break
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    def dfs_warnsdorff(start, max_depth=None):
        """DFS with Warnsdorff's heuristic: prefer low-degree unvisited neighbors."""
        nonlocal best_path, best_len
        path = [start]
        seen = {start}
        
        def dfs():
            nonlocal best_path, best_len
            if time.time() - start_time > DEADLINE:
                return
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            current = path[-1]
            # Get unvisited neighbors, sorted by degree (Warnsdorff)
            candidates = [nb for nb in adj[current] if nb not in seen]
            if not candidates:
                return
            
            # Sort by degree (prefer low degree to avoid trapping)
            candidates.sort(key=lambda x: (deg[x], x))
            
            for nb in candidates:
                path.append(nb)
                seen.add(nb)
                dfs()
                path.pop()
                seen.discard(nb)
                if time.time() - start_time > DEADLINE:
                    return
        
        dfs()
    
    def iterative_deepening(start):
        """Try iterative deepening DFS with pruning."""
        nonlocal best_path, best_len
        # First do a greedy pass to get an initial path
        path = [start]
        seen = {start}
        while True:
            current = path[-1]
            candidates = [nb for nb in adj[current] if nb not in seen]
            if not candidates:
                break
            # Pick the neighbor with fewest unvisited neighbors (Warnsdorff)
            best_nb = min(candidates, key=lambda x: (len(adj[x] - seen), x))
            path.append(best_nb)
            seen.add(best_nb)
        
        try_extend(path, seen)
        
        # Now try DFS from the end of this path to see if we can improve
        if time.time() - start_time > DEADLINE:
            return
        
        # Try backtracking from various points
        # Simple approach: try each vertex in the path as a potential branch point
        for i in range(len(path) - 1, max(len(path) - 10, -1), -1):
            if time.time() - start_time > DEADLINE:
                return
            prefix = path[:i+1]
            prefix_seen = set(prefix)
            # Try DFS from prefix[-1] with the remaining vertices
            def dfs_from_prefix():
                nonlocal best_path, best_len
                if time.time() - start_time > DEADLINE:
                    return
                if len(prefix) > best_len:
                    best_len = len(prefix)
                    best_path = list(prefix)
                
                current = prefix[-1]
                candidates = [nb for nb in adj[current] if nb not in prefix_seen]
                if not candidates:
                    return
                candidates.sort(key=lambda x: (deg[x], x))
                
                for nb in candidates:
                    prefix.append(nb)
                    prefix_seen.add(nb)
                    dfs_from_prefix()
                    prefix.pop()
                    prefix_seen.discard(nb)
                    if time.time() - start_time > DEADLINE:
                        return
            
            dfs_from_prefix()
    
    # Try all vertices as starting points, prioritized by degree
    # Start with low-degree vertices (more likely to be endpoints of long paths)
    starts = sorted(range(n), key=lambda x: (deg[x], x))
    
    # Try each start vertex
    for s in starts:
        if time.time() - start_time > DEADLINE:
            break
        if deg[s] == 0:
            continue
        
        # Quick greedy first
        path = [s]
        seen = {s}
        while True:
            current = path[-1]
            candidates = [nb for nb in adj[current] if nb not in seen]
            if not candidates:
                break
            best_nb = min(candidates, key=lambda x: (len(adj[x] - seen), x))
            path.append(best_nb)
            seen.add(best_nb)
        
        try_extend(path, seen)
        
        # If we haven't found a very long path yet, try DFS with backtracking
        if best_len < n and time.time() - start_time < DEADLINE - 20:
            # Only do full DFS for promising starts
            if deg[s] <= 3 or len(path) >= best_len - 2:
                dfs_warnsdorff(s)
    
    # Final extension pass on the best path
    if best_path:
        seen = set(best_path)
        try_extend(best_path, seen)
    
    # Ensure we return a valid non-empty path
    if not best_path:
        # Fallback: return any edge as a 2-vertex path
        for u, v in P['edges']:
            return [u, v]
        # If no edges at all (shouldn't happen), return a single vertex
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
