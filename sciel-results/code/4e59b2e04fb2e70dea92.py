import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_largest_component(adj, n):
    visited = [False] * n
    largest = []
    for start in range(n):
        if not visited[start]:
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            if len(comp) > len(largest):
                largest = comp
    return largest

def _warnsdorff_dfs(adj, start, target_len, deadline, best_path_ref):
    """
    DFS with Warnsdorff's heuristic.
    Tries to find a path of length target_len starting from 'start'.
    Updates best_path_ref if a longer path is found.
    """
    n = len(adj)
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Count unvisited neighbors for quick check
    # We can maintain a count of unvisited neighbors for each node
    unvisited_deg = [0] * n
    for u in range(n):
        for v in adj[u]:
            unvisited_deg[u] += 1 # This is just total degree initially, we need dynamic counts
        
    # Actually, maintaining dynamic counts is tricky with backtracking.
    # Let's just compute on the fly for the heuristic, but prune based on connectivity.
    
    def dfs():
        nonlocal best_path_ref
        if time.time() > deadline:
            return False
        
        cur_len = len(path)
        if cur_len > len(best_path_ref):
            best_path_ref = list(path)
            
        if cur_len == target_len:
            return True
            
        cur = path[-1]
        
        # Get unvisited neighbors
        candidates = [v for v in adj[cur] if not visited[v]]
        
        if not candidates:
            return False
            
        # Heuristic: Sort by number of unvisited neighbors (Warnsdorff)
        # Fewer unvisited neighbors first (to avoid isolating vertices)
        def heuristic(v):
            return sum(1 for u in adj[v] if not visited[u])
            
        candidates.sort(key=heuristic)
        
        # Pruning: Check if any unvisited vertex is isolated from the rest
        # If an unvisited vertex has 0 unvisited neighbors and is not cur, it's a problem?
        # No, it just means it can't be visited. But we want a long path.
        # A stronger prune: If there are multiple unvisited vertices with only 1 unvisited neighbor,
        # they must be visited in a specific order.
        
        # Simple prune: If we have unvisited vertices, and one of them has 0 unvisited neighbors,
        # and it's not adjacent to cur, we can't visit it. But we don't *have* to visit it if we just want *a* long path.
        # However, to maximize, we should avoid creating isolated unvisited vertices if possible.
        
        for v in candidates:
            visited[v] = True
            path.append(v)
            
            # Optional: Check if we've cut off any unvisited vertices completely
            # This is expensive. Let's rely on the heuristic.
            
            if dfs():
                return True
            
            path.pop()
            visited[v] = False
            
    dfs()
    return len(best_path_ref)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = P['n']
    adj = _build_adj()
    
    largest_comp = _find_largest_component(adj, n)
    if not largest_comp:
        return [0]
        
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)
    
    # Restrict adjacency to the component for efficiency
    # But we can just use the global adj and ignore nodes outside comp
    
    best_path = []
    
    # Try starting from vertices with low degree first
    # Calculate degrees within the component
    degs = {}
    for v in largest_comp:
        degs[v] = len([u for u in adj[v] if u in comp_set])
        
    sorted_starts = sorted(largest_comp, key=lambda v: degs[v])
    
    # Try a few starting vertices
    # We will try to find a path of length comp_size (Hamiltonian path)
    # If that fails, we accept the best found.
    
    # Strategy: 
    # 1. Try Warnsdorff DFS from the lowest degree vertex.
    # 2. If time allows, try from other vertices.
    
    # Since Warnsdorff is deterministic given the start, we just need one good start.
    # But if it fails to find a full path, we might get stuck in a local optimum of the heuristic.
    # Let's run it with a time limit.
    
    # We'll run the DFS from the first few low-degree vertices.
    for start in sorted_starts[:5]:
        if time.time() > deadline:
            break
            
        # Use a local deadline for this attempt, or just let it run until global deadline
        # The DFS function checks global deadline.
        # We need to pass a reference to update the best path.
        # Since Python passes lists by reference, we can just pass best_path and it will update.
        # But we want to keep the best from previous attempts if this one fails.
        
        # Let's make a local best for this attempt
        local_best = list(best_path)
        
        # The dfs function modifies best_path_ref. 
        # To avoid overwriting a good path with a worse one if the search dies early,
        # we should only update if we find something longer.
        # The current _warnsdorff_dfs updates best_path_ref only if cur_len > len(best_path_ref).
        # So passing best_path is safe.
        
        _warnsdorff_dfs(adj, start, comp_size, deadline, best_path)
        
        # If we found a Hamiltonian path, we are done
        if len(best_path) == comp_size:
            break
            
    # If we didn't find a full path, we might have a good partial path.
    # We can try to extend it or use 2-opt.
    # For now, return the best found.
    
    if not best_path:
        # Fallback to greedy
        start = sorted_starts[0]
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            cands = [v for v in adj[cur] if v not in seen and v in comp_set]
            if not cands:
                break
            # Pick min degree
            best_v = min(cands, key=lambda v: len([u for u in adj[v] if u not in seen and u in comp_set]))
            path.append(best_v)
            seen.add(best_v)
        best_path = path
        
    return best_path
