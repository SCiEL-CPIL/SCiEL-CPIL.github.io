import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    
    # Build initial adjacency
    base_adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        existing_edges.add((min(u,v), max(u,v)))
    
    deadline = time.time() + 55.0
    
    def compute_diameter(adj):
        """Compute diameter of graph given adjacency list."""
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
    
    def check_solution(added_edges):
        """Check if adding these edges gives valid solution."""
        adj = [set(s) for s in base_adj]
        
        # Check no duplicate edges
        for u, v in added_edges:
            a, b = min(u,v), max(u,v)
            if (a,b) in existing_edges:
                return False
            if u == v:
                return False
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        if not is_triangle_free(adj):
            return False
        
        # Check connected and diameter
        diam = compute_diameter(adj)
        if diam == -1 or diam > 4:
            return False
        
        return True
    
    # Generate candidate edges (not in base, not self-loops)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) not in existing_edges:
                candidates.append((u,v))
    
    # Try increasing number of edges
    for k in range(1, 10):
        if time.time() > deadline:
            break
        
        # Use greedy first to get a good upper bound, then search
        # For small k, try all combinations with pruning
        if k <= 3:
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if check_solution(list(combo)):
                    return [[int(u), int(v)] for u, v in combo]
        else:
            # For larger k, use greedy with backtracking
            # First try greedy to find a solution
            greedy_result = greedy_search()
            if greedy_result is not None and len(greedy_result) <= k:
                return [[int(u), int(v)] for u, v in greedy_result]
            
            # Try structured patterns
            pattern_result = try_patterns(k)
            if pattern_result is not None:
                return [[int(u), int(v)] for u, v in pattern_result]
    
    # Fallback: greedy
    result = greedy_search()
    if result is not None:
        return [[int(u), int(v)] for u, v in result]
    
    # Last resort: return empty (invalid but shouldn't happen)
    return []

def greedy_search():
    """Greedy search for minimum edges."""
    n = P['n']
    base_adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        existing_edges.add((min(u,v), max(u,v)))
    
    adj = [set(s) for s in base_adj]
    added = set()
    added_edges = []
    
    def compute_diameter():
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    def is_valid(u, v):
        if u == v:
            return False
        a, b = min(u,v), max(u,v)
        if (a,b) in existing_edges or (a,b) in added:
            return False
        if adj[u] & adj[v]:
            return False
        return True
    
    deadline = time.time() + 50.0
    while time.time() < deadline:
        diam = compute_diameter()
        if diam <= 4:
            break
        
        best_edge = None
        best_new_diam = diam + 1
        
        for u in range(n):
            for v in range(u+1, n):
                if not is_valid(u, v):
                    continue
                # Try adding edge
                adj[u].add(v)
                adj[v].add(u)
                new_diam = compute_diameter()
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.add((min(u,v), max(u,v)))
        added_edges.append([u, v])
    
    return added_edges

def try_patterns(k):
    """Try structured patterns."""
    n = P['n']
    base_adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        existing_edges.add((min(u,v), max(u,v)))
    
    # Pattern 1: Add edges that create a "skip" pattern
    # Try: (0,5), (5,10), (10,15), (15,20), (20,23) - 5 edges
    patterns = [
        [(0,5), (5,10), (10,15), (15,20), (20,23)],
        [(0,4), (4,9), (9,14), (14,19), (19,23)],
        [(0,6), (6,12), (12,18), (18,23)],
        [(0,5), (5,10), (10,15), (15,20), (20,23), (0,10)],
        [(0,4), (4,8), (8,12), (12,16), (16,20), (20,23)],
    ]
    
    for pattern in patterns:
        adj = [set(s) for s in base_adj]
        valid = True
        for u, v in pattern:
            a, b = min(u,v), max(u,v)
            if (a,b) in existing_edges:
                valid = False
                break
            if adj[u] & adj[v]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if not valid:
            continue
        
        # Check triangle-free
        tf = True
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        tf = False
                        break
                if not tf:
                    break
            if not tf:
                break
        if not tf:
            continue
        
        # Check diameter
        diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > diam:
                    diam = d
        
        if diam <= 4:
            return pattern
    
    return None
