import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set()
    for e in base_edges:
        base_set.add(e)
        base_set.add((e[1], e[0]))
    
    # Precompute all non-base edges
    all_possible = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_set:
                all_possible.append((u, v))
                
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
        
    def compute_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return float('inf')
                        q.append(v)
            if any(d == -1 for d in dist):
                return float('inf')
            max_d = max(max_d, max(dist))
        return max_d
        
    def fast_check(added_edges):
        adj = build_adj(added_edges)
        if not is_triangle_free(adj):
            return False
        diam = compute_diameter(adj)
        return diam <= 4

    deadline = time.time() + 85.0
    
    best_solution = None
    best_count = float('inf')
    
    def try_solution(added_edges):
        nonlocal best_solution, best_count
        if len(added_edges) >= best_count:
            return
        if fast_check(added_edges):
            best_solution = list(added_edges)
            best_count = len(added_edges)
            
    # Strategy 1: Construct a strong initial guess
    # Use edges that jump far.
    # Pattern: (i, i+k)
    # Try k=6, 7, 8 etc.
    
    initial_candidates = []
    
    # Candidate 1: Uniform jumps of 6
    added = []
    for i in range(n):
        j = i + 6
        if j < n and (i, j) not in base_set:
            added.append((i, j))
    initial_candidates.append(added)
    
    # Candidate 2: Uniform jumps of 7
    added = []
    for i in range(n):
        j = i + 7
        if j < n and (i, j) not in base_set:
            added.append((i, j))
    initial_candidates.append(added)
    
    # Candidate 3: Mix of 5 and 8
    added = []
    for i in range(n):
        if i % 2 == 0:
            j = i + 8
            if j < n and (i, j) not in base_set:
                added.append((i, j))
        else:
            j = i + 5
            if j < n and (i, j) not in base_set:
                added.append((i, j))
    initial_candidates.append(added)
    
    # Candidate 4: Greedy long edges
    added = []
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Sort possible edges by distance desc
    sorted_edges = sorted(all_possible, key=lambda e: abs(e[0]-e[1]), reverse=True)
    
    for u, v in sorted_edges:
        if len(added) >= 15: break
        # Check triangle
        ok = True
        for w in adj[u]:
            if w in adj[v]:
                ok = False
                break
        if ok:
            added.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
    initial_candidates.append(added)
    
    # Local Search on each candidate
    for candidate in initial_candidates:
        if time.time() > deadline:
            break
            
        current = list(candidate)
        
        # Prune: remove any edge that is not needed
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                candidate = current[:i] + current[i+1:]
                if fast_check(candidate):
                    current = candidate
                    improved = True
                    break
        
        try_solution(current)
        
        # Swap search
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to replace one edge with another to reduce count? No, swap keeps count same.
                # We want to reduce count.
                # Try removing an edge and adding two? No.
                # Just stick to removal and "2-opt" like moves if we were maximizing, but here we minimize.
                # Actually, if we are stuck at k edges, maybe k-1 is not reachable by single removal.
                # But usually, if a solution is valid, removing an edge might break it.
                # If we can't remove, we are likely at a local minimum.
                
                # Try a different initialization?
                pass

    # Strategy 2: Randomized Restart with Local Search
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Random subset of long edges
        k = random.randint(8, 15)
        sample = random.sample(all_possible, min(k, len(all_possible)))
        
        # Filter triangle-free
        valid_sample = []
        temp_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            temp_adj[u].add(v)
            temp_adj[v].add(u)
            
        for u, v in sample:
            ok = True
            for w in temp_adj[u]:
                if w in temp_adj[v]:
                    ok = False
                    break
            if ok:
                valid_sample.append((u, v))
                temp_adj[u].add(v)
                temp_adj[v].add(u)
                
        if not fast_check(valid_sample):
            continue
            
        current = list(valid_sample)
        try_solution(current)
        
        # Local Search: Removal
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                candidate = current[:i] + current[i+1:]
                if fast_check(candidate):
                    current = candidate
                    improved = True
                    break
        try_solution(current)
        
        # Local Search: Swap (replace 1 edge with another, maybe allows removing later?)
        # Actually, just try to find a better swap that leads to removal?
        # Simpler: Try adding an edge and then removing two?
        # This is getting complex. Stick to removal.
        
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
        
    # Fallback
    # This should not happen if candidates are good
    # Use the greedy one
    return [[int(u), int(v)] for u, v in initial_candidates[3]]

# EVOLVE-BLOCK-END
