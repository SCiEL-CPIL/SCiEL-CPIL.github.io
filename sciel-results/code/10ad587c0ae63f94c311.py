import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Andrasfai graph And(9)
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def has_independent_set_of_size_k(vertices, adj, k):
        if len(vertices) < k:
            return False
        
        def dfs(current_size, remaining):
            if current_size >= k:
                return True
            if not remaining:
                return False
            if current_size + len(remaining) < k:
                return False
            
            v = remaining[0]
            
            # Include v
            new_remaining = [u for u in remaining[1:] if not (adj[v] >> u) & 1]
            if dfs(current_size + 1, new_remaining):
                return True
            
            # Exclude v
            if dfs(current_size, remaining[1:]):
                return True
            
            return False
        
        # Sort by degree
        degs = []
        for v in vertices:
            deg = 0
            for u in vertices:
                if u != v and (adj[v] >> u) & 1:
                    deg += 1
            degs.append((deg, v))
        degs.sort(reverse=True)
        vertices_sorted = [v for d, v in degs]
        
        return dfs(0, vertices_sorted)

    def find_valid_S(adj, n, deadline):
        all_vertices = list(range(n))
        for attempt in range(5000):
            if time.time() > deadline:
                return None
            
            # Random independent set
            if not all_vertices:
                return None
            start_v = random.choice(all_vertices)
            S = [start_v]
            available = [v for v in all_vertices if v != start_v and not (adj[start_v] >> v) & 1]
            random.shuffle(available)
            for v in available:
                conflict = False
                for s in S:
                    if (adj[v] >> s) & 1:
                        conflict = True
                        break
                if not conflict:
                    S.append(v)
            
            S_set = set(S)
            remaining = [v for v in all_vertices if v not in S_set]
            
            if len(remaining) < 9:
                return S
            
            if not has_independent_set_of_size_k(remaining, adj, 9):
                return S
        return None

    while time.time() < deadline:
        S = find_valid_S(adj, n, deadline)
        if S is None:
            break
        
        new_v = n
        adj.append(0)
        for u in S:
            edges.append([new_v, u])
            adj[new_v] |= (1 << u)
            adj[u] |= (1 << new_v)
        
        n += 1

    return {'n': n, 'edges': edges}
