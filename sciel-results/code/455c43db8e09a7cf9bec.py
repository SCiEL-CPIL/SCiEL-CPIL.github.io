import time
from itertools import combinations, product

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
    
    # Bipartition: even in A, odd in B
    A = list(range(0, n, 2))  # 12 even vertices
    B = list(range(1, n, 2))  # 12 odd vertices
    
    # All candidate edges: between A and B, not in base
    candidates = []
    for u in A:
        for v in B:
            if frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for e in base_edges:
        u, v = tuple(e)
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    deadline = time.time() + 85
    
    # Fast diameter check using BFS
    def check_diameter(adj):
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if any(d == -1 or d > 4 for d in dist):
                return False
        return True
    
    # Fast triangle-free check: for each added edge (u,v), check no common neighbor
    def check_triangle_free(added_edges, adj):
        for u, v in added_edges:
            if adj[u] & adj[v]:
                return False
        return True
    
    # Build adjacency from base + added edges
    def build_adj(added_edges):
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Full validation
    def validate(added_edges):
        adj = build_adj(added_edges)
        if not check_triangle_free(added_edges, adj):
            return False
        if not check_diameter(adj):
            return False
        return True
    
    # Try k = 5, 6, 7 with structured search
    # Key idea: use greedy + local search for each k
    
    import random
    
    for k in range(5, 8):
        if time.time() > deadline:
            break
        
        found = None
        
        if k <= 6:
            # Use a smarter approach: 
            # 1. Greedy construction
            # 2. Then local search to verify/improve
            
            # Greedy: start with no edges, add edges that reduce "diameter violation" most
            # For k small, try multiple random restarts of greedy
            
            for restart in range(5000 if k <= 5 else 2000):
                if time.time() > deadline:
                    break
                
                random.seed(restart * 1000 + k)
                
                # Random subset of candidates of size k
                if len(candidates) >= k:
                    combo = random.sample(candidates, k)
                    if validate(combo):
                        found = combo
                        break
                
                # Also try greedy approach
                if restart % 10 == 0:
                    # Greedy: pick edges that help most with diameter
                    # Start with base, compute which pairs are at distance > 4
                    # (none in base path for n=24? Actually max distance is 23)
                    # Greedy: pick edge that minimizes max distance
                    current_edges = []
                    remaining = list(candidates)
                    random.shuffle(remaining)
                    
                    for _ in range(k):
                        if not remaining:
                            break
                        best_edge = None
                        best_score = float('inf')
                        # Sample some candidates for speed
                        sample = remaining[:min(50, len(remaining))]
                        for edge in sample:
                            trial = current_edges + [edge]
                            adj = build_adj(trial)
                            # Quick check: is it triangle-free so far?
                            if not check_triangle_free(trial, adj):
                                continue
                            # Compute max distance (approximate)
                            max_d = 0
                            for start in [0, 11, 12, 23]:  # Check key vertices
                                dist = [-1] * n
                                dist[start] = 0
                                queue = [start]
                                qi = 0
                                while qi < len(queue):
                                    u = queue[qi]
                                    qi += 1
                                    if dist[u] >= 4:
                                        continue
                                    for v in adj[u]:
                                        if dist[v] == -1:
                                            dist[v] = dist[u] + 1
                                            queue.append(v)
                                for d in dist:
                                    if d > max_d:
                                        max_d = d
                            if max_d < best_score:
                                best_score = max_d
                                best_edge = edge
                        if best_edge is not None:
                            current_edges.append(best_edge)
                            remaining.remove(best_edge)
                    
                    if len(current_edges) == k and validate(current_edges):
                        found = current_edges
                        break
        
        if found:
            return [[u, v] for u, v in found]
    
    # Fallback: known 7-edge solution
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    return [[hub, t] for t in targets]
