import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n = 16
    target_edges = 65
    
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            triangles.append((i, j, k))
        return triangles
    
    def compute_score(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return 0
        return max(len(adj[i]) + len(adj[j]) + len(adj[k]) for i, j, k in triangles)
    
    # Try multiple initial configurations
    best_score = float('inf')
    best_edges = None
    
    configs = []
    
    # Config 1: K_{8,8} + (0,1)
    configs.append([(u, v) for u in range(8) for v in range(8, 16)] + [(0, 1)])
    
    # Config 2: K_{8,8} + (0,2) - different within edge
    configs.append([(u, v) for u in range(8) for v in range(8, 16)] + [(0, 2)])
    
    # Config 3: Remove one cross edge, add two within edges
    # K_{8,8} minus (0,8) plus (0,1) and (2,3)
    configs.append([(u, v) for u in range(8) for v in range(8, 16) if not (u==0 and v==8)] + [(0,1), (2,3)])
    
    # Config 4: K_{7,9} + some edges
    # Part A: 7 vertices, Part B: 9 vertices
    # K_{7,9} has 63 edges, need 2 more
    configs.append([(u, v) for u in range(7) for v in range(7, 16)] + [(0,1), (14,15)])
    
    for edges in configs:
        if len(edges) != target_edges:
            continue
        adj = make_adj(edges)
        score = compute_score(adj)
        if score < best_score:
            best_score = score
            best_edges = list(edges)
    
    # Local search from best
    current_edges = list(best_edges) if best_edges else configs[0]
    current_adj = make_adj(current_edges)
    current_score = compute_score(current_adj)
    
    random.seed(42)
    
    while time.time() < DEADLINE:
        improved = False
        current_edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
        
        for _ in range(3000):
            if time.time() > DEADLINE:
                break
            
            # Pick edge to remove
            u, v = random.choice(current_edges)
            uv_key = (min(u,v), max(u,v))
            new_edges = [e for e in current_edges if (min(e[0],e[1]), max(e[0],e[1])) != uv_key]
            
            # Pick non-edge to add
            while True:
                a, b = random.randint(0, n-1), random.randint(0, n-1)
                if a < b and (a, b) not in current_edge_set:
                    break
            
            new_edges.append((a, b))
            new_adj = make_adj(new_edges)
            new_score = compute_score(new_adj)
            
            if new_score < current_score:
                current_edges = new_edges
                current_adj = new_adj
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
        
        if not improved and random.random() < 0.1:
            # Random restart
            current_edges = list(best_edges)
            current_adj = make_adj(current_edges)
            current_score = best_score
    
    result_edges = [[u, v] for u, v in best_edges]
    return {'n': n, 'edges': result_edges}
