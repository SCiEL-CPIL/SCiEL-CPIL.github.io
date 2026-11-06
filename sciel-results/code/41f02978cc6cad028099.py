import random

def construct():
    n = 40
    target_edges = 401
    
    def compute_max_book(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        max_b = 0
        for u, v in edge_set:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = None
    best_score = float('inf')
    
    # Try many random graphs
    for seed in range(2000):
        rng = random.Random(seed)
        idx = list(range(len(all_possible)))
        rng.shuffle(idx)
        edges = set(all_possible[i] for i in idx[:target_edges])
        
        score = compute_max_book(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            if best_score <= 7:
                break
    
    # Local search to improve
    rng = random.Random(999)
    for _ in range(10000):
        edge_list = list(best_edges)
        u, v = edge_list[rng.randrange(len(edge_list))]
        
        for _ in range(20):
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            a, b = min(a, b), max(a, b)
            if (a, b) in best_edges:
                continue
            
            new_edges = best_edges.copy()
            new_edges.discard((u, v))
            new_edges.add((a, b))
            
            new_score = compute_max_book(new_edges)
            if new_score < best_score:
                best_score = new_score
                best_edges = new_edges
                break
    
    return [[u, v] for u, v in best_edges]
