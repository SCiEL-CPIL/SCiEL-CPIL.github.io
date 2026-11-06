import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    n = 16
    m_target = 65
    
    # Strategy:
    # 1. Start with a highly structured initial guess: K_{a, b} + matching in larger part.
    #    K_{8,8} has 64 edges. We need 1 more. Adding an edge in the larger part creates triangles.
    #    However, K_{7,9} has 63 edges. We need 2 more.
    #    K_{6,10} has 60 edges. We need 5 more.
    #    The inspiration suggests K_{6,10} + 5 disjoint edges in the larger part (size 10) gives score 24.
    #    Let's verify:
    #    Vertices A (size 6), B (size 10).
    #    Edges: All A-B (60). Plus 5 disjoint edges in B.
    #    Degrees in A: 10 each.
    #    Degrees in B: 6 (from A) + 1 (from matching) = 7 each.
    #    Triangles: Formed by one vertex in A and an edge in B.
    #    Triangle (a, b1, b2) where b1-b2 is an edge in B.
    #    deg(a) = 10. deg(b1) = 7. deg(b2) = 7.
    #    Sum = 24.
    #    Are there other triangles? No, because there are no edges in A and no other edges in B.
    #    So the max triangle sum is exactly 24.
    #    Can we do better (lower)?
    #    The conjectured asymptotic is ~1.464n = 23.4. So 23 or 24 is the target.
    #    Let's try to optimize from this 24 structure using local search.
    
    # Initialize with K_{6,10} + 5 matching edges in B
    # A: 0-5, B: 6-15
    adj = np.zeros((n, n), dtype=int)
    
    # Bipartite part
    for u in range(6):
        for v in range(6, 16):
            adj[u, v] = 1
            adj[v, u] = 1
            
    # Matching in B (6-15)
    # Pairs: (6,7), (8,9), (10,11), (12,13), (14,15)
    matching_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in matching_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    deg = np.array([np.sum(adj[i]) for i in range(n)])
    
    # Verify edge count
    edge_count = np.sum(adj) // 2
    if edge_count != m_target:
        # Fallback to K8,8 + 1 if structure fails (it shouldn't)
        adj = np.zeros((n, n), dtype=int)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        adj[0, 1] = 1
        adj[1, 0] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)])

    def calculate_score():
        """
        Vectorized calculation of max triangle degree sum.
        A triangle exists for (u, v, w) if adj[u,v], adj[u,w], adj[v,w] are 1.
        We can iterate over edges (u,v) and find common neighbors w.
        """
        max_sum = 0
        # Iterate over edges
        # To speed up, we can precompute common neighbors?
        # Or just iterate u < v < w.
        # With n=16, n^3/6 = 560 triangles max. 16*15*14/6 = 560.
        # This is very fast.
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Find w > v such that adj[u,w] and adj[v,w]
                    # Vectorized check for w in v+1..n-1
                    common = adj[u, v+1:] & adj[v, v+1:]
                    # Indices of common neighbors
                    ws = np.where(common)[0] + v + 1
                    if len(ws) > 0:
                        sums = deg[u] + deg[v] + deg[ws]
                        current_max = np.max(sums)
                        if current_max > max_sum:
                            max_sum = current_max
        return max_sum

    current_score = calculate_score()
    best_score = current_score
    best_adj = adj.copy()
    
    # Local Search
    # Move: Swap an edge. Remove (u,v), Add (x,y).
    # Constraint: Keep edge count constant.
    # To maintain structure, we might prefer moves that keep the "bipartite-like" property,
    # but generic swaps allow escaping local minima.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Generate a random valid swap
        # Pick random edge to remove
        # Pick random non-edge to add
        
        # Optimization: Precompute list of current edges and non-edges?
        # Updating lists is expensive.
        # Instead, sample until valid.
        
        removed = None
        added = None
        
        # Try to find a removal
        attempts = 0
        while removed is None and attempts < 20:
            u, v = random.choice(all_pairs)
            if adj[u, v]:
                removed = (u, v)
            attempts += 1
            
        attempts = 0
        while added is None and attempts < 20:
            x, y = random.choice(all_pairs)
            if not adj[x, y]:
                added = (x, y)
            attempts += 1
            
        if removed is None or added is None:
            continue
            
        # Ensure we don't swap an edge with itself (impossible if one is edge, one is non-edge)
        # But ensure (u,v) != (x,y) which is guaranteed.
        
        u, v = removed
        x, y = added
        
        # Apply
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        deg[u] -= 1
        deg[v] -= 1
        deg[x] += 1
        deg[y] += 1
        
        new_score = calculate_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        elif new_score == current_score:
            # Accept equal to explore
            pass
        else:
            # Metropolis
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                pass
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[x, y] = 0
                adj[y, x] = 0
                
                deg[u] += 1
                deg[v] += 1
                deg[x] -= 1
                deg[y] -= 1
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
