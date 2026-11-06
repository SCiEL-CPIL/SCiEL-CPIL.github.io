import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 18
    
    # 1. Construct Paley(17) + Isolated
    # Quadratic residues mod 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    adj = [0] * n
    for u in range(17):
        for v in range(u + 1, 17):
            if (v - u) % 17 in residues:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    # Vertex 17 is isolated, so no changes needed for adj[17]
    
    # 2. Helper functions
    def is_k4_free(adj):
        # Check if any edge (u,v) has a triangle in common neighborhood
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Check if common has any edge
                    c = common
                    while c:
                        w = (c & -c).bit_length() - 1
                        c &= c - 1
                        if adj[w] & common:
                            return False
        return True

    def max_triangle_free_size(adj, timeout=0.1):
        start = time.time()
        best = 0
        all_vertices = (1 << n) - 1
        
        def bt(remaining, current):
            nonlocal best
            if time.time() - start > timeout:
                return
            
            # Pruning
            if current.bit_count() + remaining.bit_count() <= best:
                return
            
            if not remaining:
                if current.bit_count() > best:
                    best = current.bit_count()
                return
            
            # Pick vertex
            # Heuristic: Pick vertex with highest degree in remaining?
            # Or just lowest index.
            v = (remaining & -remaining).bit_length() - 1
            rest = remaining & ~(1 << v)
            
            # Option 1: Exclude v
            bt(rest, current)
            
            # Option 2: Include v
            # Check compatibility
            Nv_in_C = adj[v] & current
            can_add = True
            c = Nv_in_C
            while c:
                u = (c & -c).bit_length() - 1
                c &= c - 1
                if adj[u] & Nv_in_C:
                    can_add = False
                    break
            
            if can_add:
                bt(rest, current | (1 << v))
                
        bt(all_vertices, 0)
        return best

    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    # Initial evaluation
    current_adj = adj[:]
    current_score = max_triangle_free_size(current_adj, timeout=1.0)
    best_adj = current_adj[:]
    best_score = current_score
    
    # 3. Local Search
    all_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
    T = 5.0
    T_min = 0.1
    cooling = 0.99
    random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        u, v = random.choice(list(all_edges))
        
        new_adj = current_adj[:]
        
        if (new_adj[u] >> v) & 1:
            # Remove edge
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            is_valid = True
        else:
            # Add edge
            common = new_adj[u] & new_adj[v]
            valid = True
            c = common
            while c:
                w = (c & -c).bit_length() - 1
                c &= c - 1
                if new_adj[w] & common:
                    valid = False
                    break
            if not valid:
                T *= cooling
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            is_valid = True
            
        if is_valid:
            new_score = max_triangle_free_size(new_adj, timeout=0.05)
            
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
            else:
                delta = new_score - current_score
                if delta == 0:
                    if random.random() < 0.5:
                        current_adj = new_adj
                        current_score = new_score
                else:
                    prob = pow(2.71828, -delta / T)
                    if random.random() < prob:
                        current_adj = new_adj
                        current_score = new_score
                        
        T *= cooling
        
    return get_edges(best_adj)
