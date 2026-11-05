import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    n = 18
    
    # Helper: Check if a set of vertices (mask) is an independent set in a graph defined by adj
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    # Helper: Calculate largest triangle-free induced subgraph size using branch and bound
    def largest_triangle_free(adj):
        # Precompute adjacency masks
        # A triangle-free induced subgraph is a set S where no three vertices form a triangle
        # Equivalently, for every vertex v in S, the neighbors of v in S form an independent set
        
        best = 0
        
        def bnb(remaining, current_size, current_set):
            nonlocal best
            if time.time() > deadline - 3.0:
                return
            if current_size + bin(remaining).count('1') <= best:
                return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Upper bound: current_size + popcount(remaining)
            upper = current_size + bin(remaining).count('1')
            if upper <= best:
                return
            
            # Choose a vertex from remaining
            # Heuristic: pick vertex with most neighbors in remaining
            m = remaining
            best_v = -1
            best_deg = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & remaining & ~lsb).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v < 0:
                return
            
            # Option 1: include best_v
            # Check if adding best_v to current_set keeps it triangle-free
            # best_v forms triangle if it has two neighbors in current_set that are connected
            nbrs_in_set = adj[best_v] & current_set
            if is_independent(nbrs_in_set, adj):
                new_remaining = remaining & ~(1 << best_v)
                # Also remove vertices that would form triangles with best_v
                # Actually we just need to ensure future additions don't create triangles
                bnb(new_remaining, current_size + 1, current_set | (1 << best_v))
            
            # Option 2: exclude best_v
            bnb(remaining & ~(1 << best_v), current_size, current_set)
        
        bnb((1 << n) - 1, 0, 0)
        return best

    # Faster score calculation using greedy with multiple orders
    def calc_score_fast(adj):
        best = 0
        for _ in range(20):
            order = list(range(n))
            random.shuffle(order)
            S = 0
            for v in order:
                nbrs_in_S = adj[v] & S
                if is_independent(nbrs_in_S, adj):
                    S |= (1 << v)
            sz = bin(S).count('1')
            if sz > best:
                best = sz
        return best

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Build 3-partite Turán graph baseline
    # Split 18 vertices into 3 parts of size 6
    # Connect all vertices between different parts, no edges within parts
    # This is K4-free (any 4 vertices must have at least 2 in the same part, which aren't connected)
    # Triangle-free induced subgraph: can take at most 2 from each part (since 3 from same part have no edges among them,
    # but wait - within a part there are NO edges, so all 6 in a part form an independent set, hence triangle-free)
    # Actually the max triangle-free induced subgraph in T(18,3) is 6 (one whole part)
    # But we can also take vertices from multiple parts as long as no triangle is formed
    # In a 3-partite complete graph, a triangle requires one vertex from each part
    # So a triangle-free set cannot have vertices from all 3 parts simultaneously with edges between them
    # If we take vertices from parts A and B only, there are no triangles (need 3 parts for a triangle in complete 3-partite)
    # So we can take all of A + all of B = 12 vertices, which is triangle-free!
    # Wait, that's not right. In complete 3-partite, any three vertices from different parts form a triangle.
    # If we take vertices from only 2 parts, no triangles form. So max triangle-free = 6+6 = 12.
    # That's too large. We need to break the completeness.
    
    # Better approach: use a random maximal K4-free graph but with structured initialization
    # Start with a random partition and add edges probabilistically
    
    random.seed(42)
    
    # Strategy 1: Random maximal K4-free graph
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    current_score = calc_score_fast(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Strategy 2: Try different random seeds for initialization
    for seed in range(20):
        if time.time() > deadline - 10.0:
            break
        random.seed(seed + 100)
        adj2 = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj2, u, v):
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
        score2 = calc_score_fast(adj2)
        if score2 < best_score:
            best_score = score2
            best_adj = adj2[:]
            adj = adj2[:]
            current_score = score2
    
    # Local search with edge flips
    T = 1.0
    T_min = 0.01
    alpha = 0.985
    
    while time.time() < deadline - 8.0 and T > T_min:
        T *= alpha
        
        # Random move: Add or Remove an edge
        if random.random() < 0.5:
            # Try adding an edge
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
            if not non_edges:
                continue
            u, v = random.choice(non_edges)
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_score = calc_score_fast(adj)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj[:]
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            else:
                continue
        else:
            # Try removing an edge
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if ((adj[i] >> j) & 1)]
            if not edges:
                continue
            u, v = random.choice(edges)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            new_score = calc_score_fast(adj)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Periodically try a 2-edge swap
        if random.random() < 0.1:
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if ((adj[i] >> j) & 1)]
            if non_edges and edges:
                u1, v1 = random.choice(non_edges)
                u2, v2 = random.choice(edges)
                # Remove (u2,v2) and add (u1,v1)
                adj[u2] ^= (1 << v2)
                adj[v2] ^= (1 << u2)
                if can_add_edge(adj, u1, v1):
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    new_score = calc_score_fast(adj)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = adj[:]
                    else:
                        # Revert
                        adj[u1] ^= (1 << v1)
                        adj[v1] ^= (1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                else:
                    # Revert removal
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
    
    # Final verification with exact B&B if time permits
    if time.time() < deadline - 5.0:
        exact_score = largest_triangle_free(best_adj)
        if exact_score < best_score:
            best_score = exact_score
    
    return get_edge_list(best_adj)
