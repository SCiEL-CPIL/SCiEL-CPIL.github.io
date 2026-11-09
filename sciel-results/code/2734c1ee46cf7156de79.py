import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 40
    best_edges = []
    best_count = 0

    def get_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def creates_c6(u, v, adj):
        # Check if adding edge (u, v) creates a C6
        # This is true if there is a path of length 4 between u and v in current graph
        nu = adj[u]
        nv = adj[v]
        
        a_bit = nu
        while a_bit:
            a_idx = (a_bit & -a_bit).bit_length() - 1
            a_bit &= a_bit - 1
            
            c_bit = nv
            while c_bit:
                c_idx = (c_bit & -c_bit).bit_length() - 1
                c_bit &= c_bit - 1
                
                common = adj[a_idx] & adj[c_idx]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return True
        return False

    def find_c6_edge(adj):
        # Find an edge that is part of a C6
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Temporarily remove edge to check for path of length 4
                    # Or just check if there is a path of length 4 in G - (u, v)
                    # This is equivalent to: is there a path u-a-b-c-v where a!=v, c!=u, b!=u,v
                    # and a in N(u), c in N(v), b in N(a) cap N(c)
                    
                    # We can reuse creates_c6 logic?
                    # creates_c6(u, v, adj) checks if path exists in G.
                    # If we add (u,v) to G, we create a C6 if path exists in G.
                    # Here (u,v) is already in G.
                    # So we need to check if path exists in G - (u, v).
                    
                    # Let's just check if creates_c6(u, v, adj_without_uv) is true.
                    # But modifying adj is expensive.
                    # Instead, we can just check if there is a path of length 4 avoiding u and v as internal nodes?
                    # Path u-a-b-c-v.
                    # a in N(u) \ {v}
                    # c in N(v) \ {u}
                    # b in N(a) \ {u, v} intersect N(c) \ {u, v}
                    
                    nu = adj[u] & ~(1 << v)
                    nv = adj[v] & ~(1 << u)
                    
                    a_bit = nu
                    while a_bit:
                        a_idx = (a_bit & -a_bit).bit_length() - 1
                        a_bit &= a_bit - 1
                        
                        c_bit = nv
                        while c_bit:
                            c_idx = (c_bit & -c_bit).bit_length() - 1
                            c_bit &= c_bit - 1
                            
                            common = adj[a_idx] & adj[c_idx]
                            common &= ~(1 << u)
                            common &= ~(1 << v)
                            if common:
                                return [u, v]
        return None

    def repair(adj):
        while time.time() < deadline:
            edge_to_remove = find_c6_edge(adj)
            if edge_to_remove is None:
                break
            u, v = edge_to_remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    def optimize(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if not creates_c6(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True

    while time.time() < deadline:
        # Random initialization
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        repair(adj)
        optimize(adj)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_edges = edges_from_adj(adj)
            
        # Local Search
        for _ in range(20):
            if time.time() > deadline:
                break
            edges_list = edges_from_adj(adj)
            if not edges_list:
                break
            u, v = random.choice(edges_list)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            optimize(adj)
            
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_edges = edges_from_adj(adj)

    return best_edges
