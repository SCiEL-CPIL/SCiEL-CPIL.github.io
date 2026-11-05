import random
import time
import sys

def construct():
    n = 40
    deadline = time.time() + 140.0  # Safety margin
    
    def bits_to_list(bits):
        res = []
        while bits:
            lsb = bits & -bits
            res.append(lsb.bit_length() - 1)
            bits ^= lsb
        return res

    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def has_c6(u, v, adj):
        nu = adj[u] & ~(1 << v)
        if not nu: return False
        nv = adj[v] & ~(1 << u)
        if not nv: return False
        
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a
            
            forbidden_b = (1 << u) | (1 << a) | (1 << v)
            bits_b = adj[a] & ~forbidden_b
            
            while bits_b:
                lsb_b = bits_b & -bits_b
                b = lsb_b.bit_length() - 1
                bits_b ^= lsb_b
                
                forbidden_c = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                bits_c = adj[b] & ~forbidden_c
                
                while bits_c:
                    lsb_c = bits_c & -bits_c
                    c = lsb_c.bit_length() - 1
                    bits_c ^= lsb_c
                    
                    forbidden_d = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c)
                    if adj[c] & nv & ~forbidden_d:
                        return True
        return False

    def is_valid_add(u, v, adj):
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        return not has_c6(u, v, adj)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def local_search(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Add valid edges
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_valid_add(u, v, adj):
                            add_edge(adj, u, v)
                            improved = True
                            if time.time() > deadline:
                                return

            if not improved:
                # 2. Try 1-2 swaps
                edges = adj_to_edges(adj)
                random.shuffle(edges)
                
                for u, v in edges:
                    if time.time() > deadline:
                        return
                    
                    remove_edge(adj, u, v)
                    
                    # Candidates for u
                    candidates_u = []
                    for x in range(n):
                        if x == u or x == v: continue
                        if not (adj[u] & (1 << x)):
                            if is_valid_add(u, x, adj):
                                candidates_u.append(x)
                    
                    # Candidates for v
                    candidates_v = []
                    for y in range(n):
                        if y == u or y == v: continue
                        if not (adj[v] & (1 << y)):
                            if is_valid_add(v, y, adj):
                                candidates_v.append(y)
                    
                    found = False
                    best_x = -1
                    best_y = -1
                    
                    # Limit candidates to save time
                    for x in candidates_u[:30]:
                        if found: break
                        add_edge(adj, u, x)
                        for y in candidates_v[:30]:
                            if y == x: continue
                            if is_valid_add(v, y, adj):
                                found = True
                                best_x = x
                                best_y = y
                                break
                        if not found:
                            remove_edge(adj, u, x)
                    
                    if found:
                        if best_y != -1:
                            add_edge(adj, v, best_y)
                        improved = True
                    else:
                        add_edge(adj, u, v)

    def perturbation(adj, k):
        edges = adj_to_edges(adj)
        random.shuffle(edges)
        for i in range(min(k, len(edges))):
            u, v = edges[i]
            remove_edge(adj, u, v)

    best_edges = 0
    best_adj = None

    # ILS Loop
    while time.time() < deadline:
        # 1. Initialize
        adj = [0] * n
        order = all_pairs[:]
        random.shuffle(order)
        
        # Greedy build
        for u, v in order:
            if time.time() > deadline: break
            if not (adj[u] & (1 << v)):
                if is_valid_add(u, v, adj):
                    add_edge(adj, u, v)
        
        # 2. Local Search
        local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj[:]
        
        # 3. Perturbation and Re-optimize
        # Try a few perturbations
        for _ in range(10):
            if time.time() > deadline: break
            
            adj_copy = adj[:]
            perturbation(adj_copy, 8) # Remove 8 edges
            local_search(adj_copy)
            
            cnt = count_edges(adj_copy)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = adj_copy
            elif cnt > count_edges(adj):
                # Accept if better than current
                adj = adj_copy
            else:
                # Keep current adj as base for next perturbation?
                pass

    if best_adj is None:
        return []
        
    return adj_to_edges(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
