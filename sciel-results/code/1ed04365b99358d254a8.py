import random
import time
import math

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 40
    
    # Bitmask utilities
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
        """
        Check if adding (u,v) creates a C6.
        A C6 is u-a-b-c-d-v with distinct vertices.
        This requires a path of length 4 from u to v in G - (u,v).
        """
        # Neighbors of u, excluding v
        nu = adj[u] & ~(1 << v)
        # Neighbors of v, excluding u
        nv = adj[v] & ~(1 << u)
        
        # We look for a-b-c-d such that a in N(u), d in N(v),
        # and path a-b-c-d exists in G - {u,v}.
        
        # Iterate a
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a
            
            # b in N(a) excluding u, a, v
            forbidden_b = (1 << u) | (1 << a) | (1 << v)
            bits_b = adj[a] & ~forbidden_b
            
            while bits_b:
                lsb_b = bits_b & -bits_b
                b = lsb_b.bit_length() - 1
                bits_b ^= lsb_b
                
                # c in N(b) excluding u, v, a, b
                forbidden_c = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                bits_c = adj[b] & ~forbidden_c
                
                while bits_c:
                    lsb_c = bits_c & -bits_c
                    c = lsb_c.bit_length() - 1
                    bits_c ^= lsb_c
                    
                    # d must be in N(c) and N(v), and d not in {u, v, a, b, c}
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
    best_edges = 0
    best_adj = None

    def greedy_build(order, adj):
        for u, v in order:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if is_valid_add(u, v, adj):
                    add_edge(adj, u, v)

    def local_search_add(adj):
        """Try to add any valid edge until none remain."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_valid_add(u, v, adj):
                            add_edge(adj, u, v)
                            changed = True
                            if time.time() > deadline:
                                return

    def try_improve(adj):
        """
        Try to find a move that increases edge count by 1.
        Strategy: Remove one edge (u,v), then try to add two edges.
        """
        edges = adj_to_edges(adj)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                return False
            
            # Temporarily remove (u,v)
            remove_edge(adj, u, v)
            
            # Try to find two edges to add
            # Candidates for u: x such that (u,x) is valid
            # Candidates for v: y such that (v,y) is valid
            # Also candidates for other pairs? 
            # Let's stick to edges incident to u or v first, as they are the most likely to become valid after removal.
            
            candidates_u = []
            for x in range(n):
                if x == u or x == v:
                    continue
                if not (adj[u] & (1 << x)):
                    if is_valid_add(u, x, adj):
                        candidates_u.append(x)
            
            candidates_v = []
            for y in range(n):
                if y == u or y == v:
                    continue
                if not (adj[v] & (1 << y)):
                    if is_valid_add(v, y, adj):
                        candidates_v.append(y)
            
            # Try combinations (u,x) and (v,y)
            found = False
            best_x = -1
            best_y = -1
            
            # To save time, limit candidates
            for x in candidates_u[:10]:
                if found:
                    break
                add_edge(adj, u, x)
                for y in candidates_v[:10]:
                    if y == x:
                        continue
                    if is_valid_add(v, y, adj):
                        found = True
                        best_x = x
                        best_y = y
                        break
                if not found:
                    remove_edge(adj, u, x)
            
            if found:
                # We have (u, best_x) added. Add (v, best_y).
                if best_y != -1:
                    add_edge(adj, v, best_y)
                return True
            else:
                # Restore (u,v)
                add_edge(adj, u, v)
        
        return False

    # Main Search Loop
    # We use multiple restarts. Each restart does:
    # 1. Greedy build with random order
    # 2. Local search add
    # 3. Iterative improvement (remove 1, add 2)
    
    rng = random.Random(42)
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        order = all_pairs[:]
        rng.shuffle(order)
        
        greedy_build(order, adj)
        local_search_add(adj)
        
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj[:]
        
        # Try to improve further
        improved = True
        while improved and time.time() < deadline:
            improved = try_improve(adj)
            if improved:
                cnt = count_edges(adj)
                if cnt > best_edges:
                    best_edges = cnt
                    best_adj = adj[:]
        
        # If we have time, try a different heuristic for the next restart
        # e.g., sort by degree potential?
        if r % 10 == 0:
            # Random restart is good. 
            pass

    if best_adj is None:
        return []
        
    return adj_to_edges(best_adj)
