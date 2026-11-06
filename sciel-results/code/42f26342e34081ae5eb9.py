import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    q = 5
    # Generate vertices of PG(2, q)
    vertices = []
    for i in range(q):
        for j in range(q):
            vertices.append((1, i, j))
    for k in range(q):
        vertices.append((0, 1, k))
    vertices.append((0, 0, 1))
    
    n = len(vertices)  # 31
    
    # Build adjacency list and masks
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # We have 90 edges. Find the non-edge that adds the fewest C4s.
    best_cost = float('inf')
    best_edge = None
    
    # Precompute non-edges
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_mask[u] & (1 << v)):
                non_edges.append((u, v))
                
    for u, v in non_edges:
        # Count C4s added: sum_{x in N(u)} |N(x) & N(v)|
        cost = 0
        mask_v = adj_mask[v]
        mask_u = adj_mask[u]
        # Iterate through set bits of mask_u
        m = mask_u
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m -= lsb
            # Count common neighbors of x and v
            common = (adj_mask[x] & mask_v).bit_count()
            cost += common
        if cost < best_cost:
            best_cost = cost
            best_edge = (u, v)
            
    if best_edge:
        u, v = best_edge
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Current C4 count
    def count_c4s():
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = (adj_mask[i] & adj_mask[j]).bit_count()
                if c >= 2:
                    total += c * (c - 1) // 2
        return total

    current_c4 = count_c4s()
    best_c4 = current_c4
    best_edges = [list(e) for e in edges]
    
    # Local Search
    # We perform best-improvement swaps
    # To speed up, we estimate delta quickly
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_swap = None
        
        # List of current edges as indices
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        
        # Recalculate non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    non_edges.append((u, v))
                    
        # Try all swaps
        # Optimization: Only try swaps that seem promising?
        # For now, try all, but break if time is tight.
        
        for ei, e in enumerate(edges):
            a, b = e[0], e[1]
            # Cost of removing (a,b)
            # C4s containing (a,b)
            # Path a-x-y-b
            lost = 0
            mask_a = adj_mask[a]
            mask_b = adj_mask[b]
            m = mask_a
            while m:
                lsb = m & -m
                x = lsb.bit_length() - 1
                m -= lsb
                if x == b: continue
                # y in N(b) \ {a, x} such that x ~ y
                # common = N(x) & N(b)
                # remove a, x if present
                cm = adj_mask[x] & mask_b
                if a != x: cm &= ~(1 << a)
                cm &= ~(1 << x)
                lost += cm.bit_count()
                
            # Try adding each non-edge (c, d)
            for c, d in non_edges:
                if c == a or c == b or d == a or d == b:
                    continue
                
                # Cost of adding (c,d)
                gained = 0
                mask_c = adj_mask[c]
                mask_d = adj_mask[d]
                m_c = mask_c
                while m_c:
                    lsb = m_c & -m_c
                    x = lsb.bit_length() - 1
                    m_c -= lsb
                    if x == d: continue
                    cm = adj_mask[x] & mask_d
                    if c != x: cm &= ~(1 << c)
                    cm &= ~(1 << x)
                    gained += cm.bit_count()
                    
                delta = gained - lost
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (ei, c, d)
                    
        if best_swap and best_delta < 0:
            ei, c, d = best_swap
            a, b = edges[ei][0], edges[ei][1]
            # Apply swap
            edges[ei] = [c, d]
            
            # Update masks
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            adj_mask[c] |= (1 << d)
            adj_mask[d] |= (1 << c)
            
            current_c4 += best_delta
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = [list(e) for e in edges]
                improved = True

    return best_edges
