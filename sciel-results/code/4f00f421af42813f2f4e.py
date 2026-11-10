import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # LUW construction parameters
    # Girth of LUW graph CD(d, q) is at least 2d + 2 (bipartite, so even girth).
    # We need girth >= g.
    # If g is odd, we need girth >= g+1 (next even number).
    # So 2d + 2 >= g if g even, 2d + 2 >= g+1 if g odd.
    # Simplified: 2d + 2 >= g + (g % 2)
    target_girth = g if g % 2 == 0 else g + 1
    d = max(1, (target_girth - 2) // 2)
    
    # We want to minimize n = 2 * q^d.
    # So we try small q first.
    # q must be a prime power for GF(q), but for small q (2,3,4,5) we can just use modular arithmetic or specific tables.
    # For this task, we assume q is small enough that we can iterate.
    # Note: For LUW, q should be at least 2.
    
    best_n = float('inf')
    best_edges = []
    
    # Try q from 2 to 10
    for q in range(2, 11):
        if time.time() > deadline - 5:
            break
            
        # Define field arithmetic
        if q == 4:
            # GF(4) representation: 0, 1, 2, 3
            # 1 is identity. 2, 3 are roots of x^2+x+1=0?
            # Let's use a simple table for GF(4)
            # 0: 0
            # 1: 1
            # 2: alpha
            # 3: alpha+1
            # alpha^2 = alpha + 1 = 3
            # alpha^3 = 1
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        elif q == 2:
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return x & y
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        # Define rules r_c = (a_c, b_c) for c = 2..d
        # Indices 1..d are used.
        # Standard pattern:
        rule = {}
        for c in range(2, d + 1):
            # Ensure a_c, b_c are in 1..c-1
            # Pattern: alternate (1, c-1) and (c-1, 1)
            if c % 2 == 0:
                rule[c] = (1, c - 1)
            else:
                rule[c] = (c - 1, 1)
        
        # Construct graph
        # Vertices: (side, x1, ..., xd) where side in {0,1}
        # We map (side, x_tuple) to integer index
        # Total vertices: 2 * q^d
        
        # Precompute neighbors to speed up construction and checking
        # But n is small, we can just generate on fly or store list.
        
        index = {}
        order = []
        
        def get_index(vertex):
            if vertex not in index:
                index[vertex] = len(order)
                order.append(vertex)
            return index[vertex]
        
        # Start vertex
        start_vertex = (0, tuple([0] * d))
        get_index(start_vertex)
        
        # Generate all vertices?
        # It's easier to just iterate all possible tuples.
        # There are 2 * q^d vertices.
        # For q=4, d=4, 512 vertices.
        # For q=5, d=4, 1250 vertices.
        
        all_vertices = []
        for side in range(2):
            for x in range(q**d):
                # Convert x to base-q digits of length d
                digits = []
                temp = x
                for _ in range(d):
                    digits.append(temp % q)
                    temp //= q
                # digits are in reverse order?
                # Let's say x_i corresponds to digit i.
                # We need a consistent mapping.
                # Let's just use tuple of digits.
                all_vertices.append((side, tuple(digits)))
        
        # Map vertex to index
        for i, v in enumerate(all_vertices):
            index[v] = i
        
        n = len(all_vertices)
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        
        for u_idx, u in enumerate(all_vertices):
            side, x = u
            # x is tuple of length d
            # Neighbors:
            # For each y1 in GF(q)
            for y1 in range(q):
                y = [0] * (d + 1) # 1-indexed for convenience in rules
                y[1] = y1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # Point to Line?
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add(x[c-1], mul(y[a], x[b-1]))
                    else:
                        # Line to Point?
                        # y[c] = x[c] - x[a] * y[b]
                        y[c] = sub(x[c-1], mul(x[a-1], y[b]))
                
                v_vertex = (1 - side, tuple(y[1:]))
                v_idx = index[v_vertex]
                
                # Add edge (u_idx, v_idx)
                # To avoid duplicates, only add if u_idx < v_idx?
                # Or just add to adj list for both.
                adj[u_idx].append(v_idx)
        
        # Check degree
        # Each vertex should have degree k=q?
        # The LUW graph is q-regular.
        # If q != k, this construction is not directly applicable for degree k?
        # The task says "evaluator passes this task's degree k".
        # The LUW graph CD(d, q) is q-regular.
        # So we MUST have q = k.
        # If k=5, we MUST use q=5.
        # If k=3, we MUST use q=3.
        
        if q != k:
            continue
        
        # If q == k, we have a k-regular graph.
        # Check girth.
        
        # BFS Girth Check
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > deadline - 2:
                break
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        # We can break early if min_cycle < g
                        # But we need to be sure.
        
        if min_cycle >= g:
            # Valid graph
            # Construct edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if n < best_n:
                best_n = n
                best_edges = edges
            
            # Since we iterate q from small to large, and n=2*q^d is increasing with q,
            # the first valid q gives the smallest n.
            # So we can return immediately.
            return n, edges
        
    # If no valid graph found (should not happen for small k,g with q=k)
    # Return the best found or raise error
    if best_edges:
        return best_n, best_edges
    else:
        # Fallback: return empty or raise
        # For safety, return a trivial graph if possible? No, must be valid.
        # But for k=5, g=9, q=5, d=4 should work.
        raise ValueError("No valid graph found")

