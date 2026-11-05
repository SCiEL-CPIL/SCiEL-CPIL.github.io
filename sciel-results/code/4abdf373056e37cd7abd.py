import time
import random

def construct():
    """
    Returns a valid 2-coloring of K_n that is K_{2,2,2}-free in both colors.
    Attempts to find a coloring for n=30 via local search.
    Falls back to n=29 (Paley graph) if not found in time.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def get_paley_29():
        q = 29
        squares = set()
        for x in range(1, q):
            squares.add((x * x) % q)
        colors = []
        for i in range(q):
            for j in range(i + 1, q):
                d = (j - i) % q
                if d in squares:
                    colors.append(1)
                else:
                    colors.append(0)
        return {'n': q, 'colors': colors}

    # Try to find n=30
    n = 30
    # Initialize adjacency as list of lists or bitmasks
    # adj[v][w] is 0 or 1
    # We use a list of bitmasks for speed
    # adj0[v] is bitmask of neighbors in color 0
    # adj1[v] is bitmask of neighbors in color 1
    
    # Start with Paley 29
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
        
    adj0 = [0] * n
    adj1 = [0] * n
    
    for i in range(q):
        for j in range(i + 1, q):
            d = (j - i) % q
            if d in squares:
                # Color 1
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            else:
                # Color 0
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
                
    # Initialize vertex 29 edges randomly
    for i in range(q):
        if random.random() < 0.5:
            adj0[29] |= (1 << i)
            adj0[i] |= (1 << 29)
        else:
            adj1[29] |= (1 << i)
            adj1[i] |= (1 << 29)
            
    # Function to count K222 containing edge (u,v) in color c
    # c is 0 or 1
    def count_k222_edge(u, v, c):
        if c == 0:
            adj = adj0
            other_adj = adj1
        else:
            adj = adj1
            other_adj = adj0
            
        # S = N_c(u) intersect N_c(v)
        # Note: u and v are not neighbors in the "parts" sense, 
        # but in the graph sense, the edge (u,v) exists in color c.
        # The formula derived:
        # Sum over a' in N_c(v) \ {u}, b' in N_c(u) \ {v}
        # such that (a', b') is in color c
        # of C(|N_c(a') intersect N_c(b') intersect S|, 2)
        
        S = adj[u] & adj[v]
        # Remove u and v from S? 
        # u is not in adj[u]. v is not in adj[v].
        # So u, v are not in S.
        
        count = 0
        # Iterate a'
        # a' must be in adj[v]
        # a' != u
        # a' != v (v is not in adj[v])
        
        # Get list of a' candidates
        # We can iterate through bits
        a_candidates = []
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            if a != u:
                a_candidates.append(a)
            temp -= lsb
            
        b_candidates = []
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            if b != v:
                b_candidates.append(b)
            temp -= lsb
            
        for a in a_candidates:
            # Check if (a, b) edge exists in color c for each b
            # Actually, we need to sum over b as well
            # Optimize: For fixed a, we need b in adj[u] such that b in adj[a]
            # and b != v
            # And then compute intersection size
            
            # Let's just iterate b for each a
            for b in b_candidates:
                if a == b: continue
                # Check if (a, b) is in color c
                if (adj[a] >> b) & 1:
                    # Calculate intersection size
                    inter = adj[a] & adj[b] & S
                    size = inter.bit_count()
                    if size >= 2:
                        # C(size, 2)
                        count += size * (size - 1) // 2
                        
        return count

    def total_cost():
        # Sum over all edges (u,v) in color 0 of count_k222_edge(u,v,0)
        # Wait, this counts each K222 multiple times?
        # A K222 has 12 edges.
        # If we sum count_k222_edge for all 12 edges, we count each K222 12 times.
        # So Total K222 = (Sum over all edges e in color c of count_k222_edge(e, c)) / 12
        # We just want to minimize the sum.
        
        cost = 0
        for u in range(n):
            for v in range(u + 1, n):
                # Determine color of (u,v)
                if (adj0[u] >> v) & 1:
                    c = 0
                else:
                    c = 1
                cost += count_k222_edge(u, v, c)
        return cost

    # Initial cost
    current_cost = total_cost()
    
    if current_cost == 0:
        # Found immediately? Unlikely but possible
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj0[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}

    # Local Search
    best_cost = current_cost
    best_adj0 = adj0[:]
    best_adj1 = adj1[:]
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        # Current color
        if (adj0[u] >> v) & 1:
            c = 0
        else:
            c = 1
            
        # Delta cost
        # Remove contribution of (u,v) in color c
        # Add contribution of (u,v) in color 1-c
        cost_remove = count_k222_edge(u, v, c)
        
        # Flip
        if c == 0:
            adj0[u] &= ~(1 << v)
            adj0[v] &= ~(1 << u)
            adj1[u] |= (1 << v)
            adj1[v] |= (1 << u)
        else:
            adj1[u] &= ~(1 << v)
            adj1[v] &= ~(1 << u)
            adj0[u] |= (1 << v)
            adj0[v] |= (1 << u)
            
        cost_add = count_k222_edge(u, v, 1 - c)
        
        delta = cost_add - cost_remove
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj0 = adj0[:]
                best_adj1 = adj1[:]
                if best_cost == 0:
                    colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (best_adj0[i] >> j) & 1:
                                colors.append(0)
                            else:
                                colors.append(1)
                    return {'n': n, 'colors': colors}
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < pow(2.718, -delta / T):
                current_cost = new_cost
            else:
                # Revert
                if c == 0:
                    adj0[u] |= (1 << v)
                    adj0[v] |= (1 << u)
                    adj1[u] &= ~(1 << v)
                    adj1[v] &= ~(1 << u)
                else:
                    adj1[u] |= (1 << v)
                    adj1[v] |= (1 << u)
                    adj0[u] &= ~(1 << v)
                    adj0[v] &= ~(1 << u)
                    
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return best found or fallback
    if best_cost == 0:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj0[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
    else:
        return get_paley_29()
