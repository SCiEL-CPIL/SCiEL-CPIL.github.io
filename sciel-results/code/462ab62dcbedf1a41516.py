import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def check_valid(adj, n, min_deg):
        # Check min degree
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        # Check C4
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def creates_c4(adj, u, v):
        """
        Check if adding edge (u, v) creates a C4.
        u and v are not adjacent.
        C4 is u-w-x-v-u.
        Exists if there is w in N(u) and x in N(v) such that w-x is an edge.
        This is equivalent to: for some w in N(u), N(w) intersects N(v).
        Note: if w is in N(v), then w is a common neighbor.
        If w is in N(v), then w-x with x=w? No.
        If w in N(v), then w is connected to v.
        Does w have a neighbor in N(v)?
        w is in N(v), so w is a neighbor of v.
        We need w-x edge.
        If x=w, w-w is not an edge.
        So we need x != w.
        So if N(w) intersects N(v) \ {w}, then C4.
        However, if w in N(v), then w is a common neighbor of u and v? No, w in N(u) and w in N(v).
        So w is a common neighbor.
        If there is another common neighbor, C4?
        If w1, w2 are common neighbors, u-w1-v-w2-u is not a C4 unless w1-w2 is edge?
        No, u-w1-v is path. u-w2-v is path.
        C4 requires 4 distinct vertices.
        So u, w, x, v must be distinct.
        If w is common neighbor, w != u, w != v.
        If x is common neighbor, x != u, x != v.
        If w=x, it's a triangle u-w-v.
        So we need w != x.
        
        My previous check: `adj[w] & adj[v]`.
        If this is non-zero, there is x in N(w) cap N(v).
        If x == w, then w in N(w)? No.
        So x != w.
        So x is distinct from w.
        Also x in N(v), so x != v.
        w in N(u), so w != u.
        Is x distinct from u?
        x in N(v). If x == u, then u in N(v). But u, v not adjacent. So x != u.
        So u, w, x, v are distinct.
        So the check `any( (adj[w] & adj[v]) != 0 for w in N(u) )` is correct.
        """
        nu = adj[u]
        nv = adj[v]
        mask = nu
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask -= lsb
            if adj[w] & nv:
                return True
        return False

    def solve_for_n(n, min_deg):
        best_adj = None
        
        # Try multiple random seeds
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 100000 + 42)
            
            # Initialize empty graph
            adj = [0] * n
            
            # Greedy construction: shuffle edges and add if valid
            # To speed up, we can just iterate and add if valid.
            # But we want high min degree.
            # Greedy addition of edges that don't create C4 tends to fill the graph.
            
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if creates_c4(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Check if min degree is satisfied
            min_deg_actual = min(bin(adj[i]).count('1') for i in range(n))
            
            if min_deg_actual >= min_deg:
                if check_valid(adj, n, min_deg):
                    return adj
            
            # If not, try local search to improve min degree
            # But greedy might have already maximized it?
            # No, greedy order matters.
            # We can try to add more edges if we missed some?
            # The greedy loop above adds ALL valid edges in a specific order.
            # So the result is a maximal C4-free graph.
            # Different orders yield different maximal graphs.
            # We just need one with high min degree.
            
            # If we don't reach min_deg, we can try to repair?
            # Or just try next seed.
            # Given n=46, min_deg=7 is close to the theoretical max.
            # Theoretical max for C4-free graph is approx sqrt(n)/2?
            # Actually, for n=46, max min degree of C4-free graph is 7?
            # If it is 7, then random greedy should find it with high probability.
            
        return None

    # Try n from 46 down to 40
    # n=47 is impossible (R(C4, K1,8)=47)
    for n in range(46, 39, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        result_adj = solve_for_n(n, min_deg)
        
        if result_adj is not None:
            edges = get_edges(result_adj, n)
            return {'n': n, 'edges': edges}
            
    # Fallback
    return {'n': 39, 'edges': []}
