"""Program for R(K2,2,2) maximization using bitmask constraint satisfaction."""
import numpy as np
import random
import time

def construct():
    # We want to maximize n such that there exists a 2-coloring of K_n
    # with no monochromatic K_{2,2,2}.
    
    def has_k222_bitmask(n, adj_bitmasks):
        """
        Check if the graph defined by adj_bitmasks (color 1) contains K_{2,2,2}.
        adj_bitmasks[i] is a bitmask of neighbors of i in color 1.
        """
        # For each pair (u, v) in color 1:
        for u in range(n):
            # Only consider v > u where edge (u,v) is color 1
            # adj_bitmasks[u] has bit v set if color 1
            mask_v = adj_bitmasks[u]
            # Iterate over set bits in mask_v
            # We can just iterate v from u+1 to n-1
            for v in range(u + 1, n):
                if not (mask_v & (1 << v)):
                    continue
                # Common neighbors of u and v in color 1
                common = adj_bitmasks[u] & adj_bitmasks[v]
                # Remove u and v from common (though they shouldn't be there usually, 
                # but just in case of self-loops or if we include them)
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # Check if common contains a K_{2,2} in the color-1 subgraph.
                # This is equivalent to: there exist a, b in common such that
                # (adj_bitmasks[a] & adj_bitmasks[b] & common) has at least 2 bits set.
                # Because if a,b have common neighbors c,d in common, 
                # then {u,v}, {a,b}, {c,d} form K_{2,2,2}.
                
                # Efficient check for K_{2,2} in induced subgraph on 'common'
                # Let C = common.
                # We need to check if there are a, b in C such that |N(a) ∩ N(b) ∩ C| >= 2.
                
                # Iterate through pairs (a, b) in C
                # To speed up, we can iterate a in C, then b in C > a.
                # Number of bits in C is popcount(common). If < 4, no K2,2.
                if common.bit_count() < 4:
                    continue
                    
                # Iterate a in C
                temp_a = common
                while temp_a:
                    # Extract lowest set bit
                    a_bit = temp_a & -temp_a
                    a = a_bit.bit_length() - 1
                    temp_a ^= a_bit
                    
                    # Iterate b in C, b > a
                    temp_b = common & ~(a_bit - 1) # bits > a
                    temp_b &= common # ensure in common
                    # Actually, just iterate all b in common, b != a
                    # It's easier to just iterate all pairs in C
                    # Let's use a list of vertices in C for this small check
                    # Since C size is small usually in valid graphs, but can be large.
                    # For n=30, max common is 28. 28*27/2 = 378 pairs.
                    # Inside, bit_count is fast.
                    
                    # Optimization: 
                    # We can just check if there exists a pair (a,b) in C 
                    # such that (adj[a] & adj[b] & C).bit_count() >= 2.
                    
                    # Let's collect vertices in C into a list for iteration
                    # This is done per (u,v) pair, which is O(n^2).
                    # Total O(n^4) worst case if we are not careful.
                    # n=30 -> 30^4 = 810,000 operations. Very fast.
                    
                    verts_in_c = []
                    t = common
                    while t:
                        bit = t & -t
                        verts_in_c.append(bit.bit_length() - 1)
                        t ^= bit
                    
                    m = len(verts_in_c)
                    if m < 4:
                        continue
                        
                    found = False
                    for i in range(m):
                        a = verts_in_c[i]
                        for j in range(i + 1, m):
                            b = verts_in_c[j]
                            # Check if a and b have at least 2 common neighbors in C
                            inter = adj_bitmasks[a] & adj_bitmasks[b] & common
                            if inter.bit_count() >= 2:
                                return True
                    # If no pair found, continue to next (u,v)
        return False

    def check_valid(n, colors):
        """
        Check if both color classes are K_{2,2,2}-free.
        colors is a list of 0/1 for edges in lexicographic order.
        """
        # Build bitmasks for color 1
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 1:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        if has_k222_bitmask(n, adj1):
            return False
            
        # Build bitmasks for color 0
        adj0 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                idx += 1
        
        if has_k222_bitmask(n, adj0):
            return False
            
        return True

    def count_violations(n, colors):
        """
        Count the number of monochromatic K_{2,2,2} subgraphs.
        """
        adj1 = [0] * n
        adj0 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 1:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                else:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                idx += 1
        
        def count_k222(adj):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common.bit_count() < 4:
                        continue
                    verts = []
                    t = common
                    while t:
                        bit = t & -t
                        verts.append(bit.bit_length() - 1)
                        t ^= bit
                    m = len(verts)
                    for i in range(m):
                        a = verts[i]
                        for j in range(i + 1, m):
                            b = verts[j]
                            inter = adj[a] & adj[b] & common
                            if inter.bit_count() >= 2:
                                count += 1
            return count
            
        return count_k222(adj1) + count_k222(adj0)

    # Try Paley graphs
    def try_paley(p):
        if p % 4 != 1:
            return None
        squares = {x * x % p for x in range(1, p)}
        squares.discard(0)
        colors = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    colors.append(1)
                else:
                    colors.append(0)
        if check_valid(p, colors):
            return colors
        return None

    # Local Search
    def local_search(n, max_iters=50000):
        random.seed(42)
        colors = [random.choice([0, 1]) for _ in range(n * (n - 1) // 2)]
        
        # Precompute edge indices for random access
        # idx = i*(2*n - i - 1)//2 + (j - i - 1)
        
        def get_edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (2 * n - i - 1) // 2 + (j - i - 1)
            
        current_violations = count_violations(n, colors)
        
        for it in range(max_iters):
            if current_violations == 0:
                return colors
                
            # Pick a random edge
            e = random.randint(0, n * (n - 1) // 2 - 1)
            
            # Find i, j from e
            # Inverse of idx formula
            # k = n - 1 - i
            # e = (n*k - k*(k+1)/2) + (j - i - 1) ... complicated
            # Easier: just iterate to find i,j? No, n=30, 435 edges.
            # We can precompute a list of (i,j) for each e.
            
            old_color = colors[e]
            colors[e] = 1 - old_color
            new_violations = count_violations(n, colors)
            
            if new_violations < current_violations:
                current_violations = new_violations
            else:
                # Revert
                colors[e] = old_color
                # With some probability, keep the change (simulated annealing style)
                if random.random() < 0.05:
                    pass # keep change
                else:
                    colors[e] = old_color
                    # Actually, standard WalkSAT reverts if not better.
                    # Let's just revert if not better to avoid drift, 
                    # or use a small probability to keep it.
                    # If we revert always, we might get stuck in local minima.
                    # Let's revert if worse, but allow equal?
                    if new_violations > current_violations:
                         colors[e] = old_color
                    # If equal, keep it to explore?
        
        return None

    # Precompute edge list for faster lookup in local search
    # This is done inside local_search implicitly by the formula, 
    # but count_violations is the bottleneck.
    # count_violations is O(n^4). For n=30, 30^4 = 810k. 
    # 50000 iters * 810k = 4e10 operations. Too slow in Python.
    
    # We need a faster violation count or incremental update.
    # Or reduce iterations.
    # Or use a smarter heuristic.
    
    # Let's try Paley first.
    for p in [29, 31, 17, 13, 5]:
        res = try_paley(p)
        if res:
            return {'n': p, 'colors': res}
            
    # If Paley fails for 29, try local search for 29.
    # To make local search feasible, we need to optimize count_violations.
    # We can precompute the list of K_{2,2,2} candidates? No, they change.
    
    # Alternative: Use the fact that we only need to check pairs (u,v).
    # There are ~435 pairs. For each, we do bit ops.
    # 435 * (bit_count checks) is fast.
    # 435 * (loop over common neighbors pairs) is the slow part.
    # If we limit the search or use a better heuristic, it might work.
    
    # Let's try a simpler local search with fewer iterations but faster check.
    # The check `has_k222_bitmask` returns True/False.
    # `count_violations` counts them.
    
    # Let's try to run local search for n=29 with a time limit.
    start_time = time.time()
    limit = 110 # seconds
    
    # Optimized Local Search
    def local_search_optimized(n):
        random.seed(1)
        # Start with a random coloring
        colors = [random.choice([0, 1]) for _ in range(n * (n - 1) // 2)]
        
        # Precompute edge endpoints
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                idx += 1
                
        def get_violations():
            adj1 = [0] * n
            adj0 = [0] * n
            for (i, j), c in zip(edges, colors):
                if c == 1:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                else:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                    
            def cnt(adj):
                c = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (adj[u] & (1 << v)):
                            continue
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        if common.bit_count() < 4:
                            continue
                        # Fast K22 check in common
                        # We can just check if there is any pair (a,b) in common
                        # with inter.bit_count() >= 2
                        # To speed up, we can stop early if we find one.
                        # But we need count for heuristic.
                        # For heuristic, we can just count number of (u,v) pairs that are "bad".
                        # A (u,v) is bad if common contains K22.
                        # This is a binary property per (u,v).
                        # So violation count = sum over (u,v) of [common contains K22].
                        # This is much faster to count!
                        
                        # Check if common contains K22
                        verts = []
                        t = common
                        while t:
                            bit = t & -t
                            verts.append(bit.bit_length() - 1)
                            t ^= bit
                        m = len(verts)
                        for i in range(m):
                            a = verts[i]
                            for j in range(i + 1, m):
                                b = verts[j]
                                inter = adj[a] & adj[b] & common
                                if inter.bit_count() >= 2:
                                    c += 1
                                    break
                    # end v
                # end u
                return c
                
            return cnt(adj1) + cnt(adj0)
            
        current_v = get_violations()
        if current_v == 0:
            return colors
            
        for it in range(100000):
            if time.time() - start_time > limit:
                break
            if current_v == 0:
                return colors
                
            e = random.randint(0, len(colors) - 1)
            i, j = edges[e]
            old_c = colors[e]
            colors[e] = 1 - old_c
            new_v = get_violations()
            
            if new_v < current_v:
                current_v = new_v
            else:
                # Revert
                colors[e] = old_c
                # Small chance to keep to escape local minima
                if random.random() < 0.01:
                    colors[e] = 1 - old_c
                    current_v = new_v
        return None

    for n in [29, 30]:
        res = local_search_optimized(n)
        if res:
            return {'n': n, 'colors': res}

    # Fallback
    return {'n': 1, 'colors': []}
