import time
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses optimized Bron-Kerbosch with bitset adjacency and greedy coloring.
    """
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s limit
    
    # Build candidate set: numbers in [1,255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    idx_map = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency as list of Python ints (bitsets)
    adj = [0] * n
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            cj = candidates[j]
            if gcd(ci, cj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Full mask
    full_mask = (1 << n) - 1
    
    # Greedy coloring: returns (order, color_of) where order is vertex order,
    # color_of[v] is the color assigned. Vertices with higher colors come later in order.
    def greedy_color(P):
        """
        Greedy sequential coloring of subgraph induced by P.
        Returns (order_list, color_dict) where order_list is in the order processed,
        and color_dict[v] is the color (1-indexed) assigned to v.
        The maximum color gives an upper bound on clique size in P.
        """
        if P == 0:
            return [], {}
        
        # Extract vertices from P
        verts = []
        tmp = P
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            verts.append(v)
            tmp ^= lsb
        
        # Sort by degree in P (descending) for better coloring quality
        verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
        
        color = {}
        order = []
        for v in verts:
            # Find minimum color not used by neighbors already colored
            used = set()
            for u in order:
                if (adj[v] >> u) & 1:
                    used.add(color[u])
            c = 1
            while c in used:
                c += 1
            color[v] = c
            order.append(v)
        
        return order, color
    
    best_clique = [set()]
    
    def bron_kerbosch(P, X, R):
        """
        P, X: bitmasks
        R: set of vertices in current clique
        """
        if time.time() > deadline:
            return
        
        if P == 0 and X == 0:
            if len(R) > len(best_clique[0]):
                best_clique[0] = set(R)
            return
        
        order, color = greedy_color(P)
        if not order:
            return
        
        # Process in reverse order (highest color first)
        for idx in range(len(order) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = order[idx]
            # Upper bound: if current size + color[v] <= best, prune
            if len(R) + color[v] <= len(best_clique[0]):
                break
            
            # Add v to clique
            new_R = R | {v}
            # Recurse: neighbors of v in P
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(new_P, new_X, new_R)
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
    
    # Initial solution: all multiples of 3 (85 elements)
    mult3 = set(range(3, 256, 3))
    best_clique[0] = mult3.copy()
    
    # Run Bron-Kerbosch
    bron_kerbosch(full_mask, 0, set())
    
    # Ensure 255 is in the result
    if 255 not in best_clique[0]:
        best_clique[0].add(255)
        # Remove any element not connected to 255
        to_remove = [x for x in best_clique[0] if x != 255 and gcd(x, 255) == 1]
        for x in to_remove:
            best_clique[0].discard(x)
    
    # Local search: greedy addition
    current = set(best_clique[0])
    improved = True
    while improved:
        improved = False
        for v in candidates:
            if v in current:
                continue
            vi = idx_map[v]
            ok = True
            for u in current:
                if u == v:
                    continue
                ui = idx_map[u]
                if not ((adj[vi] >> ui) & 1):
                    ok = False
                    break
            if ok:
                current.add(v)
                improved = True
                if len(current) > len(best_clique[0]):
                    best_clique[0] = current.copy()
    
    # Local search: 1-swap improvements
    for iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        current = set(best_clique[0])
        
        for u in list(current):
            if time.time() > deadline:
                break
            ui = idx_map[u]
            for v in candidates:
                if v in current:
                    continue
                vi = idx_map[v]
                # Check if v connects to all in current except u
                ok = True
                for w in current:
                    if w == u:
                        continue
                    wi = idx_map[w]
                    if not ((adj[vi] >> wi) & 1):
                        ok = False
                        break
                if ok:
                    new_set = (current - {u}) | {v}
                    if len(new_set) > len(best_clique[0]):
                        best_clique[0] = new_set
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    # Final greedy addition
    current = set(best_clique[0])
    for v in candidates:
        if v not in current:
            vi = idx_map[v]
            ok = True
            for u in current:
                if u == v:
                    continue
                ui = idx_map[u]
                if not ((adj[vi] >> ui) & 1):
                    ok = False
                    break
            if ok:
                current.add(v)
    if len(current) > len(best_clique[0]):
        best_clique[0] = current
    
    # Verify 255 is present
    if 255 not in best_clique[0]:
        best_clique[0].add(255)
    
    return list(best_clique[0])
