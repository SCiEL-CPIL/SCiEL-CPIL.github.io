import time
import random

DEADLINE = 140.0

def _construct_andrasfai_k5():
    """Construct the Andrásfai graph A_5 on 25 vertices.
    
    A_5 is the circulant graph on Z_25 where i~j iff (j-i) mod 25 is in {1,2,4,8,16,32,64,128} mod 25.
    Actually, A_k is defined on Z_{k^2} with i~j iff j-i is a quadratic residue mod k^2 that is NOT a square mod k.
    For k=5: residues mod 25 that are squares mod 5 but not squares mod 25... 
    Simpler: A_5 has vertices 0..24, edge (i,j) iff (j-i) mod 25 in S where S = {1,2,3,4,6,7,9,11,14,16,18,21,22,23,24}
    Actually let me use the standard definition: A_k on Z_{k^2}, i~j iff j-i mod k^2 is a nonzero square mod k^2 
    but NOT a square mod k. For k=5, squares mod 25: {0,1,4,6,9,11,14,16,19,21,24}. 
    Squares mod 5: {0,1,4}. So non-squares mod 5 that are squares mod 25: none of {1,4} work since they're squares mod 5.
    
    Let me just use the known structure: A_5 is the complement of the line graph of K_5,5? No.
    
    Actually, the simplest correct construction: A_5 is the graph on Z_25 where i~j iff 
    (j-i) mod 5 is in {1,2,3,4} AND (j-i) mod 25 is a specific set. 
    
    Let me use a different approach: construct A_5 directly as the unique (up to iso) triangle-free 
    graph on 25 vertices with degree 8 and alpha 5. I'll use the circulant: 
    A_5 = Circ(25, {1,2,4,8,16, 24,23,21,17,9})
    Check: this is symmetric. Let me verify it's triangle-free and alpha=5 by construction knowledge.
    
    Actually, the standard A_k: vertices are pairs (x,y) in Z_k x Z_k. Edge between (x1,y1) and (x2,y2) 
    iff x1 != x2 and y2 - y1 = x1*x2 mod k.
    """
    k = 5
    n = k * k
    adj = [0] * n
    for i in range(n):
        x1 = i // k
        y1 = i % k
        for j in range(i + 1, n):
            x2 = j // k
            y2 = j % k
            if x1 != x2:
                if (y2 - y1) % k == (x1 * x2) % k:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj, n

def _has_triangle_fast(adj, n):
    """Fast triangle check using bitsets."""
    for i in range(n):
        ai = adj[i]
        # Only check triangles where i < j < k
        for j in range(i + 1, n):
            if not (ai >> j) & 1:
                continue
            common = ai & adj[j]
            # Check if any common neighbor k > j
            for k in range(j + 1, n):
                if (common >> k) & 1:
                    return True
    return False

def _greedy_alpha(adj, n):
    """Greedy independent set size."""
    candidates = (1 << n) - 1
    count = 0
    while candidates:
        min_deg = n
        min_v = -1
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            c ^= lb
        if min_v == -1:
            break
        count += 1
        candidates &= ~adj[min_v]
        candidates &= ~(1 << min_v)
    return count

def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set via branch and bound. Returns size if < limit, else >= limit."""
    best = [0]
    
    def bnb(chosen_count, candidates):
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        
        # Branch: exclude v
        bnb(chosen_count, rest)
        
        # Branch: include v
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)
        
    bnb(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_fast(adj, n):
        return False
    if _greedy_alpha(adj, n) >= 10:
        return False
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True

def _construct_extended_a5():
    """Construct a triangle-free graph on 40 vertices with alpha < 10.
    
    Strategy: Take A_5 (25 vertices, alpha=5, triangle-free).
    Blow up some vertices to create more vertices while maintaining triangle-freeness.
    
    Key insight: If we replace vertex v in A_5 with an independent set of size s_v, 
    and add edges between the blown-up sets according to the original edges, 
    we get a triangle-free graph if and only if the original is triangle-free 
    AND we don't create triangles within or across blown-up sets.
    
    But blowing up a vertex v to an independent set of size s means:
    - The s copies of v are not connected to each other (independent)
    - Each copy connects to all neighbors of v in the original
    - This can create triangles if two neighbors of v are also adjacent
    
    So simple blow-up doesn't preserve triangle-freeness in general.
    
    Alternative: Use a "lexicographic product" style or a different construction.
    
    Better approach: Use the fact that A_5 has a specific structure. 
    Let me try a different deterministic construction.
    
    Actually, let me try: take two copies of A_5 and connect them in a specific way.
    Or: use a circulant graph on 40 vertices.
    
    Let me try a circulant: Circ(40, S) where S is a symmetric set.
    For triangle-freeness: no three elements a,b,c in S such that a+b=0, a+c=0, b+c=0 mod 40 
    (i.e., no {a,b,c} subset of S with a+b+c=0 mod 40, considering all sign combinations).
    
    Actually for circulant graph on n vertices with connection set S (symmetric, 0 not in S),
    triangle-free means: there are no x,y,z such that x-y, y-z, z-x are all in S.
    Equivalently, no a,b in S such that a-b is also in S (with appropriate interpretation).
    
    Let me try a random search over circulants with local optimization.
    """
    n = 40
    
    # Try random maximal triangle-free graphs with local search
    best_adj = None
    best_alpha = 100
    
    for seed in range(500):
        if time.time() > time.time() + 0.1:  # This is a placeholder
            break
            
    # Let me use a smarter approach: start from A_5 and extend
    adj5, n5 = _construct_andrasfai_k5()
    
    # We want to go from 25 to 40 vertices. Add 15 more vertices.
    # Strategy: for each new vertex, connect it to a subset of existing vertices 
    # such that no triangle is formed.
    
    adj = [0] * 40
    # Copy A_5
    for i in range(25):
        adj[i] = adj5[i]
    
    # Add vertices 25-39
    # For each new vertex v, find a maximal set of neighbors in 0..24 such that 
    # no triangle is formed.
    
    for v in range(25, 40):
        # Start with no edges from v
        # Try to add edges to existing vertices
        # To avoid triangles: if we add edge (v, u), then for any w adjacent to u, 
        # we cannot add edge (v, w) if u and w are adjacent... 
        # Actually: triangle v-u-w exists if v~u, u~w, w~v.
        # So when adding v, we need to ensure that for any pair (u,w) both adjacent to v, 
        # u and w are NOT adjacent.
        # Equivalently: the set of neighbors of v must be an independent set in the existing graph.
        
        # So: find a large independent set in the existing graph (vertices 0..v-1) 
        # and connect v to all of them.
        
        # Greedy: pick vertices one by one
        neighbors = 0
        candidates = (1 << v) - 1  # vertices 0..v-1
        
        # To maximize the independent set, use greedy with min-degree heuristic
        while candidates:
            # Find vertex with min degree in candidates
            min_deg = v
            min_u = -1
            c = candidates
            while c:
                lb = c & (-c)
                u = lb.bit_length() - 1
                deg = (adj[u] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_u = u
                c ^= lb
            if min_u == -1:
                break
            neighbors |= (1 << min_u)
            candidates &= ~adj[min_u]
            candidates &= ~(1 << min_u)
        
        # Connect v to all neighbors
        adj[v] = neighbors
        for u in range(v):
            if (neighbors >> u) & 1:
                adj[u] |= (1 << v)
    
    return adj, 40

def _local_search(adj, n, max_iter=1000):
    """Try to reduce independence number by adding/removing edges while maintaining triangle-freeness."""
    adj = adj[:]  # copy
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding edges
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    # Check if adding (i,j) creates a triangle
                    common = adj[i] & adj[j]
                    if common == 0:
                        # Adding is safe for triangle-freeness
                        # Check if it reduces alpha
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        alpha_new = _greedy_alpha(adj, n)
                        if alpha_new < _greedy_alpha(adj, n):
                            improved = True
                        else:
                            # Revert
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
        
        if not improved:
            break
    
    return adj

def construct():
    start_time = time.time()
    
    # First, try the deterministic extended A_5 construction
    adj, n = _construct_extended_a5()
    
    # Validate
    if _is_valid(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # If the deterministic construction fails validation, try random approaches
    for attempt in range(200):
        if time.time() > start_time + DEADLINE - 5:
            break
        
        # Random maximal triangle-free graph
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if _is_valid(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: try smaller n
    for n_try in range(39, 20, -1):
        if time.time() > start_time + DEADLINE - 5:
            break
        for attempt in range(100):
            if time.time() > start_time + DEADLINE - 5:
                break
            adj = [0] * n_try
            pairs = [(i, j) for i in range(n_try) for j in range(i + 1, n_try)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if _is_valid(adj, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
    
    return {'n': 0, 'edges': []}
