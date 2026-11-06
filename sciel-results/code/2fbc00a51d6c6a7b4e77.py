import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def has_k4(adj, n):
        """Check if graph has a K4. Return True if found."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    m = common
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        m ^= lsb
                        if common & adj[w]:
                            return True
        return False

    def has_independent_6(adj, n):
        """Check if graph has an independent set of size 6."""
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
        def dfs(verts, depth):
            if depth == 6:
                return True
            if not verts:
                return False
            v = verts
            while v:
                lsb = v & -v
                u = lsb.bit_length() - 1
                v ^= lsb
                new_verts = v & comp_adj[u]
                if dfs(new_verts, depth + 1):
                    return True
        return dfs(full_mask, 0)

    def is_valid(n, adj):
        if has_k4(adj, n):
            return False
        if has_independent_6(adj, n):
            return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def build_35_vertex_graph():
        """
        Construct a K4-free graph on 35 vertices with independence number 5.
        This is based on the known (4,6)-Ramsey graph construction.
        We use a specific structure: take 5 disjoint 7-cycles (or similar) and connect them.
        
        Actually, the standard construction for R(4,6) > 35 is often derived from the 35-vertex
        (3,3)-Ramsey graph or similar. Let's use a known valid graph.
        
        A common construction is the "Clebsch" like or "Paley" like.
        However, the most reliable way to get a high score quickly is to hardcode a known valid graph
        if we can generate it, or use a very effective local search.
        
        Let's try a specific construction:
        Vertices 0..34.
        Partition into 5 sets of 7: S0, S1, S2, S3, S4.
        Within each Si, make it a 7-cycle (C7). C7 is K4-free and has alpha=3.
        Between Si and Sj, connect vertices based on their index mod 7.
        Specifically, connect (i, a) in Si to (j, b) in Sj if (b-a) mod 7 is in {1, 2, 4}.
        This is the Paley graph of order 7 (which is the 7-vertex Paley graph, a 3-regular graph? No, Paley(7) is 3-regular? 
        Squares mod 7: 1, 2, 4. So yes, 3-regular. It is a 7-cycle? No, it's the (7,3) graph, which is the Paley graph.
        Is Paley(7) K4-free? Yes. Alpha(Paley(7)) = 2? Let's check.
        1 is connected to 2, 4, 5 (since 1-2=6= -1? No, (2-1)=1, (4-1)=3? No.
        Squares: 1^2=1, 2^2=4, 3^2=2. So squares are {1, 2, 4}.
        Edges from 0: to 1, 2, 4.
        Edges from 1: to 2, 3, 5.
        This is the 7-vertex Paley graph. It is K4-free.
        Alpha: Independent set?
        0, 3, 5? 0-3: 3-0=3 (not sq). 0-5: 5-0=5 (not sq). 3-5: 5-3=2 (sq, edge). So not independent.
        0, 3, 6? 0-3: no. 0-6: 6-0=6 (not sq). 3-6: 6-3=3 (not sq). So {0,3,6} is independent. Size 3.
        So alpha(C7-like) = 3.
        
        Now, the connections between blocks.
        If we connect blocks with Paley(7) edges, we need to ensure no K4 is formed across blocks.
        A K4 could be 2 vertices from Si and 2 from Sj? Or 1 from Si and 3 from Sj?
        If we have 1 vertex u in Si and 3 vertices v1, v2, v3 in Sj.
        u is connected to v1, v2, v3.
        v1, v2, v3 must form a triangle in Sj to make a K4? No, they just need to be connected to each other to form K4 with u?
        If u is connected to v1, v2, v3, and v1-v2, v2-v3, v3-v1 are edges, then {u, v1, v2, v3} is a K4.
        So we need to ensure that for any u in Si, the neighbors of u in Sj do not contain a triangle.
        Since Sj is Paley(7) (3-regular), does it have triangles?
        0-1-2: 0-1 (1, sq), 1-2 (1, sq), 0-2 (2, sq). Yes, {0,1,2} is a triangle.
        So if u is connected to 0, 1, 2 in Sj, we have a K4.
        u is connected to {0,1,2} in Sj if the relative indices form squares.
        Let u be at index a in Si. It connects to b in Sj if (b-a) mod 7 in {1,2,4}.
        So u connects to {a+1, a+2, a+4} in Sj.
        Do {a+1, a+2, a+4} form a triangle in Paley(7)?
        Let's check {1, 2, 4} (shifted by a=0).
        1-2: diff 1 (sq). 2-4: diff 2 (sq). 4-1: diff -3=4 (sq).
        Yes, {1,2,4} is a triangle in Paley(7).
        So this construction creates K4s.
        
        We need a different connection rule.
        We need the neighborhood of any u in Si within Sj to be an independent set in Sj.
        Since alpha(Paley(7)) = 3, and the neighborhood size is 3, we need the neighborhood to be exactly an independent set.
        The independent sets of size 3 in Paley(7) are the "lines" or something?
        {0, 3, 6} is one.
        {1, 4, 0}? 1-4 (3, no), 4-0 (4, yes). No.
        The independent sets of size 3 are {0,3,6}, {1,4,0}? No.
        Let's list them.
        0,3,6 is one.
        1,4,0? No.
        1,4,? 1-4 no. 4-? 4-1 no.
        Actually, Paley(7) is a strongly regular graph (7,3,1,2).
        Every pair of adjacent vertices has 1 common neighbor.
        Every pair of non-adjacent vertices has 2 common neighbors.
        The independent sets of size 3 are the complements of the triangles?
        There are 7 triangles?
        The number of independent sets of size 3 is 7.
        They are {0,3,6}, {1,4,0}... wait.
        If we shift {0,3,6} by 1, we get {1,4,0}. Is {1,4,0} independent?
        1-4: diff 3 (no). 4-0: diff 4 (yes, edge). So {1,4,0} is NOT independent.
        So the independent sets are not just shifts.
        
        Let's try a different block size or connection.
        How about 5 blocks of 7, but connect them with a 1-regular or 2-regular graph?
        If we connect with a perfect matching, alpha might be too high.
        
        Alternative: Use the known 35-vertex graph from Exoo.
        Since I cannot easily generate it from scratch, I will rely on a very aggressive local search
        starting from a good seed.
        
        Let's try a seed based on the 5x7 grid with a specific pattern.
        Or, better, just use a high-quality local search with multiple restarts and a good initial guess.
        
        Initial guess: Random graph with p=0.5, then local search.
        Or, start with the 35-vertex Paley graph? Paley(37) is 37 vertices.
        Paley(37) is K4-free?
        Paley graphs are triangle-free if p is not 1 mod 4? No.
        Paley(p) has triangles.
        
        Let's stick to the local search approach but make it much faster and more robust.
        """
        return None

    def local_search(n, time_limit, seed=None):
        """Local search to find a valid graph on n vertices."""
        if seed is not None:
            random.seed(seed)
        
        start = time.time()
        deadline_ls = start + time_limit
        
        # Start with random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # We want to minimize total violations.
        # Violations = #K4 + #I6.
        # We use a greedy approach: find the edge flip that reduces violations the most.
        
        def count_k4s(adj, n):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        m = common
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            m ^= lsb
                            if common & adj[w]:
                                # This is a K4 containing u, v, w, and the neighbor in common & adj[w]
                                # Actually, this counts each K4 multiple times.
                                # We just need to know if there are any, or count them for the score.
                                # For the delta, we need the exact change.
                                count += 1
            return count

        def count_i6s(adj, n):
            # This is expensive. We only use it for the final check or in the delta if needed.
            # For the delta, we count I4s in the non-neighborhood.
            return 0

        # To speed up, we only compute the delta for the most promising edges.
        # Or we compute the delta for all edges, but make it fast.
        
        improved = True
        while improved and time.time() < deadline_ls:
            improved = False
            best_delta = 0
            best_u = -1
            best_v = -1
            
            # Evaluate all edges
            for u in range(n):
                for v in range(u+1, n):
                    is_edge = bool((adj[u] >> v) & 1)
                    # Fast delta calculation
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    # Count edges in common (for K4 destruction/creation)
                    e_count = 0
                    m = common
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        m ^= lsb
                        e_count += (common & adj[w]).bit_count()
                    e_count //= 2
                    
                    # Count independent 4-sets in non-neighborhood (for I6 creation/destruction)
                    non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
                    i4_count = 0
                    # This is the bottleneck. We need to count I4s in non_common.
                    # If non_common is small, this is fast.
                    vertices = []
                    m = non_common
                    while m:
                        lsb = m & -m
                        vertices.append(lsb.bit_length() - 1)
                        m ^= lsb
                    
                    n_v = len(vertices)
                    if n_v >= 4:
                        for i in range(n_v):
                            for j in range(i+1, n_v):
                                if (adj[vertices[i]] >> vertices[j]) & 1:
                                    continue
                                for k in range(j+1, n_v):
                                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                                        continue
                                    for l in range(k+1, n_v):
                                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                                            continue
                                        i4_count += 1
                    
                    if is_edge:
                        d = -e_count + i4_count
                    else:
                        d = e_count - i4_count
                        
                    if d < best_delta:
                        best_delta = d
                        best_u = u
                        best_v = v
                        improved = True
                        
            if best_u != -1:
                # Flip edge
                if (adj[best_u] >> best_v) & 1:
                    adj[best_u] &= ~(1 << best_v)
                    adj[best_v] &= ~(1 << best_u)
                else:
                    adj[best_u] |= (1 << best_v)
                    adj[best_v] |= (1 << best_u)
                    
                # Check if we reached a valid state early
                if not has_k4(adj, n) and not has_independent_6(adj, n):
                    return adj
        
        # If we exit the loop, check if it's valid
        if not has_k4(adj, n) and not has_independent_6(adj, n):
            return adj
        
        return None

    best_graph = None
    best_n = 0
    
    # 1. Try local search for n=35, 36, 37
    # We want the max n, so try 37 first, then 36, then 35.
    for target_n in [37, 36, 35]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time for multiple restarts
        num_restarts = 5
        time_per_restart = remaining / num_restarts
        
        found = False
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            adj = local_search(target_n, time_per_restart, seed=42 + r)
            if adj is not None:
                if is_valid(target_n, adj):
                    if target_n > best_n:
                        best_n = target_n
                        best_graph = adj
                    found = True
                    break
        
        if found:
            break

    # Fallback: if best_graph is None, try a smaller n
    if best_graph is None:
        for target_n in [34, 33, 32, 31, 30]:
            if time.time() > deadline:
                break
            
            remaining = deadline - time.time()
            if remaining < 2:
                break
            
            adj = local_search(target_n, remaining, seed=42)
            if adj is not None:
                if is_valid(target_n, adj):
                    if target_n > best_n:
                        best_n = target_n
                        best_graph = adj
                    break

    # Final fallback: Paley 17
    if best_graph is None:
        p = 17
        squares = {x * x % p for x in range(1, p)}
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_graph = adj
        best_n = p

    return {'n': best_n, 'edges': get_edges(best_graph, best_n)}

# Ensure random is available
random.seed(42)
