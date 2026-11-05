import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 18
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def is_k4_free(adj_bm):
        # Check if graph is K4-free using bitmasks
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bm[i] & (1 << j)):
                    continue
                # Common neighbors of i and j
                common = adj_bm[i] & adj_bm[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check if any two common neighbors are connected
                cm = common
                while cm:
                    lsb = cm & (-cm)
                    k = lsb.bit_length() - 1
                    cm ^= lsb
                    rest = common & (cm | (1 << k))
                    rest &= adj_bm[k]
                    if rest:
                        return False
        return True

    def max_triangle_free_size_exact(adj_bm):
        """
        Find the largest triangle-free induced subgraph size.
        Uses branch and bound with bit-parallel operations.
        """
        best = 0
        
        # Precompute for each vertex v, the set of edges in the subgraph induced by neighbors
        # Actually, we'll use a recursive search:
        # State: (candidates, current_set, current_size)
        # candidates: bitmask of vertices that can be added without creating a triangle
        
        # Key insight: A set S is triangle-free iff for every pair (u,v) in S, 
        # there is no w in S such that u-w and v-w are edges.
        
        # We'll use a different approach: enumerate all subsets? No, too slow.
        # Use recursive backtracking with pruning.
        
        # Let's try to find a large triangle-free set by starting with all vertices
        # and removing vertices that create triangles.
        
        # Actually, let's use a standard maximum independent set approach on the 
        # "triangle graph" or use a different characterization.
        
        # Better approach: Use the fact that a triangle-free graph has chromatic number <= 3?
        # No, that's not helpful directly.
        
        # Let's use a branch and bound where we decide for each vertex whether to include it.
        # We maintain the set of included vertices and check triangle-freeness incrementally.
        
        # Optimized: Use bitmask representation.
        # tri_free_mask: vertices that can be added to current set without creating a triangle
        
        full_mask = (1 << n) - 1
        
        def search(candidates, current_mask, current_size):
            nonlocal best
            
            if current_size + candidates.bit_count() <= best:
                return
            
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning: if current_size + popcount(candidates) <= best, return
            # Already checked above
            
            # Choose a vertex to branch on
            # Heuristic: pick vertex with highest degree in candidates
            v = -1
            max_deg = -1
            cm = candidates
            while cm:
                lsb = cm & (-cm)
                u = lsb.bit_length() - 1
                cm ^= lsb
                deg = (adj_bm[u] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if v == -1:
                if current_size > best:
                    best = current_size
                return
            
            # Branch 1: Include v
            # Check if adding v creates a triangle with current_mask
            # v creates a triangle if there exist u, w in current_mask such that 
            # u-v, w-v, u-w are all edges.
            # Actually, we need to check: for any edge (u,w) in current_mask, 
            # if both u and w are neighbors of v, then {u,v,w} is a triangle.
            
            # Let's check: neighbors of v in current_mask
            nv_in_current = adj_bm[v] & current_mask
            
            # Check if nv_in_current contains an edge
            # For each pair in nv_in_current, check if they are connected
            has_triangle = False
            nv_iter = nv_in_current
            while nv_iter:
                lsb = nv_iter & (-nv_iter)
                u = lsb.bit_length() - 1
                nv_iter ^= lsb
                # Check if u is connected to any other vertex in nv_in_current
                rest = nv_in_current & adj_bm[u]
                rest &= ~(1 << u)
                if rest:
                    has_triangle = True
                    break
            
            if not has_triangle:
                # Safe to include v
                new_current = current_mask | (1 << v)
                # Update candidates: remove v and any vertex that would create a triangle
                # A vertex x can be added if it doesn't form a triangle with any pair in new_current
                # This is complex. Instead, let's just remove v from candidates and recurse.
                # The triangle-freeness will be checked when we add more vertices.
                
                # Actually, we need to ensure that future additions don't create triangles.
                # A vertex x can be added if:
                # 1. x is not in current_mask
                # 2. For any edge (u,w) in current_mask, not (u-x and w-x)
                # 3. For any u in current_mask, if u-x is an edge, then x must not connect to any neighbor of u in current_mask
                # This is getting complex. Let's use a simpler pruning.
                
                # Simpler: just remove v from candidates and recurse.
                # The triangle check is done when adding vertices.
                new_candidates = candidates & ~(1 << v)
                search(new_candidates, new_current, current_size + 1)
            
            # Branch 2: Exclude v
            new_candidates = candidates & ~(1 << v)
            search(new_candidates, current_mask, current_size)
        
        search(full_mask, 0, 0)
        return best

    # Strategy: Try several known K4-free graph constructions
    # 1. Complete tripartite graph K_{6,6,6}
    # 2. Complement of perfect matching
    # 3. Other structured graphs
    
    candidates = []
    
    # Candidate 1: Complete tripartite K_{6,6,6}
    adj1 = [[0] * n for _ in range(n)]
    for i in range(6):
        for j in range(6, 12):
            adj1[i][j] = 1
            adj1[j][i] = 1
        for k in range(12, 18):
            adj1[i][k] = 1
            adj1[k][i] = 1
    for j in range(6, 12):
        for k in range(12, 18):
            adj1[j][k] = 1
            adj1[k][j] = 1
    
    adj_bm1 = [0] * n
    for i in range(n):
        for j in range(n):
            if adj1[i][j]:
                adj_bm1[i] |= (1 << j)
    
    if is_k4_free(adj_bm1):
        candidates.append((adj1, adj_bm1, "K666"))
    
    # Candidate 2: Complement of perfect matching (9 edges removed from complete graph)
    adj2 = [[1] * n for _ in range(n)]
    for i in range(n):
        adj2[i][i] = 0
    # Remove a perfect matching
    for i in range(9):
        u = 2 * i
        v = 2 * i + 1
        adj2[u][v] = 0
        adj2[v][u] = 0
    
    adj_bm2 = [0] * n
    for i in range(n):
        for j in range(n):
            if adj2[i][j]:
                adj_bm2[i] |= (1 << j)
    
    if is_k4_free(adj_bm2):
        candidates.append((adj2, adj_bm2, "ComplementMatching"))
    
    # Candidate 3: 3-partite with sizes 5,6,7
    adj3 = [[0] * n for _ in range(n)]
    # Part 0: 0-4 (5 vertices)
    # Part 1: 5-10 (6 vertices)
    # Part 2: 11-17 (7 vertices)
    for i in range(5):
        for j in range(5, 11):
            adj3[i][j] = 1
            adj3[j][i] = 1
        for k in range(11, 18):
            adj3[i][k] = 1
            adj3[k][i] = 1
    for j in range(5, 11):
        for k in range(11, 18):
            adj3[j][k] = 1
            adj3[k][j] = 1
    
    adj_bm3 = [0] * n
    for i in range(n):
        for j in range(n):
            if adj3[i][j]:
                adj_bm3[i] |= (1 << j)
    
    if is_k4_free(adj_bm3):
        candidates.append((adj3, adj_bm3, "K567"))
    
    # Candidate 4: 4-partite with sizes 4,4,5,5
    adj4 = [[0] * n for _ in range(n)]
    # Part 0: 0-3 (4)
    # Part 1: 4-7 (4)
    # Part 2: 8-12 (5)
    # Part 3: 13-17 (5)
    parts = [(0, 4), (4, 8), (8, 13), (13, 18)]
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            for i in range(parts[a][0], parts[a][1]):
                for j in range(parts[b][0], parts[b][1]):
                    adj4[i][j] = 1
                    adj4[j][i] = 1
    
    adj_bm4 = [0] * n
    for i in range(n):
        for j in range(n):
            if adj4[i][j]:
                adj_bm4[i] |= (1 << j)
    
    if is_k4_free(adj_bm4):
        candidates.append((adj4, adj_bm4, "K4455"))
    
    # Evaluate each candidate
    best_score = 18
    best_edges = []
    
    for adj, adj_bm, name in candidates:
        if time.time() > deadline:
            break
        
        score = max_triangle_free_size_exact(adj_bm)
        
        if score < best_score:
            best_score = score
            best_edges = get_edges_from_adj(adj)
    
    # If no candidates or all have high score, try a few random graphs with K4 removal
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            adj_rand = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_rand[i][j] = 1
                        adj_rand[j][i] = 1
            
            # Remove K4s
            adj_bm_rand = [0] * n
            for i in range(n):
                for j in range(n):
                    if adj_rand[i][j]:
                        adj_bm_rand[i] |= (1 << j)
            
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                # Find a K4
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj_bm_rand[i] & (1 << j)):
                            continue
                        common = adj_bm_rand[i] & adj_bm_rand[j]
                        common &= ~(1 << i)
                        common &= ~(1 << j)
                        cm = common
                        while cm:
                            lsb = cm & (-cm)
                            k = lsb.bit_length() - 1
                            cm ^= lsb
                            rest = common & (cm | (1 << k))
                            rest &= adj_bm_rand[k]
                            if rest:
                                lsb2 = rest & (-rest)
                                l = lsb2.bit_length() - 1
                                # Found K4: i, j, k, l
                                # Remove one edge
                                u, v = i, j
                                adj_rand[u][v] = 0
                                adj_rand[v][u] = 0
                                adj_bm_rand[u] &= ~(1 << v)
                                adj_bm_rand[v] &= ~(1 << u)
                                changed = True
                                break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            
            if time.time() > deadline - 5:
                break
            
            score = max_triangle_free_size_exact(adj_bm_rand)
            if score < best_score:
                best_score = score
                best_edges = get_edges_from_adj(adj_rand)
    
    return best_edges
