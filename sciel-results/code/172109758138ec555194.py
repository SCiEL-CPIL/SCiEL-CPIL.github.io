import itertools
import numpy as np
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate all weight-4 supports
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # For each support, precompute which other supports share exactly 3 coordinates
    # and the relative sign positions
    # We'll build the conflict graph incrementally
    
    # Total vertices: n_supports * 16
    n_vertices = n_supports * 16
    
    # For each vertex (support_idx, sign_idx), precompute its neighbors in the conflict graph
    # Two vertices conflict if:
    # 1. Same support, different signs -> they're at distance 2*sqrt(sum of squared diffs)
    #    Actually same support with different signs: the vectors differ in some coordinates
    #    v_i has signs s_i, v_j has signs s_j on the same support
    #    distance^2 = sum over 4 positions of (s_i[k] - s_j[k])^2
    #    Each term is 0 or 4, so distance^2 = 4 * hamming_distance(s_i, s_j)
    #    For conflict: distance < max_norm = 2
    #    distance < 2 means distance^2 < 4, i.e., hamming_distance = 0 (same vector)
    #    So same support, different signs: always compatible (distance >= 2*sqrt(4)=4 > 2)
    #    Wait, max_norm = sqrt(4) = 2. So we need distance >= 2.
    #    Same support, different signs: distance^2 = 4*hamming, min hamming=1, so distance^2 >= 4, distance >= 2. OK.
    #    Same support, same sign: distance = 0, but that's the same vector. We only pick one per support-sign.
    
    # 2. Different supports sharing exactly 3 coordinates:
    #    v_i on support Si with signs s_i, v_j on support Sj with signs s_j
    #    They share 3 coordinates. On those 3, the values must differ for distance >= 2.
    #    distance^2 = sum over all 8 non-zero positions
    #    On the 3 shared: (s_i[pos] - s_j[pos])^2, each is 0 or 4
    #    On the 1 unique to i: s_i^2 = 1
    #    On the 1 unique to j: s_j^2 = 1
    #    So distance^2 = 4 * (number of shared positions where signs differ) + 2
    #    For conflict: distance^2 < 4, i.e., 4*(diffs) + 2 < 4, i.e., diffs = 0
    #    So conflict when all 3 shared positions have the SAME sign.
    
    # 3. Different supports sharing 0, 1, or 2 coordinates:
    #    distance^2 = 4 (from i's unique) + 4 (from j's unique) + ... 
    #    If share 0: distance^2 = 4+4 = 8, distance = 2*sqrt(2) > 2. OK.
    #    If share 1: distance^2 = 4+4+4*(0 or 1) = 8 or 12. Min is 8 > 4. OK.
    #    If share 2: distance^2 = 4+4+4*(0 or 1 or 2) = 8, 12, or 16. Min is 8 > 4. OK.
    #    So only conflicts are: same support (same sign), or different supports sharing 3 with same signs on all 3.
    
    # Build adjacency list
    # For efficiency, precompute for each support pair (i, j) with |Si ∩ Sj| = 3:
    #   the intersection and relative positions
    
    # Precompute support intersections
    # support_inter[i][j] = (inter_tuple, pos_in_i, pos_in_j) if |inter| == 3, else None
    # To save memory, use a dict keyed by (i, j) for i < j
    
    support_pairs = {}  # (i, j) -> (inter, pos_i, pos_j)
    for i in range(n_supports):
        Si = supports[i]
        for j in range(i + 1, n_supports):
            Sj = supports[j]
            inter = tuple(set(Si) & set(Sj))
            if len(inter) == 3:
                pos_i = [Si.index(k) for k in inter]
                pos_j = [Sj.index(k) for k in inter]
                support_pairs[(i, j)] = (inter, pos_i, pos_j)
    
    # Build adjacency list for all n_vertices vertices
    # Vertex id = support_idx * 16 + sign_idx
    adj = [[] for _ in range(n_vertices)]
    
    # For each support pair with intersection size 3
    for (i, j), (inter, pos_i, pos_j) in support_pairs.items():
        for si in range(16):
            for sj in range(16):
                # Check if signs are same on all 3 intersection positions
                # sign_options[si][pos_i[k]] == sign_options[sj][pos_j[k]] for all k
                # Precompute sign_options
                pass
    
    # Precompute sign_options
    sign_options = list(itertools.product([-1, 1], repeat=4))
    
    # Rebuild adjacency with proper sign checking
    for (i, j), (inter, pos_i, pos_j) in support_pairs.items():
        for si in range(16):
            s_i = sign_options[si]
            for sj in range(16):
                s_j = sign_options[sj]
                # Check if signs match on all 3 intersection positions
                conflict = True
                for k in range(3):
                    if s_i[pos_i[k]] != s_j[pos_j[k]]:
                        conflict = False
                        break
                if conflict:
                    vid_i = i * 16 + si
                    vid_j = j * 16 + sj
                    adj[vid_i].append(vid_j)
                    adj[vid_j].append(vid_i)
    
    # Also, same support same sign is the same vertex, so no self-loops needed.
    # Different signs on same support are always compatible.
    
    # Now solve Maximum Independent Set using Tomita's algorithm
    # Convert to 0-indexed and use bitsets for efficiency
    
    n_v = n_vertices
    
    # Use Python integers as bitsets
    # adj_bits[v] = bitmask of neighbors of v
    adj_bits = [0] * n_v
    for v in range(n_v):
        mask = 0
        for u in adj[v]:
            mask |= (1 << u)
        adj_bits[v] = mask
    
    # Tomita's algorithm for Maximum Independent Set
    # We want to find the largest set of vertices with no edges between them.
    # This is equivalent to Maximum Clique in the complement graph.
    # But let's do MIS directly.
    
    # Actually, let me use a branch-and-bound for MIS:
    # State: current candidate set P (bitmask), current solution S (bitmask)
    # Upper bound: |P| + |S|
    # If |P| + |S| <= best, prune.
    # Pick a vertex v from P, branch on include v (remove v and its neighbors from P) 
    # or exclude v (remove v from P).
    
    # For efficiency, use a greedy coloring bound on the complement graph (clique bound).
    # Actually, for MIS, the bound is just |P| since any independent set in P has size <= |P|.
    # That's weak. Let's use the clique bound on the complement.
    
    # Alternative: use a simpler approach. Since the graph is structured, let's try
    # a greedy approach first, then local search.
    
    # Actually, let me implement a proper branch-and-bound with coloring bound.
    # For MIS, we can use the fact that the complement graph's cliques are our independent sets.
    # So we solve Max Clique on the complement graph.
    
    # Complement adjacency: comp_adj_bits[v] = all bits except v and adj_bits[v]
    all_bits = (1 << n_v) - 1
    comp_adj_bits = [0] * n_v
    for v in range(n_v):
        comp_adj_bits[v] = all_bits & ~(adj_bits[v] | (1 << v))
    
    # Tomita's Max Clique algorithm
    best_clique_size = 0
    best_clique = []
    
    def tomita_max_clique(P, S, depth):
        """
        P: candidate set (bitmask)
        S: current clique (bitmask)
        depth: current depth
        """
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if depth > best_clique_size:
                best_clique_size = depth
                best_clique = list(S)
            return
        
        # Upper bound: |P| + depth
        if P.bit_count() + depth <= best_clique_size:
            return
        
        # Greedy coloring of P for upper bound
        # Color the vertices in P such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the clique size in P.
        # Order vertices by degree in P for better coloring.
        
        P_list = []
        tmp = P
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            P_list.append(v)
            tmp ^= lsb
        
        # Sort by degree in P (descending)
        P_list.sort(key=lambda v: (comp_adj_bits[v] & P).bit_count(), reverse=True)
        
        # Greedy coloring
        color_of = {}
        colors_used = 0
        for v in P_list:
            # Find the smallest color not used by neighbors
            neighbor_colors = set()
            for u in P_list:
                if u == v:
                    continue
                if (comp_adj_bits[v] >> u) & 1:
                    if u in color_of:
                        neighbor_colors.add(color_of[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            colors_used = max(colors_used, c + 1)
        
        # Upper bound: colors_used + depth
        if colors_used + depth <= best_clique_size:
            return
        
        # Branch on vertices in order of color (highest color first)
        # Process vertices grouped by color
        for c in range(colors_used, 0, -1):
            # Get vertices with color c
            P_c = 0
            for v in P_list:
                if color_of[v] == c:
                    P_c |= (1 << v)
            
            # Upper bound for this sub-branch
            if P_c.bit_count() + depth <= best_clique_size:
                continue
            
            # Pick a vertex from P_c to branch on
            # Choose the one with highest degree in P_c
            v = None
            max_deg = -1
            tmp = P_c
            while tmp:
                lsb = tmp & (-tmp)
                u = lsb.bit_length() - 1
                deg = (comp_adj_bits[u] & P_c).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
                tmp ^= lsb
            
            if v is None:
                continue
            
            # Branch 1: include v
            new_P = P_c & comp_adj_bits[v]
            new_S = S | (1 << v)
            tomita_max_clique(new_P, new_S, depth + 1)
            
            # Branch 2: exclude v
            P_c ^= (1 << v)
            # Continue with remaining vertices in P_c
            # This is handled by the loop structure, but we need to be careful
            # Actually, the standard Tomita processes one vertex at a time
            # Let me restructure
    
    # The above has issues with the branching structure. Let me use a cleaner implementation.
    
    def max_clique_tomita(P, S, depth):
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if depth > best_clique_size:
                best_clique_size = depth
                best_clique = list(S)
            return
        
        # Bound
        if P.bit_count() + depth <= best_clique_size:
            return
        
        # Greedy coloring
        P_list = []
        tmp = P
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            P_list.append(v)
            tmp ^= lsb
        
        P_list.sort(key=lambda v: (comp_adj_bits[v] & P).bit_count(), reverse=True)
        
        color_of = {}
        colors_used = 0
        for v in P_list:
            neighbor_colors = set()
            for u in P_list:
                if u != v and (comp_adj_bits[v] >> u) & 1:
                    if u in color_of:
                        neighbor_colors.add(color_of[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            colors_used = max(colors_used, c + 1)
        
        if colors_used + depth <= best_clique_size:
            return
        
        # Branch by color groups
        for c in range(colors_used, 0, -1):
            P_c = 0
            for v in P_list:
                if color_of[v] == c:
                    P_c |= (1 << v)
            
            if P_c.bit_count() + depth <= best_clique_size:
                continue
            
            # Process vertices in P_c one by one
            P_remaining = P_c
            while P_remaining:
                if time.time() > deadline:
                    return
                
                # Pick vertex with max degree in P_remaining
                v = None
                max_deg = -1
                tmp = P_remaining
                while tmp:
                    lsb = tmp & (-tmp)
                    u = lsb.bit_length() - 1
                    deg = (comp_adj_bits[u] & P_remaining).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        v = u
                    tmp ^= lsb
                
                if v is None:
                    break
                
                # Bound check
                if (P_remaining & ~(1 << v)).bit_count() + depth <= best_clique_size:
                    break
                
                # Include v
                new_P = (P_remaining & ~(1 << v)) & comp_adj_bits[v]
                max_clique_tomita(new_P, S | (1 << v), depth + 1)
                
                # Exclude v
                P_remaining ^= (1 << v)
    
    # Run the algorithm
    max_clique_tomita(all_bits, 0, 0)
    
    if best_clique_size > 0:
        # Convert clique to vectors
        vectors = []
        for vid in best_clique:
            support_idx = vid // 16
            sign_idx = vid % 16
            s = sign_options[sign_idx]
            v = [0] * D
            Si = supports[support_idx]
            for k in range(4):
                v[Si[k]] = s[k]
            vectors.append(v)
        
        vec_arr = np.array(vectors, dtype=np.int64)
        
        # Verify
        norms = np.linalg.norm(vec_arr, axis=1)
        max_norm = np.max(norms)
        dists = np.linalg.norm(vec_arr[:, np.newaxis] - vec_arr[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= max_norm - 1e-9:
            return vec_arr
    
    # Fallback: Seed solution (weight 2)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)
