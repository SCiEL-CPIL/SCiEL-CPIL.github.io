"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    adj_bitmasks: list of ints, where adj_bitmasks[i] is a bitmask of neighbors of i.
    Returns the size of the largest triangle-free induced subgraph.
    """
    # Precompute: for each vertex v, the bitmask of vertices that form a triangle with v?
    # No, simpler: A set S is triangle-free if for every edge (u,v) in S, 
    # (N(u) & N(v) & S) is empty.
    
    # We use a recursive branch and bound on the set of available vertices.
    # To make it fast, we order vertices and prune aggressively.
    
    # Let's use a simple heuristic search with bitmasks.
    # Since n=18 is small, we can try to find the independence number of the "triangle hypergraph"
    # or just do a smart DFS.
    
    # Optimization: Precompute common neighbors for all pairs to check triangles in O(1).
    # tri_mask[v] = bitmask of vertices u such that (v, u) is an edge and they have a common neighbor?
    # Actually, for a set S, it is triangle-free iff for all u,v in S, (N(u) & N(v) & S) == 0.
    
    # Let's implement a branch and bound that tries to build a maximal triangle-free set.
    # We want to maximize the size.
    
    best_size = 0
    full_mask = (1 << n) - 1
    
    def dfs(available, current_size, current_mask):
        nonlocal best_size
        if current_size + available.bit_count() <= best_size:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Heuristic: pick a vertex from available
        # To prune better, we should pick a vertex that is "constrained" or just iterate.
        # For speed, let's just pick the lowest set bit.
        v_bit = available & -available
        v = v_bit.bit_length() - 1
        
        # Option 1: Include v
        # If we include v, we must remove from 'available' any vertex u such that
        # adding u to (current_mask | v_bit) would create a triangle.
        # A triangle is created if there is an edge (v, u) and a common neighbor w in current_mask.
        # Or if there is an edge (v, u) and a common neighbor w in (current_mask | v_bit) ?? No, w must be in the set.
        # Condition for S = current_mask | v_bit | u_bit to be triangle-free:
        # 1. (v, u) is not an edge? If it is, then N(v) & N(u) & S must be 0.
        #    Since v and u are in S, we check if N(v) & N(u) & (current_mask) is non-empty.
        #    If yes, we cannot add u.
        # 2. Also, u must not form a triangle with any pair in current_mask.
        #    i.e., for all w in current_mask, if (u,w) is edge, then N(u) & N(w) & current_mask must be 0?
        #    Actually, if (u,w) is edge and they have a common neighbor z in current_mask, triangle (u,w,z).
        
        # Let's precompute "bad" vertices for v given current_mask.
        # If we add v, a vertex u in available is "bad" if:
        # a) u is adjacent to v AND (N(u) & N(v) & current_mask) != 0.
        # b) u is NOT adjacent to v, but u forms a triangle with some pair in current_mask?
        #    Wait, if current_mask is triangle-free, adding u is safe w.r.t current_mask pairs
        #    UNLESS u is adjacent to two vertices in current_mask that are adjacent to each other?
        #    No, if current_mask is triangle-free, it contains no edges between non-adjacent vertices?
        #    Triangle-free means no 3-cycle.
        #    If u is added, we check edges from u to vertices in current_mask.
        #    Let N_u_in_cur = N(u) & current_mask.
        #    If there are two vertices w1, w2 in N_u_in_cur such that w1-w2 is an edge, then (u, w1, w2) is a triangle.
        #    So u is bad if the subgraph induced by N(u) & current_mask contains an edge.
        
        # This check is expensive if done naively.
        # Given n=18, let's just try a simpler approach:
        # Iterate subsets in decreasing size? No, 2^18 is okay if checks are fast.
        
        # Let's stick to the bitmask iteration but optimized.
        pass

    # Actually, the previous `_max_tf_size_bitmask` was O(2^n * n^2).
    # 2^18 * 324 ~ 8.5 * 10^7 operations. In Python, this might take 10-20 seconds.
    # We need it to be faster or called fewer times.
    
    # Let's use a different strategy:
    # 1. Generate a good graph.
    # 2. Use a fast C-like loop in Python? No.
    # 3. Use NumPy?
    
    # Let's try to optimize the check for a specific mask.
    # Precompute for each pair (i,j) the bitmask of common neighbors.
    # Then a mask is triangle-free iff for all i<j in mask, if adj[i][j], then (common[i][j] & mask) == 0.
    
    # We will compute the score by iterating all masks, but we will do it in a tight loop.
    # To speed up, we can use `int.bit_count()` which is fast in Py3.8+.
    
    # Precompute common neighbors bitmask for all pairs
    cn = [0] * (n * n)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bitmasks[i] & (1 << j):
                # i and j are adjacent
                # Common neighbors are in adj_bitmasks[i] & adj_bitmasks[j]
                # Exclude i and j themselves (though they are not neighbors of themselves usually)
                cm = adj_bitmasks[i] & adj_bitmasks[j]
                cn[i * n + j] = cm
                cn[j * n + i] = cm

    best = 0
    # Iterate all masks
    # To speed up, we can skip masks that are subsets of known triangle-free masks? No.
    # Just iterate.
    for mask in range(1 << n):
        pc = mask.bit_count()
        if pc <= best:
            continue
        
        # Check if triangle-free
        # Iterate over edges in the induced subgraph
        # We can iterate over all pairs i<j in mask
        is_tf = True
        # Optimization: only check pairs that are edges
        # We can iterate i from 0 to n-1
        # if i in mask:
        #   for j in i+1 to n-1:
        #     if j in mask and adj[i][j]:
        #       if cn[i*n+j] & mask:
        #         is_tf = False
        #         break
        # This is still O(n^2) per mask.
        
        # Can we do better?
        # If we precompute for each vertex i, the list of neighbors j>i.
        # Then for a mask, we iterate i in mask, and for each neighbor j in mask, check cn.
        
        if is_tf:
            for i in range(n):
                if not (mask >> i) & 1:
                    continue
                # Check neighbors j > i
                # We can use bit operations to get neighbors of i that are in mask and > i
                # neighbors_gt_i = adj_bitmasks[i] & mask & ~((1 << (i+1)) - 1)
                # But we need to check cn for each pair.
                
                # Let's just loop j
                for j in range(i + 1, n):
                    if (mask >> j) & 1:
                        if adj_bitmasks[i] & (1 << j):
                            if cn[i * n + j] & mask:
                                is_tf = False
                                break
                if not is_tf:
                    break
            
        if is_tf:
            best = pc
            
    return best

def _adj_to_bitmasks(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    bm = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        bm[i] = mask
    return bm

def _check_k4_free_bitmask(bm, n):
    """Check if graph represented by bitmasks is K4-free."""
    # A K4 exists if there is a triangle (i,j,k) and a vertex l adjacent to all three.
    # For each triangle, check if intersection of their neighbor sets is non-empty.
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                # i-j edge
                # Find common neighbors k > j
                common = bm[i] & bm[j]
                # Iterate k in common
                while common:
                    k_bit = common & -common
                    k = k_bit.bit_length() - 1
                    # Check if there is l > k adjacent to i, j, k
                    # Intersection of bm[i], bm[j], bm[k]
                    # If non-empty, K4 found
                    if bm[i] & bm[j] & bm[k]:
                        return False
                    common -= k_bit
    return True

def _random_k4_free_graph(n, seed):
    """Generate a random K4-free graph using bitmask representation."""
    rng = random.Random(seed)
    bm = [0] * n
    
    # Try to add edges in random order
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for i, j in edges_to_try:
        if rng.random() < 0.5:
            # Check if adding (i,j) creates a K4
            # This happens if there is a vertex k such that (i,k), (j,k) exist,
            # and there is a vertex l such that (i,l), (j,l), (k,l) exist.
            # Actually, simpler:
            # If we add (i,j), a K4 is formed if there is a triangle (i,j,k) in the NEW graph
            # that has a common neighbor l.
            # Or, if there is a triangle (i,k,l) and (j,k,l) ... no.
            
            # Condition for K4 upon adding (i,j):
            # There exists k such that (i,k) and (j,k) are edges (in current graph).
            # AND there exists l such that (i,l), (j,l), (k,l) are edges.
            # Note: l can be k? No, simple graph.
            # So, for each k in N(i) ∩ N(j):
            #   Check if N(i) ∩ N(j) ∩ N(k) is non-empty.
            
            common_ij = bm[i] & bm[j]
            creates_k4 = False
            temp_common = common_ij
            while temp_common:
                k_bit = temp_common & -temp_common
                k = k_bit.bit_length() - 1
                # Check if N(i) & N(j) & N(k) is non-empty
                if bm[i] & bm[j] & bm[k]:
                    creates_k4 = True
                    break
                temp_common -= k_bit
            
            if not creates_k4:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
    
    # Convert back to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                edges.append([i, j])
    return bm, edges

def _local_search_bitmask(bm, n, max_iter=500, seed=0, deadline=None):
    """Local search on bitmask representation."""
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    # Precompute list of all edges and non-edges for fast sampling?
    # Just sample random pairs.
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        if not (bm[i] & (1 << j)):
            # Try adding edge
            # Check K4
            common_ij = bm[i] & bm[j]
            creates_k4 = False
            temp_common = common_ij
            while temp_common:
                k_bit = temp_common & -temp_common
                k = k_bit.bit_length() - 1
                if bm[i] & bm[j] & bm[k]:
                    creates_k4 = True
                    break
                temp_common -= k_bit
            
            if not creates_k4:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        else:
            # Try removing edge
            bm[i] ^= (1 << j)
            bm[j] ^= (1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
    # Convert best_bm to edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    
    # 1. Baseline: Turan graph T(18, 3) which is K4-free.
    # Parts of size 6.
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    baseline_bm = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    baseline_bm[a] |= (1 << b)
                    baseline_bm[b] |= (1 << a)
    
    baseline_score = _max_tf_size_fast(baseline_bm, n)
    best_score = baseline_score
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if baseline_bm[i] & (1 << j):
                best_edges.append([i, j])
    
    # 2. Random restarts with local search
    for seed in range(100):
        if time.time() > deadline:
            break
        
        bm, edges = _random_k4_free_graph(n, seed)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search
        if time.time() > deadline - 10:
            break
            
        ls_edges, ls_score = _local_search_bitmask(bm, n, max_iter=100, seed=seed, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    return best_edges
# EVOLVE-BLOCK-END
