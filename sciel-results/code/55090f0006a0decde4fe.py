"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Branch and bound to find the largest triangle-free induced subgraph.
    adj_bitmasks: list of ints, where adj_bitmasks[i] is a bitmask of neighbors of i.
    Returns the size of the largest triangle-free induced subgraph.
    """
    # Precompute for each vertex v, the set of vertices u such that v-u is an edge
    # and they share a common neighbor (i.e., adding both to a set that already
    # contains that common neighbor would create a triangle).
    # Actually, for a set S to be triangle-free: for every pair (u,v) in S with
    # edge (u,v), we need N(u) & N(v) & S == 0.
    
    # Key insight: if we build S incrementally, when we add vertex v to current set C,
    # a vertex u in the remaining pool is "conflicting" with v if:
    #   - (u,v) is an edge AND N(u) & N(v) & C != 0 (triangle u-v-w with w in C)
    #   - OR u is adjacent to two vertices in C that are adjacent to each other
    #     (triangle u-w1-w2)
    
    # For speed, precompute:
    # For each pair (i,j) that is an edge, common_neighbors[i][j] = adj[i] & adj[j]
    # For each vertex i, neighbors[i] = adj[i]
    
    # We'll use a recursive B&B with memoization on (available_mask, current_mask) 
    # but that's too large. Instead, we'll use a smarter ordering.
    
    # Precompute edge list as bitmasks for quick checking
    # For each vertex i, precompute which other vertices form triangles with i given a set C.
    
    # Simpler approach: iterate over all subsets but with early termination.
    # Since n=18, 2^18 = 262144. For each subset, checking if triangle-free
    # can be done quickly with bit operations.
    
    # Precompute: for each pair (i,j) with i<j and edge (i,j), 
    # tri_conflict[i][j] = adj[i] & adj[j]  (vertices that would complete a triangle)
    
    # A mask M is triangle-free iff for all i<j in M with edge (i,j):
    #   (adj[i] & adj[j] & M) == 0
    
    # Even faster: precompute for each vertex i, the "triangle constraint mask":
    # When i is in the set, any pair (j,k) in the set that are both adjacent to i
    # AND adjacent to each other would form a triangle.
    # So for vertex i, define conflict_i = { (j,k) : i~j, i~k, j~k }
    # Then M is triangle-free iff for all i in M: no edge (j,k) in M where both j,k in N(i)∩M.
    # Equivalently: for all i in M: the subgraph induced by N(i) & M has no edges.
    
    # Let's precompute: for each i, and for each subset of N(i), whether it's independent.
    # That's still complex. Let's just do the direct check but optimized.
    
    # Precompute adj as list of ints (already have adj_bitmasks)
    # For each pair (i,j) with i<j and edge, store the common neighbor mask
    cn = {}
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bitmasks[i] & (1 << j):
                cn[(i, j)] = adj_bitmasks[i] & adj_bitmasks[j]
    
    # Sort vertices by degree (descending) for better pruning
    deg = [adj_bitmasks[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Map: position in order -> vertex
    # We'll work in the ordered vertex space
    
    best = 0
    full_mask = (1 << n) - 1
    
    # For the B&B, we process vertices in 'order'.
    # State: (idx, available_mask, current_size, current_mask)
    # available_mask contains vertices at positions >= idx that haven't been decided yet.
    
    # Precompute for each vertex v, given current_mask, which vertices in available
    # cannot be added alongside v.
    
    # Let's use a different strategy: iterate over all masks but use a fast check.
    # For n=18, we can precompute a lookup table for small subsets.
    
    # Actually, let's just do a well-optimized full enumeration.
    # The key optimization: for a mask M, check triangle-free by iterating over
    # edges in M and checking if any edge has a common neighbor in M.
    
    # Precompute: for each vertex i, list of neighbors j > i (as bitmask)
    # And for each such edge (i,j), the common neighbor mask.
    
    edge_info = []  # list of (i, j, common_mask) for all edges i<j
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bitmasks[i] & (1 << j):
                common = adj_bitmasks[i] & adj_bitmasks[j]
                edge_info.append((i, j, common))
    
    # For a given mask, check if triangle-free:
    # For each (i, j, common) in edge_info where i and j are both in mask:
    #   if common & mask != 0: not triangle-free
    
    # Optimize: for each vertex i, precompute the set of edges (i,j) as a bitmask of j's.
    # Then for a mask, the edges in the induced subgraph are those (i,j) where
    # i in mask, j in mask, and j in adj[i].
    
    # Let's precompute for each i: adj_gt[i] = bitmask of neighbors j > i
    adj_gt = [0] * n
    for i in range(n):
        mask = 0
        for j in range(i + 1, n):
            if adj_bitmasks[i] & (1 << j):
                mask |= (1 << j)
        adj_gt[i] = mask
    
    # For each i and each neighbor j > i, store common neighbor mask in a dict
    # indexed by (i,j)
    cn_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bitmasks[i] & (1 << j):
                cn_dict[(i, j)] = adj_bitmasks[i] & adj_bitmasks[j]
    
    def is_triangle_free(mask):
        """Check if the induced subgraph on 'mask' is triangle-free."""
        # Iterate over vertices i in mask
        m = mask
        while m:
            i_bit = m & -m
            i = i_bit.bit_length() - 1
            m -= i_bit
            # Get neighbors of i that are in mask and > i
            nbrs = adj_gt[i] & mask
            while nbrs:
                j_bit = nbrs & -nbrs
                j = j_bit.bit_length() - 1
                nbrs -= j_bit
                # Check if (i,j) has a common neighbor in mask
                if cn_dict[(i, j)] & mask:
                    return False
        return True
    
    # Now enumerate all masks, but skip those with size <= best
    # Use a Gray code or just iterate. For n=18, 2^18 iterations.
    # Each check is O(E_in_mask) which is small on average.
    
    for mask in range(1, 1 << n):
        pc = mask.bit_count()
        if pc <= best:
            continue
        if is_triangle_free(mask):
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
    """Local search on bitmask representation with systematic 1-flip moves."""
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    # Precompute all possible edge flips
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Systematically try all single-edge flips
        improved = False
        # Shuffle order for diversity
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        
        for i, j in pairs:
            if deadline and time.time() > deadline:
                break
            
            if not (bm[i] & (1 << j)):
                # Try adding edge (i,j)
                # Check if it creates a K4
                common_ij = bm[i] & bm[j]
                creates_k4 = False
                temp = common_ij
                while temp:
                    k_bit = temp & -temp
                    k = k_bit.bit_length() - 1
                    if bm[i] & bm[j] & bm[k]:
                        creates_k4 = True
                        break
                    temp -= k_bit
                
                if not creates_k4:
                    bm[i] |= (1 << j)
                    bm[j] |= (1 << i)
                    new_score = _max_tf_size_fast(bm, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_bm = bm[:]
                        improved = True
                    else:
                        bm[i] ^= (1 << j)
                        bm[j] ^= (1 << i)
            else:
                # Try removing edge (i,j)
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                    improved = True
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        
        if not improved:
            # No improvement in this pass, try random restart within local neighborhood
            # Make a few random changes and continue
            for _ in range(5):
                if deadline and time.time() > deadline:
                    break
                i, j = rng.sample(range(n), 2)
                i, j = min(i, j), max(i, j)
                if not (bm[i] & (1 << j)):
                    common_ij = bm[i] & bm[j]
                    creates_k4 = False
                    temp = common_ij
                    while temp:
                        k_bit = temp & -temp
                        k = k_bit.bit_length() - 1
                        if bm[i] & bm[j] & bm[k]:
                            creates_k4 = True
                            break
                        temp -= k_bit
                    if not creates_k4:
                        bm[i] |= (1 << j)
                        bm[j] |= (1 << i)
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
    deadline = start_time + 115  # Safety margin before 120s limit
    
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
    
    # 2. Start from Turan and do local search (remove edges to break large TF sets)
    if time.time() < deadline - 10:
        ls_edges, ls_score = _local_search_bitmask(baseline_bm[:], n, max_iter=30, seed=42, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # 3. Random restarts with local search
    for seed in range(50):
        if time.time() > deadline:
            break
        
        bm, edges = _random_k4_free_graph(n, seed)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search with more iterations if we have time
        remaining = deadline - time.time()
        if remaining > 5:
            max_iters = min(20, int(remaining / 2))
            ls_edges, ls_score = _local_search_bitmask(bm, n, max_iter=max_iters, seed=seed, deadline=deadline)
            if ls_score < best_score:
                best_score = ls_score
                best_edges = ls_edges
    
    # 4. Final polish: if we have time, do more local search on the best solution
    if time.time() < deadline - 5:
        best_bm = [0] * n
        for e in best_edges:
            u, v = e
            best_bm[u] |= (1 << v)
            best_bm[v] |= (1 << u)
        final_edges, final_score = _local_search_bitmask(best_bm, n, max_iter=30, seed=999, deadline=deadline)
        if final_score < best_score:
            best_score = final_score
            best_edges = final_edges
    
    return best_edges
# EVOLVE-BLOCK-END
