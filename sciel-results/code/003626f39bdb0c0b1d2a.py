"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def _build_bitmasks(n, edges):
    """Build bitmask adjacency list from edge list."""
    masks = [0] * n
    for i, j in edges:
        masks[i] |= (1 << j)
        masks[j] |= (1 << i)
    return masks

def _has_k5(masks, n):
    """
    Check if graph has a clique of size 5.
    Returns True if a K5 is found.
    Strategy: For each vertex v, check if its neighborhood contains a K4.
    """
    for v in range(n):
        # Neighbors of v
        mask_v = masks[v]
        
        # We need to find 4 vertices a, b, c, d in mask_v such that they form a K4.
        # This is equivalent to finding a triangle (a,b,c) in mask_v such that 
        # (a,b,c) have a common neighbor d in mask_v.
        
        # Iterate over pairs (a, b) in mask_v
        # Get bits of mask_v
        # Optimization: Precompute list of neighbors? No, bit iteration is fine for n<=50.
        
        # Let's extract neighbors into a list for easier iteration over pairs
        neigh = []
        m = mask_v
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            neigh.append(idx)
            m ^= lsb
            
        k = len(neigh)
        if k < 4:
            continue
            
        # Check all triples (a, b, c) in neigh
        # If a, b, c form a triangle, check if they have a common neighbor in neigh
        for i in range(k):
            a = neigh[i]
            for j in range(i+1, k):
                b = neigh[j]
                # Check if a and b are connected
                if not (masks[a] & (1 << b)):
                    continue
                for l in range(j+1, k):
                    c = neigh[l]
                    # Check if a,c and b,c are connected
                    if not (masks[a] & (1 << c)) or not (masks[b] & (1 << c)):
                        continue
                    
                    # a, b, c form a triangle.
                    # Do they have a common neighbor d in neigh?
                    # Common neighbors of a, b, c in the whole graph:
                    common = masks[a] & masks[b] & masks[c]
                    # Intersect with mask_v (neighbors of v) to ensure d is in neigh
                    # Actually, if d is in common and d != v, and d is adjacent to v, then d is in K5.
                    # The K5 would be {v, a, b, c, d}.
                    # d must be adjacent to v, so d is in mask_v.
                    # So we check if (common & mask_v) has any bit set.
                    # Note: a,b,c are in mask_v, so they are in common? No, common is neighbors of a,b,c.
                    # a is neighbor of b,c but not a. So a is not in masks[a].
                    # So common contains vertices adjacent to a,b,c.
                    # We need d such that d is adjacent to a,b,c AND d is adjacent to v.
                    # So d is in (masks[a] & masks[b] & masks[c]) & mask_v.
                    
                    if common & mask_v:
                        return True
    return False

def _has_ind5(masks, n):
    """
    Check if graph has an independent set of size 5.
    Independent set in G is a clique in complement.
    """
    # Build complement masks
    full_mask = (1 << n) - 1
    comp_masks = [0] * n
    for i in range(n):
        # Complement of masks[i] excluding i itself
        comp_masks[i] = full_mask ^ masks[i] ^ (1 << i)
        
    return _has_k5(comp_masks, n)

def _is_valid(masks, n):
    """Check if graph is (5,5)-free."""
    if _has_k5(masks, n):
        return False
    if _has_ind5(masks, n):
        return False
    return True

def _try_add_vertex(masks, n, max_n, deadline, rng):
    """
    Try to add a new vertex to the graph.
    Returns (new_masks, new_n) if successful, else (None, n).
    """
    # We try to add vertex n (0-indexed, so new index is n)
    # We need to choose a subset S of {0, ..., n-1} to be neighbors of n.
    # Constraints:
    # 1. S must not contain a K4 (else {n} U K4 is K5)
    # 2. {0, ..., n-1} \ S must not contain a K4 (else Ind5)
    
    # Heuristic: Random sampling with bias towards "balanced" sets
    # Or: Start with empty set, add vertices one by one if valid?
    # Let's try random subsets first, as it's simpler and often works for small n.
    
    # To make it smarter, we can estimate the "danger" of adding a vertex u to S.
    # If we add u to S, we increase the risk of K4 in S.
    # If we don't add u to S, we increase the risk of K4 in complement.
    
    # Let's try a few random densities
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Random density between 0.3 and 0.7
        p = rng.uniform(0.3, 0.7)
        new_mask = 0
        for i in range(n):
            if rng.random() < p:
                new_mask |= (1 << i)
                
        # Check validity of adding vertex n with neighbors defined by new_mask
        # New graph has n+1 vertices.
        # New masks:
        # old masks remain same (plus bit for n if connected)
        # new mask for n is new_mask
        
        # We can check validity without building full new_masks array if we just check constraints.
        # Constraint 1: No K5 in new graph.
        # K5 can only involve the new vertex n (since old graph was valid).
        # So we just need to check if N(n) contains a K4.
        # N(n) is the set of bits in new_mask.
        # We need to check if the subgraph induced by new_mask contains a K4.
        
        # Constraint 2: No Ind5 in new graph.
        # Ind5 can only involve n.
        # So we need to check if Non-N(n) contains a K4 in the complement?
        # No, Ind5 in G is K5 in Comp(G).
        # New vertex n in Comp(G) has neighbors = Non-N(n) in G.
        # So we need to check if Non-N(n) contains a K4 in the original graph's complement?
        # Wait.
        # Ind5 in G means 5 vertices no edges between them.
        # If n is in the Ind5, then the other 4 vertices must be in Non-N(n) and form an Ind4 in G.
        # Ind4 in G is K4 in Comp(G).
        # So we need to check if the subgraph of Non-N(n) in Comp(G) contains a K4.
        # This is equivalent to checking if Non-N(n) contains an Ind4 in G.
        
        # Let's write a helper to check if a subset of vertices contains a K4 (or Ind4).
        # For efficiency, we can check K4 in the induced subgraph.
        
        def _subset_has_k4(subset_mask, masks, n):
            # Check if induced subgraph on subset_mask has a K4
            # Extract vertices
            verts = []
            m = subset_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            k = len(verts)
            if k < 4:
                return False
            
            # Check all 4-combinations? C(k,4). k can be up to n.
            # If n=37, C(37,4) is ~ 65k. This is fast.
            # But we do this 100 times per add attempt. 6.5M ops. OK.
            
            # Optimization: Backtracking for K4
            # For each v in verts:
            #   If N(v) within verts has a K3
            for v in verts:
                # Neighbors of v within subset
                # mask_v = masks[v] & subset_mask
                # We need K3 in mask_v
                # Iterate pairs (a,b) in mask_v
                # Check if a,b connected and have common neighbor c in mask_v
                
                mask_v = masks[v] & subset_mask
                
                # Get list of neighbors in subset
                neigh = []
                mm = mask_v
                while mm:
                    lsb = mm & (-mm)
                    idx = lsb.bit_length() - 1
                    neigh.append(idx)
                    mm ^= lsb
                    
                # Check for triangle in neigh
                # If triangle found, return True (K4 = v + triangle)
                # Wait, v + triangle is K4.
                # So we just need to check if neigh contains a triangle.
                
                # Check all pairs in neigh
                len_neigh = len(neigh)
                for i in range(len_neigh):
                    a = neigh[i]
                    for j in range(i+1, len_neigh):
                        b = neigh[j]
                        if not (masks[a] & (1 << b)):
                            continue
                        # a, b connected. Do they have common neighbor c in neigh?
                        common = masks[a] & masks[b] & mask_v
                        # common contains c such that c is neighbor of a, b, and v.
                        # We need c to be in neigh (which it is, since we intersected with mask_v).
                        # But we also need c != a, c != b.
                        # Since graph is simple, c is not a or b.
                        # So if common is not 0, we have a triangle a,b,c.
                        if common:
                            return True
            return False

        def _subset_has_ind4(subset_mask, masks, n):
            # Check if induced subgraph on subset_mask has an Ind4
            # This is K4 in complement
            # Build local complement masks? Or just check pairs.
            # Ind4 means 4 vertices with no edges between them.
            # Equivalent to: In the subgraph, no edges.
            # So for any pair (a,b) in subset, they must NOT be connected.
            # We can just check if the subgraph has ANY edge. If it has no edges, it's an Ind4.
            # Wait, Ind4 means the induced subgraph is empty (no edges).
            # So we just need to check if there are no edges in the induced subgraph.
            
            verts = []
            m = subset_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            k = len(verts)
            if k < 4:
                return False
            
            # Check if any edge exists in induced subgraph
            for i in range(k):
                a = verts[i]
                # Check edges from a to b where b in verts and b > a
                # We can just check if (masks[a] & subset_mask) has any bits set for b > a?
                # Simpler: check all pairs.
                # If any pair is connected, it's not an Ind4.
                # We want to know if it IS an Ind4.
                # So if we find ANY edge, return False.
                # If we finish and find NO edges, return True.
                
                # Optimization:
                # mask_a = masks[a] & subset_mask
                # Remove a itself? masks[a] doesn't have a.
                # We just need to check if mask_a has any bit corresponding to a vertex in verts.
                # Since we are iterating all pairs, we can just do:
                # For each a in verts:
                #   if (masks[a] & subset_mask) != 0: 
                #       # Wait, this checks if a is connected to ANY vertex in subset.
                #       # If a is connected to b in subset, then edge exists.
                #       # So if we find any a with degree > 0 in induced subgraph, return False.
                #       # But we need to be careful: (masks[a] & subset_mask) includes neighbors of a in subset.
                #       # If this is non-zero, there is an edge incident to a.
                #       # So if ANY vertex has degree > 0, it's not an Ind4.
                #       # So if ALL vertices have degree 0, it is an Ind4.
                
                if (masks[a] & subset_mask) != 0:
                    return False
            return True

        # Check Constraint 1: N(n) (new_mask) must not contain K4
        if _subset_has_k4(new_mask, masks, n):
            continue
            
        # Check Constraint 2: Non-N(n) must not contain Ind4
        # Non-N(n) is (full_mask_n ^ new_mask)
        full_mask_n = (1 << n) - 1
        non_new_mask = full_mask_n ^ new_mask
        if _subset_has_ind4(non_new_mask, masks, n):
            continue
            
        # If we get here, adding vertex n with neighbors new_mask is valid.
        # Construct new masks
        new_masks = [0] * (n + 1)
        for i in range(n):
            # Copy old mask
            new_masks[i] = masks[i]
            # If i is connected to n, set bit n
            if new_mask & (1 << i):
                new_masks[i] |= (1 << n)
        
        new_masks[n] = new_mask
        return new_masks, n + 1

    return None, n

def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_37_edges()
    masks = _build_bitmasks(n, edges)
    
    # Verify initial validity (should be true)
    # if not _is_valid(masks, n):
    #    return n, edges
    
    best_n = n
    best_masks = masks[:]
    
    rng = random.Random(42)
    
    # Try to expand
    current_n = n
    current_masks = masks[:]
    
    # We will loop, trying to add vertices.
    # If we fail to add a vertex, we might try to "perturb" the graph?
    # For now, simple greedy expansion.
    
    while current_n < 46:
        if time.time() > deadline:
            break
            
        new_masks, new_n = _try_add_vertex(current_masks, current_n, 46, deadline, rng)
        
        if new_n > current_n:
            current_n = new_n
            current_masks = new_masks
            best_n = current_n
            best_masks = current_masks[:]
        else:
            # Failed to add a vertex. 
            # We can try a few more times with different random seeds?
            # Or break if we've tried enough?
            # Let's try a few more times to see if we can find a valid configuration.
            # If we fail 50 times, we assume we're stuck.
            stuck_count = 0
            while stuck_count < 50:
                if time.time() > deadline:
                    break
                new_masks, new_n = _try_add_vertex(current_masks, current_n, 46, deadline, rng)
                if new_n > current_n:
                    current_n = new_n
                    current_masks = new_masks
                    best_n = current_n
                    best_masks = current_masks[:]
                    break
                stuck_count += 1
            else:
                # Stuck, break outer loop
                break

    # Convert best_masks to edge list
    final_edges = []
    for i in range(best_n):
        m = best_masks[i]
        # Only add edges j > i
        for j in range(i + 1, best_n):
            if m & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges

# EVOLVE-BLOCK-END
