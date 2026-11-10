import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limit

    n = 18
    best_score = n + 1
    best_adj = None

    # Precompute all size-12 subsets as bitmasks
    subsets_12 = []
    for combo in itertools.combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)

    def is_triangle_free(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                # If u and v are connected, and they share a common neighbor w in mask, 
                # then u-v-w is a triangle.
                if (adj[u] & adj[v] & mask):
                    return False
        return True

    def max_triangle_free_size(adj):
        """Calculate the size of the largest triangle-free induced subgraph."""
        # Check sizes from n down to 1
        for k in range(n, 0, -1):
            if k == 12:
                # Use precomputed masks
                for mask in subsets_12:
                    if is_triangle_free(mask, adj):
                        return k
            elif k > 12:
                # For k > 12, check all combinations (few of them)
                for comb in itertools.combinations(range(n), k):
                    mask = 0
                    for u in comb:
                        mask |= (1 << u)
                    if is_triangle_free(mask, adj):
                        return k
            else:
                # For k < 12, we can just return 12 if we get here? 
                # No, we need the exact max. But if no size 12, max is < 12.
                # We can check k=11, 10...
                # However, if we found no size 12, we should check 11.
                # But usually we stop at the first found max.
                # Since we iterate down from n, if we reach k < 12, it means no larger set is TF.
                # We can just return 1? Or check?
                # For efficiency, if k < 12, we can assume it's likely small, but let's be exact.
                # Actually, if no size 12 is TF, the max is <= 11.
                # We can just return 11? No, it could be 5.
                # Let's just check.
                for comb in itertools.combinations(range(n), k):
                    mask = 0
                    for u in comb:
                        mask |= (1 << u)
                    if is_triangle_free(mask, adj):
                        return k
        return 0

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # For any common neighbor w, if w is connected to any other common neighbor x, 
        # then u-v-w-x forms a K4.
        # Actually, if u and v are connected, and w is a common neighbor, 
        # we need to ensure no other common neighbor x is connected to w.
        # Wait, K4 is {u, v, w, x}. 
        # Edges: (u,v), (u,w), (u,x), (v,w), (v,x), (w,x).
        # So if we add (u,v), and w, x are common neighbors, and (w,x) exists, 
        # then we have a K4.
        # So we need to check if the subgraph induced by common neighbors is edge-free.
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            # Check if w has any neighbors in the remaining common neighbors
            # (adj[w] & common) will include w itself? No, adj[w] doesn't have self-loop.
            # But it will include other common neighbors.
            if (adj[w] & common):
                return False
        return True

    def generate_random_max_k4_free():
        """Generate a maximal K4-free graph by randomly adding edges."""
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Seed: 3 disjoint K6's (Complement of K6,6,6)
    # This graph is K4-free (max clique size 6, but wait, K6 contains K4).
    # Wait, a K6 is NOT K4-free. A K6 contains many K4s.
    # The seed in the current program returns edges of 3 disjoint K6s?
    # Let's check the current program's fallback:
    # parts = [range(0, 6), range(6, 12), range(12, 18)]
    # return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    # This creates edges BETWEEN the parts. So it's K6,6,6 (complete tripartite).
    # K6,6,6 is K4-free? 
    # A clique in a complete multipartite graph can have at most one vertex from each part.
    # So max clique size is 3. Thus it is K4-free.
    # The triangle-free induced subgraph size:
    # In K6,6,6, any two vertices from different parts are connected.
    # A triangle-free induced subgraph can have at most one vertex from any pair of parts?
    # No, if we take vertices from part A and part B, they are all connected.
    # If we take 2 from A and 1 from B, we have a triangle (A1-B1-A2-B1-A1).
    # So a triangle-free induced subgraph in a complete multipartite graph with parts P1, P2, P3
    # can contain vertices from at most 2 parts?
    # If it contains vertices from 3 parts, say a in A, b in B, c in C. 
    # (a,b), (b,c), (c,a) are all edges. So it's a triangle.
    # So any triangle-free induced subgraph can use vertices from at most 2 parts.
    # To maximize size, we take all vertices from 2 parts.
    # Size = 6 + 6 = 12.
    # So the seed score is 12.

    seed_adj = [0] * n
    # Create K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    seed_adj[a] |= (1 << b)
                    seed_adj[b] |= (1 << a)
    
    best_adj = seed_adj[:]
    best_score = 12 # We know this score is 12

    def local_search(adj):
        """Try to eliminate all triangle-free 12-subsets by adding edges."""
        adj = list(adj)  # copy
        max_iter = 200
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Check if there are any triangle-free 12-subsets
            tf_masks = []
            for mask in subsets_12:
                if is_triangle_free(mask, adj):
                    tf_masks.append(mask)
                    if len(tf_masks) > 20: # Early exit if too many
                        break
            
            if not tf_masks:
                return adj
            
            # Pick a random triangle-free 12-subset
            mask = random.choice(tf_masks)
            
            # Find non-edges within this subset
            non_edges = []
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                for v in range(u+1, n):
                    if (mask & (1 << v)) and not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
            
            if not non_edges:
                break
            
            # Try to add a random non-edge that doesn't create K4
            random.shuffle(non_edges)
            added = False
            for u, v in non_edges:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if not added:
                # Cannot add any edge from this subset without creating K4
                # Try a different subset or break
                break
        
        return adj

    # Phase 1: Local search from seed
    adj_ls = local_search(seed_adj)
    score = max_triangle_free_size(adj_ls)
    if score < best_score:
        best_score = score
        best_adj = adj_ls[:]

    # Phase 2: Random restarts
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        random.seed(attempts * 7919) # Deterministic-ish restarts
        
        adj = generate_random_max_k4_free()
        
        # Quick check: does it have any triangle-free 12-subset?
        tf_count = 0
        for mask in subsets_12:
            if is_triangle_free(mask, adj):
                tf_count += 1
                if tf_count > 50:  # Too many, skip
                    break
        
        if tf_count == 0:
            # Score is < 12
            score = max_triangle_free_size(adj)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
                if best_score <= 8:
                    break # Very good, stop?
        
        if tf_count <= 50:
            # Promising: run local search
            adj_ls = local_search(adj)
            score = max_triangle_free_size(adj_ls)
            if score < best_score:
                best_score = score
                best_adj = adj_ls[:]
                if best_score <= 8:
                    break

    # Phase 3: If time remains, try more local searches from random graphs
    while time.time() < deadline - 2:
        random.seed(int(time.time() * 1000) % 1000000)
        adj = generate_random_max_k4_free()
        adj_ls = local_search(adj)
        score = max_triangle_free_size(adj_ls)
        if score < best_score:
            best_score = score
            best_adj = adj_ls[:]

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
