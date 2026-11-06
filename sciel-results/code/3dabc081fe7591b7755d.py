import time
import random
import numpy as np
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def generate_conference_graph(n):
    """Generate a graph from a conference matrix (n-1 must be even, i.e., n odd).
    Conference matrix C satisfies C*C^T = (n-1)*I, C symmetric, zero diagonal, +/-1 off-diagonal.
    Graph: edge (i,j) iff C[i][j] == 1.
    This gives a regular graph with specific spectral properties.
    """
    if n % 2 == 0:
        return None
    
    # Try to construct a conference matrix using greedy/random approach
    # For small n, we can use known constructions or random search
    # A simpler approach: use the Paley construction generalized
    # For n = q+1 where q is a prime power ≡ 3 mod 4, we can use Paley-type
    # For now, use random symmetric ±1 matrix with zero diagonal and check properties
    
    best_C = None
    best_score = -1
    deadline = time.time() + 5
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Random symmetric matrix with zero diagonal
        C = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                C[i][j] = random.choice([-1, 1])
                C[j][i] = C[i][j]
        
        # Check if C*C^T ≈ (n-1)*I
        CC = C @ C.T
        off_diag = CC - np.diag(np.diag(CC))
        # We want off-diagonal to be 0 and diagonal to be n-1
        error = np.sum(off_diag**2) + np.sum((np.diag(CC) - (n-1))**2)
        
        if error < best_score:
            best_score = error
            best_C = C.copy()
        
        # If perfect, break early
        if error == 0:
            break
    
    if best_C is None or best_score > n * n:
        return None
    
    # Build graph from best_C
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_C[i][j] == 1:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def get_bitmasks(adj, n):
    """Convert adjacency to bitmask representation for fast operations."""
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def has_k4_bitmask(masks, n):
    """Check if graph contains a K4 using bitmasks."""
    for u in range(n):
        # Get neighbors of u as bitmask
        nu = masks[u]
        # Iterate over pairs of neighbors
        # Extract neighbor list
        neighbors = []
        m = nu
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            neighbors.append(v)
            m ^= lsb
        
        for i in range(len(neighbors)):
            v = neighbors[i]
            # Common neighbors of u and v
            common_uv = masks[u] & masks[v]
            # Remove u and v from consideration
            common_uv &= ~(1 << u)
            common_uv &= ~(1 << v)
            
            # For each pair (v, w) where w is in common_uv, check if they share a common neighbor in common_uv
            # Actually, we need: for triangle u-v-w, check if there exists x in N(u)∩N(v)∩N(w)
            # Let's iterate over w in common_uv
            mw = common_uv
            while mw:
                lsb = mw & (-mw)
                w = lsb.bit_length() - 1
                mw ^= lsb
                
                # Check if there's a common neighbor of v and w that is also in N(u)
                # common_vw = masks[v] & masks[w]
                # We need x in common_vw ∩ masks[u], x != u, v, w
                common_vw = masks[v] & masks[w]
                common_vw &= masks[u]
                common_vw &= ~(1 << u)
                common_vw &= ~(1 << v)
                common_vw &= ~(1 << w)
                
                if common_vw:
                    return True
    return False

def has_independent_set_bitmask(masks, n, k):
    """Check if graph contains an independent set of size k using bitmask backtracking."""
    # An independent set in G is a clique in complement
    # Complement mask for vertex v: all bits set except v and its neighbors
    all_mask = (1 << n) - 1
    comp_masks = [0] * n
    for v in range(n):
        comp_masks[v] = all_mask & ~(masks[v] | (1 << v))
    
    def backtrack(candidates, depth, remaining):
        if remaining == 0:
            return True
        if candidates.bit_count() < remaining:
            return False
        if depth >= n:
            return False
        
        # Choose a vertex from candidates
        # Use a heuristic: pick vertex with fewest candidates in its neighborhood
        # For simplicity, pick the first available
        m = candidates
        if m == 0:
            return False
        
        # Try each vertex in candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # New candidates: intersection of current candidates and complement neighborhood of v
            new_candidates = candidates & comp_masks[v]
            # Remove v from new_candidates
            new_candidates &= ~(1 << v)
            
            if new_candidates.bit_count() < remaining - 1:
                continue
            
            if backtrack(new_candidates, depth + 1, remaining - 1):
                return True
        
        return False
    
    return backtrack((1 << n) - 1, 0, k)

def has_independent_set_on_subset_bitmask(masks, vertices, k):
    """Check if induced subgraph on vertices has independent set of size k."""
    if len(vertices) < k:
        return False
    
    # Map vertices to local indices
    vert_list = list(vertices)
    vert_to_idx = {v: i for i, v in enumerate(vert_list)}
    m = len(vert_list)
    
    # Build local masks
    local_masks = [0] * m
    for i, v in enumerate(vert_list):
        mask = masks[v]
        for j, w in enumerate(vert_list):
            if w != v and (mask >> w) & 1:
                local_masks[i] |= (1 << j)
    
    # Now check for independent set of size k in this local graph
    all_mask = (1 << m) - 1
    comp_masks = [0] * m
    for v in range(m):
        comp_masks[v] = all_mask & ~(local_masks[v] | (1 << v))
    
    def backtrack(candidates, remaining):
        if remaining == 0:
            return True
        if candidates.bit_count() < remaining:
            return False
        
        m_local = candidates
        while m_local:
            lsb = m_local & (-m_local)
            v = lsb.bit_length() - 1
            m_local ^= lsb
            
            new_candidates = candidates & comp_masks[v]
            new_candidates &= ~(1 << v)
            
            if new_candidates.bit_count() < remaining - 1:
                continue
            
            if backtrack(new_candidates, remaining - 1):
                return True
        
        return False
    
    return backtrack((1 << m) - 1, k)

def spectral_score(masks, n):
    """Compute a spectral-based score for the graph.
    Higher score = better separation between clique and independent set constraints.
    We use the eigenvalue gap of the adjacency matrix.
    """
    # Build adjacency matrix
    A = np.zeros((n, n))
    for u in range(n):
        for v in range(u+1, n):
            if (masks[u] >> v) & 1:
                A[u][v] = 1
                A[v][u] = 1
    
    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(A)
    except:
        return 0
    
    # For a good Ramsey graph, we want:
    # - No K4: related to eigenvalues (if lambda_max < (n-3)/2 roughly, no K4)
    # - No independent set of 6: related to eigenvalues of complement
    # 
    # A simpler heuristic: maximize the spectral gap
    # lambda_max - lambda_min should be large
    gap = eigenvalues[-1] - eigenvalues[0]
    
    # Also penalize if lambda_max is too high (risk of K4)
    # And if lambda_min is too low (risk of large independent set in complement)
    penalty = 0
    if eigenvalues[-1] > n * 0.4:
        penalty += (eigenvalues[-1] - n * 0.4) * 10
    if eigenvalues[0] < -n * 0.4:
        penalty += (-n * 0.4 - eigenvalues[0]) * 10
    
    return gap - penalty

def local_search_polish(edges, n, time_limit):
    """Perform local search by trying edge swaps to improve spectral properties
    while maintaining K4-free and alpha <= 5 constraints."""
    adj = build_adjacency(n, edges)
    masks = get_bitmasks(adj, n)
    
    # Verify initial validity
    if has_k4_bitmask(masks, n) or has_independent_set_bitmask(masks, n, 6):
        return edges
    
    best_edges = list(edges)
    best_score = spectral_score(masks, n)
    deadline = time.time() + time_limit
    
    # Precompute all possible edge positions
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Try random edge swaps
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Pick two random pairs: one to add, one to remove
            if len(edges) < 10:
                # Just try adding
                pair = random.choice(all_pairs)
                if pair not in adj[pair[0]]:
                    u, v = pair
                    # Check K4: N(u) ∩ N(v) must be triangle-free
                    common = masks[u] & masks[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    # Check if common contains a triangle
                    valid = True
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        a = lsb.bit_length() - 1
                        cm ^= lsb
                        # Check if a has a neighbor in common
                        if masks[a] & common & ~(1 << a):
                            # Need to check if there's a triangle
                            # For each pair in common ∩ N(a), check if they're connected
                            na_in_common = masks[a] & common & ~(1 << a)
                            nm = na_in_common
                            while nm:
                                lsb2 = nm & (-nm)
                                b = lsb2.bit_length() - 1
                                nm ^= lsb2
                                if (masks[b] >> a) & 1 and (masks[b] & common & ~(1 << b) & ~(1 << a)):
                                    # Check if a and b have a common neighbor in common
                                    # This is getting complex, let's just check K4 after adding
                                    pass
                            break
                        # Actually, let's just do the full K4 check after tentative add
                    # Simpler: tentatively add, check, revert
                    adj[u].add(v)
                    adj[v].add(u)
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    
                    if not has_k4_bitmask(masks, n) and not has_independent_set_bitmask(masks, n, 6):
                        score = spectral_score(masks, n)
                        if score > best_score:
                            best_score = score
                            best_edges = list(adj_pairs_to_edges(adj, n))
                            improved = True
                        else:
                            # Revert
                            adj[u].remove(v)
                            adj[v].remove(u)
                            masks[u] &= ~(1 << v)
                            masks[v] &= ~(1 << u)
                    else:
                        # Revert
                        adj[u].remove(v)
                        adj[v].remove(u)
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                continue
            
            # Edge swap: remove one edge, add another
            edge_to_remove = random.choice(edges)
            u1, v1 = edge_to_remove
            if u1 not in adj[v1]:
                continue
                
            # Find an edge to add
            pair_to_add = random.choice(all_pairs)
            u2, v2 = pair_to_add
            if (u2, v2) in [(u1, v1), (v1, u1)]:
                continue
            if u2 in adj[v2]:
                continue
            
            # Tentatively perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            masks[u1] &= ~(1 << v1)
            masks[v1] &= ~(1 << u1)
            
            adj[u2].add(v2)
            adj[v2].add(u2)
            masks[u2] |= (1 << v2)
            masks[v2] |= (1 << u2)
            
            if not has_k4_bitmask(masks, n) and not has_independent_set_bitmask(masks, n, 6):
                score = spectral_score(masks, n)
                if score > best_score:
                    best_score = score
                    best_edges = adj_pairs_to_edges(adj, n)
                    improved = True
                else:
                    # Revert
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    masks[u2] &= ~(1 << v2)
                    masks[v2] &= ~(1 << u2)
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    masks[u1] |= (1 << v1)
                    masks[v1] |= (1 << u1)
            else:
                # Revert
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                masks[u2] &= ~(1 << v2)
                masks[v2] &= ~(1 << u2)
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                masks[u1] |= (1 << v1)
                masks[v1] |= (1 << u1)
        
        if not improved:
            # Try some random restarts of the local search
            break
    
    return best_edges

def adj_pairs_to_edges(adj, n):
    """Convert adjacency sets to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges

def try_add_vertex_bitmask(masks, n, deadline):
    """Try to add a new vertex to the graph while maintaining constraints."""
    new_v = n
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        # Random adjacency for new vertex
        new_adj_mask = 0
        for u in range(n):
            if random.random() < 0.5:
                new_adj_mask |= (1 << u)
        
        # Check K4 constraint:
        # The subgraph induced by new_adj must be triangle-free
        # i.e., for any three vertices a, b, c in new_adj, not all three are connected
        valid_k4 = True
        na = new_adj_mask
        while na and valid_k4:
            lsb = na & (-na)
            a = lsb.bit_length() - 1
            na ^= lsb
            
            # For each pair (a, b) in new_adj where a-b is an edge,
            # check if there's a c in new_adj such that a-c and b-c are edges
            na_b = new_adj_mask & masks[a] & ~(1 << a)
            while na_b and valid_k4:
                lsb_b = na_b & (-na_b)
                b = lsb_b.bit_length() - 1
                na_b ^= lsb_b
                
                # Check if there's a c in new_adj connected to both a and b
                common_ab = masks[a] & masks[b] & new_adj_mask
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                
                if common_ab:
                    valid_k4 = False
                    break
        
        if not valid_k4:
            continue
        
        # Check independent set constraint:
        # Non-neighbors of new_v must not contain an independent set of size 5
        non_neighbors_mask = ((1 << n) - 1) & ~new_adj_mask
        
        if non_neighbors_mask.bit_count() >= 5:
            # Check if induced subgraph on non_neighbors has independent set of size 5
            # Convert to local representation
            nn_list = []
            m = non_neighbors_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                nn_list.append(v)
                m ^= lsb
            
            if has_independent_set_on_subset_bitmask(masks, nn_list, 5):
                continue
        
        # Valid! Return the new adjacency
        return new_adj_mask
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    best_graph = None
    best_n = 0
    
    # Try Paley graphs
    for q in [17, 29]:
        if time.time() > deadline:
            break
        g = generate_paley(q)
        if g:
            adj = build_adjacency(q, g['edges'])
            masks = get_bitmasks(adj, q)
            if not has_k4_bitmask(masks, q) and not has_independent_set_bitmask(masks, q, 6):
                best_graph = g
                best_n = q
    
    # Try conference graph for n=35
    if time.time() < deadline:
        g35 = generate_conference_graph(35)
        if g35:
            adj35 = build_adjacency(35, g35['edges'])
            masks35 = get_bitmasks(adj35, 35)
            if not has_k4_bitmask(masks35, 35) and not has_independent_set_bitmask(masks35, 35, 6):
                best_graph = g35
                best_n = 35
    
    # Try other n values with conference graph
    if time.time() < deadline:
        for n_try in [33, 31, 29, 27, 25]:
            if time.time() > deadline:
                break
            if n_try <= best_n:
                continue
            g_try = generate_conference_graph(n_try)
            if g_try:
                adj_try = build_adjacency(n_try, g_try['edges'])
                masks_try = get_bitmasks(adj_try, n_try)
                if not has_k4_bitmask(masks_try, n_try) and not has_independent_set_bitmask(masks_try, n_try, 6):
                    best_graph = g_try
                    best_n = n_try
    
    if best_graph is None:
        # Fallback: try small graphs
        for n_try in range(10, 1, -1):
            if time.time() > deadline:
                break
            # Try random regular-ish graph
            for attempt in range(50):
                if time.time() > deadline:
                    break
                edges = []
                for i in range(n_try):
                    for j in range(i+1, n_try):
                        if random.random() < 0.4:
                            edges.append([i, j])
                adj = build_adjacency(n_try, edges)
                masks = get_bitmasks(adj, n_try)
                if not has_k4_bitmask(masks, n_try) and not has_independent_set_bitmask(masks, n_try, 6):
                    best_graph = {'n': n_try, 'edges': edges}
                    best_n = n_try
                    break
            if best_graph and best_n >= n_try:
                break
    
    if best_graph is None:
        return {'n': 1, 'edges': []}
    
    current_n = best_n
    current_edges = list(best_graph['edges'])
    current_adj = build_adjacency(current_n, current_edges)
    current_masks = get_bitmasks(current_adj, current_n)
    
    # Try to extend by adding vertices
    target_n = 35
    while current_n < target_n and time.time() < deadline:
        new_adj_mask = try_add_vertex_bitmask(current_masks, current_n, deadline)
        if new_adj_mask is None:
            # Try a few more times with different random seeds
            found = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                new_adj_mask = try_add_vertex_bitmask(current_masks, current_n, deadline)
                if new_adj_mask is not None:
                    found = True
                    break
            if not found:
                break
        
        new_v = current_n
        # Add edges
        for u in range(current_n):
            if (new_adj_mask >> u) & 1:
                current_edges.append([new_v, u])
                current_adj[u].add(new_v)
        current_adj.append(set())
        for u in range(current_n):
            if (new_adj_mask >> u) & 1:
                current_adj[new_v].add(u)
        current_n += 1
        current_masks = get_bitmasks(current_adj, current_n)
    
    # Local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        current_edges = local_search_polish(current_edges, current_n, remaining_time - 5)
        current_adj = build_adjacency(current_n, current_edges)
        current_masks = get_bitmasks(current_adj, current_n)
        
        # Verify
        if has_k4_bitmask(current_masks, current_n) or has_independent_set_bitmask(current_masks, current_n, 6):
            # Revert to pre-polish
            current_edges = list(best_graph['edges'])
            current_n = best_n
    
    return {'n': current_n, 'edges': current_edges}
