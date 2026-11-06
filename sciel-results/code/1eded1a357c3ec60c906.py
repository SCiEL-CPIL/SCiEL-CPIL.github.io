"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    total_pairs = n * (n - 1) // 2  # 120
    
    def build_bitmasks(edges):
        """Build adjacency bitmasks and degree array from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_bitmask(adj, deg):
        """Compute max triangle degree sum using bitmask operations."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j, only vertices > j to avoid duplicates
                common = adj[i] & adj[j] & (~((1 << (j + 1)) - 1))
                while common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
                    common ^= k_bit
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_bitmasks(edges)
        return compute_score_bitmask(adj, deg)
    
    def spectral_lambda(adj, deg, iters=50):
        """Estimate largest eigenvalue via power iteration on adjacency matrix."""
        # Use random vector
        v = [random.random() for _ in range(n)]
        for _ in range(iters):
            new_v = [0.0] * n
            for i in range(n):
                # Multiply row i of adjacency by v
                ai = adj[i]
                s = 0.0
                while ai:
                    bit = ai & (-ai)
                    j = bit.bit_length() - 1
                    s += v[j]
                    ai ^= bit
                new_v[i] = s
            # Normalize
            norm = sum(x * x for x in new_v) ** 0.5
            if norm < 1e-12:
                break
            for i in range(n):
                new_v[i] /= norm
            v = new_v
        # Rayleigh quotient
        num = 0.0
        den = 0.0
        for i in range(n):
            ai = adj[i]
            s = 0.0
            while ai:
                bit = ai & (-ai)
                j = bit.bit_length() - 1
                s += v[j]
                ai ^= bit
            num += v[i] * s
            den += v[i] * v[i]
        if den < 1e-12:
            return 0.0
        return num / den
    
    def get_non_edges(adj, exclude=None):
        """Get list of non-edges (pairs not in graph)."""
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    if exclude is None or (i, j) != exclude:
                        non_edges.append((i, j))
        return non_edges
    
    def local_search(edges, max_iters=50000):
        """Fast local search using bitmask scoring with spectral guidance."""
        adj, deg = build_bitmasks(edges)
        current_score = compute_score_bitmask(adj, deg)
        
        # Build non-edge list
        non_edges = get_non_edges(adj)
        if not non_edges:
            return edges, current_score
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Spectral guidance: compute eigenvalue for ordering
            # Try all possible swaps: remove edge i, add non-edge j
            # For efficiency, sample swaps guided by degree imbalance
            n_edges = len(edges)
            
            # Build list of (edge_idx, non_edge) candidates, prioritized by
            # removing high-degree vertex edges and adding to low-degree vertices
            candidates = []
            for ei in range(n_edges):
                u1, v1 = edges[ei]
                # Priority: prefer removing edges incident to high-degree vertices
                priority = deg[u1] + deg[v1]
                for ne in non_edges:
                    u2, v2 = ne
                    # Don't add back the same edge
                    if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                        continue
                    # Prefer adding edges that reduce degree variance
                    # New degrees would be: u1-1, v1-1, u2+1, v2+1
                    # This is a heuristic
                    candidates.append((priority, ei, ne))
            
            # Sort by priority (high degree removal first) and take a subset
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -x[0])
            
            # Try top candidates
            tried = 0
            max_tries = min(len(candidates), 200)
            for priority, ei, ne in candidates[:max_tries]:
                if time.time() > deadline - 5.0:
                    break
                tried += 1
                
                u1, v1 = edges[ei]
                u2, v2 = ne
                
                # Apply swap to bitmasks
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = compute_score_bitmask(adj, deg)
                
                if new_score < current_score:
                    # Accept
                    edges[ei] = [u2, v2]
                    current_score = new_score
                    improved = True
                    # Rebuild non_edges
                    non_edges = get_non_edges(adj)
                    break
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
        
        return edges, current_score
    
    def local_search_random(edges, max_iters=20000):
        """Random local search with bitmask scoring."""
        adj, deg = build_bitmasks(edges)
        current_score = compute_score_bitmask(adj, deg)
        non_edges = get_non_edges(adj)
        if not non_edges:
            return edges, current_score
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            n_edges = len(edges)
            ei = random.randint(0, n_edges - 1)
            u1, v1 = edges[ei]
            
            # Try a few random non-edges
            for _ in range(10):
                if not non_edges:
                    break
                ne_idx = random.randint(0, len(non_edges) - 1)
                u2, v2 = non_edges[ne_idx]
                if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                    continue
                
                # Apply swap
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = compute_score_bitmask(adj, deg)
                
                if new_score < current_score:
                    edges[ei] = [u2, v2]
                    current_score = new_score
                    improved = True
                    non_edges = get_non_edges(adj)
                    break
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
        
        return edges, current_score
    
    # K_{6,10} + 5 internal edges baseline
    # Part A: vertices 0-5 (6 vertices), Part B: vertices 6-15 (10 vertices)
    # Complete bipartite: 6*10 = 60 edges
    # Need 5 more edges, all within Part B (to create triangles)
    # Place them to minimize triangle degree sums
    # In K_{6,10}, deg(A)=10, deg(B)=6
    # An internal edge in B: both endpoints get +1 degree
    # Triangle: one A vertex connected to two B vertices that are connected
    # deg sum = 10 + (6+1) + (6+1) = 24
    
    # Place 5 internal edges in B to spread out
    # B vertices: 6,7,8,9,10,11,12,13,14,15
    # Use a 5-cycle or star-like structure
    # Let's use edges that form a path/cycle to distribute
    internal_edges = [(6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
    
    base_edges = []
    for u in range(6):
        for v in range(6, 16):
            base_edges.append([u, v])
    for u, v in internal_edges:
        base_edges.append([u, v])
    
    assert len(base_edges) == 65, f"Expected 65 edges, got {len(base_edges)}"
    base_score = compute_score(base_edges)
    
    best_edges = base_edges[:]
    best_score = base_score
    
    # Also try K_{8,8}+1 as alternative starting point
    alt_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    alt_score = compute_score(alt_edges)
    
    # Run local search from base
    edges_copy = [e[:] for e in base_edges]
    ls_edges, ls_score = local_search(edges_copy, max_iters=30000)
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Run random local search from base
    edges_copy2 = [e[:] for e in base_edges]
    ls_edges2, ls_score2 = local_search_random(edges_copy2, max_iters=20000)
    if ls_score2 < best_score:
        best_score = ls_score2
        best_edges = ls_edges2[:]
    
    # Multiple restarts with perturbations
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb base: swap 2-4 random edges
        perturbed = [e[:] for e in base_edges]
        adj, deg = build_bitmasks(perturbed)
        non_edges = get_non_edges(adj)
        if not non_edges:
            continue
        
        num_swaps = random.randint(2, 5)
        valid = True
        for _ in range(num_swaps):
            ei = random.randint(0, len(perturbed) - 1)
            u1, v1 = perturbed[ei]
            # Find a non-edge to swap in
            ne_idx = random.randint(0, len(non_edges) - 1)
            u2, v2 = non_edges[ne_idx]
            if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                continue
            perturbed[ei] = [u2, v2]
            # Update non_edges
            non_edges = get_non_edges(build_bitmasks(perturbed)[0])
            if not non_edges:
                valid = False
                break
        
        if not valid:
            continue
        
        # Verify
        if len(perturbed) != 65:
            continue
        seen = set()
        dup = False
        for u, v in perturbed:
            if u == v:
                dup = True
                break
            key = (min(u, v), max(u, v))
            if key in seen:
                dup = True
                break
            seen.add(key)
        if dup:
            continue
        
        pert_score = compute_score(perturbed)
        if pert_score < best_score:
            best_score = pert_score
            best_edges = perturbed[:]
        
        # Local search from perturbation
        edges_copy3 = [e[:] for e in perturbed]
        ls_edges3, ls_score3 = local_search(edges_copy3, max_iters=10000)
        if ls_score3 < best_score:
            best_score = ls_score3
            best_edges = ls_edges3[:]
    
    # Final polish: exhaustive 1-edge swap from best
    adj, deg = build_bitmasks(best_edges)
    current_score = best_score
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 3.0:
            break
        non_edges = get_non_edges(adj)
        for ei in range(len(best_edges)):
            if time.time() > deadline - 3.0:
                break
            u1, v1 = best_edges[ei]
            for u2, v2 in non_edges:
                if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                    continue
                # Try swap
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = compute_score_bitmask(adj, deg)
                
                if new_score < current_score:
                    best_edges[ei] = [u2, v2]
                    current_score = new_score
                    improved = True
                    non_edges = get_non_edges(adj)
                    break
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
    
    best_score = current_score
    
    # Verify final result
    final_adj, final_deg = build_bitmasks(best_edges)
    final_score = compute_score_bitmask(final_adj, final_deg)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
