import time
import random

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    """
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set(adj_mask, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses bitmasks with greedy coloring bound for speed.
    Returns True if an independent set of size >= limit exists.
    """
    all_mask = (1 << n) - 1
    
    # Greedy coloring of the complement graph (i.e., color the original graph's non-edges)
    # Actually, for independent set in G, we need upper bound on alpha(G).
    # Upper bound: greedy coloring of G gives chi(G), and alpha(G) <= n / chi(G) is not right.
    # Better: color the complement of G. The chromatic number of complement >= alpha(G).
    # Greedy coloring of complement gives an upper bound on alpha(G).
    
    # Actually, the standard approach: 
    # For finding max independent set in G, we can use the fact that 
    # alpha(G) <= number of color classes in any proper coloring of G's complement.
    # But simpler: use a greedy bound.
    
    # Let's use a simpler but effective approach:
    # At each node, compute an upper bound on the max independent set we can still find.
    # Upper bound = current_size + greedy_max_independent_set_bound(rem_mask)
    
    def _greedy_bound(rem_mask):
        """Greedy upper bound on max independent set size in rem_mask."""
        # Greedy: repeatedly pick vertex with min degree in remaining, add to IS, remove its neighbors
        # This gives a LOWER bound on alpha, not upper.
        # For UPPER bound, we need coloring.
        # Simpler upper bound: just popcount(rem_mask)
        # Better: use the fact that in any independent set, no two adjacent vertices.
        # Upper bound via greedy coloring of the subgraph induced by rem_mask:
        # Color the subgraph. The number of colors used is an upper bound on... 
        # Wait, chromatic number of G is NOT an upper bound on alpha(G).
        # Actually: alpha(G) * chi(G) >= n is not generally true.
        # 
        # Let me just use a simple but effective bound:
        # Upper bound = current_size + popcount(rem_mask)
        # And use good vertex ordering for branching.
        return rem_mask.bit_count()
    
    # Better approach: use a proper upper bound via greedy coloring of the complement
    # In the complement of G restricted to rem_mask, a clique in complement = independent set in G.
    # Greedy coloring of G (not complement) gives chi(G), and alpha(G) <= n/chi(G) is wrong.
    # 
    # Actually, the correct relationship: if we properly color G, then each color class is an independent set.
    # So the largest color class size is a LOWER bound on alpha(G).
    # For an UPPER bound, we need to color the complement.
    # 
    # Let's use a different strategy: just use tight branching with good heuristics.
    
    def _branch(rem_mask, current_size):
        if current_size >= limit:
            return True
        if rem_mask == 0:
            return False
        if current_size + rem_mask.bit_count() < limit:
            return False
            
        # Pick vertex with max degree in rem_mask for better pruning
        best_v = -1
        best_deg = -1
        m = rem_mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            deg = (adj_mask[v] & rem_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                
        if best_v == -1:
            return False
            
        # Branch 1: Include best_v
        rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
        if _branch(rem_include, current_size + 1):
            return True
            
        # Branch 2: Exclude best_v
        rem_exclude = rem_mask & ~(1 << best_v)
        if _branch(rem_exclude, current_size):
            return True
            
        return False

    return _branch(all_mask, 0)

def _select_subset_greedy_random(n_full, adj_full_mask, target_n, deadline, rng):
    """
    Select a subset of target_n vertices from n_full.
    Mix of random and greedy (prefer high-degree vertices to reduce alpha).
    Returns (selected_indices, sub_adj_mask) or (None, None) on timeout.
    """
    # Strategy: Start with a random subset, then try to improve by swapping
    # low-degree vertices for high-degree ones.
    
    # First, compute degrees in the full graph
    degrees = [adj_full_mask[i].bit_count() for i in range(n_full)]
    
    # Create a list of (degree, index) sorted by degree descending
    deg_order = sorted(range(n_full), key=lambda i: degrees[i], reverse=True)
    
    # Method 1: Random subset
    selected = set(rng.sample(range(n_full), target_n))
    
    # Method 2: Greedy - take top target_n by degree, then randomize a bit
    # Combine: take 70% from top degrees, 30% random
    top_count = int(target_n * 0.7)
    random_count = target_n - top_count
    
    top_candidates = list(deg_order[:top_count * 2])  # Extra candidates
    rng.shuffle(top_candidates)
    top_selected = set(top_candidates[:top_count])
    
    remaining = [i for i in range(n_full) if i not in top_selected]
    random_selected = set(rng.sample(remaining, min(random_count, len(remaining))))
    
    selected = top_selected | random_selected
    
    # If we don't have enough, add more random
    if len(selected) < target_n:
        remaining = [i for i in range(n_full) if i not in selected]
        extra = rng.sample(remaining, target_n - len(selected))
        selected = selected | set(extra)
    
    # Ensure exactly target_n
    if len(selected) > target_n:
        selected_list = list(selected)
        rng.shuffle(selected_list)
        selected = set(selected_list[:target_n])
    
    # Build subgraph
    selected_list = list(selected)
    old_to_new = {old: new for new, old in enumerate(selected_list)}
    
    sub_adj_mask = [0] * target_n
    for old_i in selected_list:
        new_i = old_to_new[old_i]
        mask = adj_full_mask[old_i]
        # For each neighbor of old_i in the full graph, check if in selected
        m = mask
        while m:
            bit = m & -m
            old_j = bit.bit_length() - 1
            if old_j in old_to_new:
                new_j = old_to_new[old_j]
                sub_adj_mask[new_i] |= (1 << new_j)
            m ^= bit
    
    return selected_list, sub_adj_mask

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    rng = random.Random(42)
    
    # Try q=7 (57 vertices) first
    # Then q=5 (31 vertices) as fallback
    
    qs_to_try = [7, 5]
    
    for q in qs_to_try:
        if time.time() > deadline - 15:
            break
            
        try:
            n_full, edges_full, adj_full = _construct_polarity_graph(q)
            
            # Verify C4-free
            if not _verify_c4_free(n_full, adj_full):
                continue
                
            # Convert to bitmasks
            adj_mask = [0] * n_full
            for i in range(n_full):
                mask = 0
                for j in adj_full[i]:
                    mask |= (1 << j)
                adj_mask[i] = mask
            
            # If n_full <= 43, check if we can use the whole graph or a subset
            if n_full <= 43:
                # Check if full graph has alpha < 11
                if not _has_independent_set(adj_mask, n_full, limit=11):
                    if n_full > best_n:
                        best_n = n_full
                        best_edges = edges_full
                else:
                    # Try subsets of size n_full-1, n_full-2, ... down to best_n+1
                    for target_n in range(n_full - 1, max(best_n, 0), -1):
                        if time.time() > deadline - 5:
                            break
                        # Try a few random subsets
                        for attempt in range(100):
                            if time.time() > deadline - 5:
                                break
                            selected = rng.sample(range(n_full), target_n)
                            selected_set = set(selected)
                            old_to_new = {old: new for new, old in enumerate(selected)}
                            
                            sub_adj_mask = [0] * target_n
                            for old_i in selected:
                                new_i = old_to_new[old_i]
                                for old_j in adj_full[old_i]:
                                    if old_j in selected_set:
                                        new_j = old_to_new[old_j]
                                        sub_adj_mask[new_i] |= (1 << new_j)
                            
                            if not _has_independent_set(sub_adj_mask, target_n, limit=11):
                                candidate_edges = []
                                for i in range(target_n):
                                    mask = sub_adj_mask[i]
                                    while mask:
                                        bit = mask & -mask
                                        j = bit.bit_length() - 1
                                        if i < j:
                                            candidate_edges.append([i, j])
                                        mask ^= bit
                                best_n = target_n
                                best_edges = candidate_edges
                                break
                            if best_n >= target_n:
                                break
                        if best_n >= target_n:
                            break
            else:
                # n_full > 43, need to find subset of size up to 43
                target_sizes = [43, 42, 41, 40, 39]
                
                for target_n in target_sizes:
                    if target_n > n_full:
                        continue
                    if target_n <= best_n:
                        continue
                    
                    if time.time() > deadline - 10:
                        break
                    
                    found = False
                    attempts = 0
                    max_attempts = 2000
                    
                    while attempts < max_attempts and not found:
                        if time.time() > deadline - 5:
                            break
                        attempts += 1
                        
                        # Use greedy-random hybrid selection
                        selected_list, sub_adj_mask = _select_subset_greedy_random(
                            n_full, adj_mask, target_n, deadline, rng
                        )
                        
                        if sub_adj_mask is None:
                            break
                        
                        # Check alpha
                        if not _has_independent_set(sub_adj_mask, target_n, limit=11):
                            found = True
                            candidate_edges = []
                            for i in range(target_n):
                                mask = sub_adj_mask[i]
                                while mask:
                                    bit = mask & -mask
                                    j = bit.bit_length() - 1
                                    if i < j:
                                        candidate_edges.append([i, j])
                                    mask ^= bit
                            best_n = target_n
                            best_edges = candidate_edges
                            break
                        
                    if found:
                        if best_n == 43:
                            return {'n': best_n, 'edges': best_edges}
                        # Found a valid size, break to try next q or stop
                        break
                    
        except Exception:
            continue
            
    if best_n == 0:
        # Fallback: small valid graph
        best_n = 3
        best_edges = [[0, 1], [1, 2], [0, 2]]
        
    return {'n': best_n, 'edges': best_edges}
