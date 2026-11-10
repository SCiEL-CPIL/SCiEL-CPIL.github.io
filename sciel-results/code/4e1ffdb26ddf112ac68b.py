import time
import random
import itertools

def _max_tf_size(adj_mask, n):
    """Find the largest triangle-free subset size using branch and bound."""
    # For each vertex, precompute its neighbors as a bitmask
    # Use recursive backtracking to find max triangle-free induced subgraph
    
    # First, try to find if there's a triangle-free set of size k
    # We'll use a greedy approach with backtracking
    
    def _has_tf_set(adj_mask, n, target_size):
        """Check if there exists a triangle-free induced subgraph of size target_size."""
        # Greedy: try to build a triangle-free set of size target_size
        # A set S is triangle-free iff for every v in S, the neighbors of v in S form an independent set
        
        # Use a simpler approach: enumerate subsets of size target_size
        # But that's C(18,12) = 18564, which is manageable
        for subset in itertools.combinations(range(n), target_size):
            # Check if this subset induces a triangle-free graph
            # Triangle-free means no three vertices in the subset form a triangle
            found_triangle = False
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    u, v = subset[i], subset[j]
                    if not (adj_mask[u] >> v) & 1:
                        continue
                    # u and v are adjacent; check if any w in subset is adjacent to both
                    common = adj_mask[u] & adj_mask[v]
                    for k in range(j+1, len(subset)):
                        w = subset[k]
                        if (common >> w) & 1:
                            found_triangle = True
                            break
                    if found_triangle:
                        break
                if found_triangle:
                    break
            if not found_triangle:
                return True
        return False
    
    # Binary search on the answer
    lo, hi = 1, n
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_tf_set(adj_mask, n, mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def _is_k4_free_mask(adj_mask, n):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_mask[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = adj_mask[a] & adj_mask[b]
            # For each c in common, check if c has any neighbor in common
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True

def _find_largest_tf(adj_mask, n):
    """Find the largest triangle-free induced subgraph size using branch and bound."""
    # Precompute adjacency as bitmasks (already have adj_mask)
    # Use a greedy + local search approach
    
    # Step 1: Greedy construction
    def greedy_tf(adj_mask, n, order):
        """Build a triangle-free set greedily in given vertex order."""
        selected = []
        selected_mask = 0
        for v in order:
            # Check if adding v keeps the set triangle-free
            # v can be added iff: for every pair (a,b) in selected that are both neighbors of v, a and b are NOT adjacent
            # Equivalently: the neighbors of v in selected must form an independent set
            neigh_in_sel = adj_mask[v] & selected_mask
            # Check if neigh_in_sel is an independent set
            is_indep = True
            nb = neigh_in_sel
            while nb:
                lb = nb & (-nb)
                a = lb.bit_length() - 1
                if adj_mask[a] & neigh_in_sel:
                    is_indep = False
                    break
                nb ^= lb
            if is_indep:
                selected.append(v)
                selected_mask |= (1 << v)
        return selected
    
    # Step 2: Try multiple random orderings and take the best
    best_size = 0
    rng = random.Random(12345)
    for _ in range(200):
        order = list(range(n))
        rng.shuffle(order)
        sel = greedy_tf(adj_mask, n, order)
        if len(sel) > best_size:
            best_size = len(sel)
    
    # Step 3: Local search - try to extend the best solution
    # Get the best greedy solution
    best_sel = []
    for _ in range(50):
        order = list(range(n))
        rng.shuffle(order)
        sel = greedy_tf(adj_mask, n, order)
        if len(sel) > len(best_sel):
            best_sel = sel
    
    # Try to add vertices to best_sel
    selected_mask = 0
    for v in best_sel:
        selected_mask |= (1 << v)
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if (selected_mask >> v) & 1:
                continue
            neigh_in_sel = adj_mask[v] & selected_mask
            is_indep = True
            nb = neigh_in_sel
            while nb:
                lb = nb & (-nb)
                a = lb.bit_length() - 1
                if adj_mask[a] & neigh_in_sel:
                    is_indep = False
                    break
                nb ^= lb
            if is_indep:
                selected_mask |= (1 << v)
                improved = True
    
    best_size = max(best_size, selected_mask.bit_count())
    
    # Step 4: Verify with exhaustive check for small sizes
    # If best_size >= 12, verify that no 12-set is triangle-free
    if best_size >= 12:
        # Check all 12-subsets
        for subset in itertools.combinations(range(n), 12):
            found_triangle = False
            for i in range(12):
                for j in range(i+1, 12):
                    u, v = subset[i], subset[j]
                    if not (adj_mask[u] >> v) & 1:
                        continue
                    common = adj_mask[u] & adj_mask[v]
                    for k in range(j+1, 12):
                        w = subset[k]
                        if (common >> w) & 1:
                            found_triangle = True
                            break
                    if found_triangle:
                        break
                if found_triangle:
                    break
            if not found_triangle:
                return 12
    
    return best_size

def _is_k4_free_after_add(adj_mask, n, u, v):
    """Fast check: after adding edge (u,v), is graph still K4-free?
    Only need to check K4s containing the new edge (u,v).
    A K4 containing edge (u,v) has two other vertices c,d that are:
    - common neighbors of u and v (i.e., adjacent to both u and v)
    - c and d are adjacent to each other
    So: among common neighbors of u,v, is there an edge?"""
    common = adj_mask[u] & adj_mask[v]
    c_bits = common
    while c_bits:
        lb = c_bits & (-c_bits)
        c = lb.bit_length() - 1
        if adj_mask[c] & common:
            return False
        c_bits ^= lb
    return True

def _compute_score(adj_mask, n):
    """Compute the largest triangle-free induced subgraph size."""
    return _find_largest_tf(adj_mask, n)

def _dpll_add_edges_to_hit_12(adj_mask, n, max_time):
    """DPLL-style: add edges to ensure every 12-set contains a triangle.
    Returns modified adj_mask or None if timeout."""
    start = time.time()
    deadline = start + max_time
    
    # Find all current 12-sets that are triangle-free
    def get_tf_12_sets(adj_mask, n):
        tf_sets = []
        for subset in itertools.combinations(range(n), 12):
            found_triangle = False
            for i in range(12):
                for j in range(i+1, 12):
                    u, v = subset[i], subset[j]
                    if not (adj_mask[u] >> v) & 1:
                        continue
                    common = adj_mask[u] & adj_mask[v]
                    for k in range(j+1, 12):
                        w = subset[k]
                        if (common >> w) & 1:
                            found_triangle = True
                            break
                    if found_triangle:
                        break
                if found_triangle:
                    break
            if not found_triangle:
                tf_sets.append(subset)
        return tf_sets
    
    # Iteratively find a triangle-free 12-set and add an edge to break it
    for iteration in range(200):
        if time.time() > deadline:
            return None
        
        tf_sets = get_tf_12_sets(adj_mask, n)
        if not tf_sets:
            return adj_mask  # Success! No triangle-free 12-sets
        
        # Pick the first triangle-free 12-set
        target = tf_sets[0]
        
        # Find pairs in target that are non-adjacent; adding an edge between them
        # might create a triangle in this set
        # For a triangle to form in target, we need three mutually adjacent vertices
        # Try adding edges between non-adjacent pairs in target
        candidates = []
        for i in range(12):
            for j in range(i+1, 12):
                u, v = target[i], target[j]
                if (adj_mask[u] >> v) & 1:
                    continue  # Already adjacent
                # Check if adding (u,v) would create a triangle in target
                # Need a w in target adjacent to both u and v
                common = adj_mask[u] & adj_mask[v]
                for k in range(12):
                    if k == i or k == j:
                        continue
                    w = target[k]
                    if (common >> w) & 1:
                        candidates.append((u, v))
                        break
        
        if not candidates:
            # No single edge addition can create a triangle in this set
            # This means the set is an independent set or has no "almost triangles"
            # We need a different strategy - maybe the graph structure is problematic
            # Try a different approach: look at the structure
            return None
        
        # Pick a random candidate
        u, v = random.choice(candidates)
        
        # Add the edge if K4-free
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        if not _is_k4_free_after_add(adj_mask, n, u, v):
            # Can't add this edge; try another
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            # Try other candidates
            found = False
            for cu, cv in candidates:
                if cu == u and cv == v:
                    continue
                adj_mask[cu] |= (1 << cv)
                adj_mask[cv] |= (1 << cu)
                if _is_k4_free_after_add(adj_mask, n, cu, cv):
                    found = True
                    break
                else:
                    adj_mask[cu] &= ~(1 << cv)
                    adj_mask[cv] &= ~(1 << cu)
            if not found:
                # Cannot fix this set with single edge additions
                return None
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin for 150s limit
    n = 18
    
    def time_ok():
        return time.time() < deadline
    
    best_score = 18  # worst case
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        """Return (score, adj_mask) if K4-free, else None."""
        if not _is_k4_free_mask(adj_mask, n):
            return None
        score = _compute_score(adj_mask, n)
        return (score, adj_mask)
    
    # Construction 1: K6,6,6 (Turan graph)
    adj1 = [0] * n
    for i in range(6):
        for j in range(6, 12):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
        for j in range(12, 18):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    
    result = evaluate(adj1)
    if result:
        best_score, best_adj_mask = result
    
    # Construction 2: Random dense K4-free graphs with fast incremental check
    random.seed(42)
    
    for trial in range(200):
        if not time_ok():
            break
        
        # Start with empty graph, add random edges while maintaining K4-free
        adj = [0] * n
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        for u, v in edges:
            if not time_ok():
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_after_add(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        result = evaluate(adj)
        if result:
            score, _ = result
            if score < best_score:
                best_score = score
                best_adj_mask = adj[:]
        
        # DPLL-style: try to add edges to eliminate triangle-free 12-sets
        if score <= 12 and time_ok():
            dpll_result = _dpll_add_edges_to_hit_12(adj, n, 5.0)
            if dpll_result is not None:
                result2 = evaluate(dpll_result)
                if result2:
                    score2, _ = result2
                    if score2 < best_score:
                        best_score = score2
                        best_adj_mask = dpll_result[:]
    
    # Construction 3: Local search - greedy edge addition with fast check
    if time_ok():
        adj = best_adj_mask[:]
        for pass_num in range(5):
            if not time_ok():
                break
            edges = list(itertools.combinations(range(n), 2))
            random.shuffle(edges)
            improved = False
            for u, v in edges:
                if not time_ok():
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_after_add(adj, n, u, v):
                    result = evaluate(adj)
                    if result:
                        score, _ = result
                        if score < best_score:
                            best_score = score
                            best_adj_mask = adj[:]
                            improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            if not improved and pass_num > 0:
                break
    
    # Construction 4: DPLL on the best graph found so far
    if time_ok():
        adj = best_adj_mask[:]
        dpll_result = _dpll_add_edges_to_hit_12(adj, n, 30.0)
        if dpll_result is not None:
            result = evaluate(dpll_result)
            if result:
                score, _ = result
                if score < best_score:
                    best_score = score
                    best_adj_mask = dpll_result[:]
    
    # Construction 5: More random trials with different seeds
    for seed in [1, 7, 13, 99, 123, 456, 789]:
        if not time_ok():
            break
        random.seed(seed)
        for trial in range(30):
            if not time_ok():
                break
            adj = [0] * n
            edges = list(itertools.combinations(range(n), 2))
            random.shuffle(edges)
            for u, v in edges:
                if not time_ok():
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_k4_free_after_add(adj, n, u, v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            result = evaluate(adj)
            if result:
                score, _ = result
                if score < best_score:
                    best_score = score
                    best_adj_mask = adj[:]
                elif score <= 12:
                    dpll_result = _dpll_add_edges_to_hit_12(adj, n, 3.0)
                    if dpll_result is not None:
                        result2 = evaluate(dpll_result)
                        if result2:
                            score2, _ = result2
                            if score2 < best_score:
                                best_score = score2
                                best_adj_mask = dpll_result[:]
    
    # Construction 6: Final polish - try all single-edge additions to best graph
    if time_ok():
        adj = best_adj_mask[:]
        for u in range(n):
            for v in range(u+1, n):
                if not time_ok():
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_after_add(adj, n, u, v):
                    score = _compute_score(adj, n)
                    if score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
    
    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_mask[u] >> v) & 1:
                best_edges.append([u, v])
    
    return best_edges
