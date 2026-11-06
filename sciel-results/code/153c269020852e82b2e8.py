import time

# ---------- Exact 3-colorability (triangle-free graph, 16 vertices) ----------
def _is_3_colorable(adj):
    """adj: list of 16 sets/lists. Returns True if graph is 3-colorable."""
    n = len(adj)
    # Precompute neighbor bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    # Order vertices by descending degree (static)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    # Precompute "earlier neighbors" for each vertex in the order
    earlier_mask = [0] * n
    for i, v in enumerate(order):
        mask = 0
        for w in adj[v]:
            if pos[w] < i:
                mask |= (1 << w)
        earlier_mask[i] = mask

    # color_mask[c] is a bitmask of vertices assigned color c
    color_mask = [0, 0, 0]
    color = [-1] * n
    
    def backtrack(i):
        v = order[i]
        vmask = 1 << v
        
        # Try each color
        for c in range(3):
            # Color c is forbidden if any neighbor has color c
            if adj_mask[v] & color_mask[c]:
                continue
            
            # Assign color c to v
            color[v] = c
            color_mask[c] |= vmask
            
            if i == n - 1:
                return True
            if backtrack(i + 1):
                return True
            
            # Undo
            color_mask[c] &= ~vmask
            color[v] = -1
        
        return False

    return backtrack(0)

def _build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    return True

def _can_add_edge(u, v, adj):
    if u == v or v in adj[u]:
        return False
    au = adj[u]
    for w in au:
        if w in adj[v]:
            return False
    return True

def _is_not_3_colorable(adj):
    """Returns True if graph is NOT 3-colorable."""
    return not _is_3_colorable(adj)

def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline

    def try_add_edge(u, v, adj, edges):
        if not _can_add_edge(u, v, adj):
            return None
        new_adj = [s.copy() for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if _is_3_colorable(new_adj):
            return None
        return new_adj, edges + [[u, v]]

    def try_add_edge_fast(u, v, adj, edges):
        """Faster version that avoids copying adj if triangle check fails."""
        if not _can_add_edge(u, v, adj):
            return None
        new_adj = [s.copy() for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if _is_3_colorable(new_adj):
            return None
        return new_adj, edges + [[u, v]]

    def run_greedy(seed_order=None):
        # Start with Groetzsch
        base_edges = _groetzsch_edges()
        adj = _build_adj(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free(adj) or _is_3_colorable(adj):
            return adj, edges, len(edges)

        # Determine order of pairs to try
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        if seed_order is not None:
            import random
            random.seed(seed_order)
            random.shuffle(all_pairs)
        else:
            # Sort by potential gain? Or just fixed order.
            pass

        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    changed = True
        
        # Local Search: 1-1 Swaps to escape local optima
        improved = True
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Try to add any edge
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    improved = True
                    break
            
            if improved or not elapsed_ok():
                continue

            # Phase 2: Try 1-1 swap to change structure
            import random
            if edges:
                # Try multiple random swaps
                for _ in range(20):
                    if not elapsed_ok():
                        break
                    
                    idx = random.randint(0, len(edges) - 1)
                    u_rem, v_rem = edges[idx]
                    
                    cand_adj = [s.copy() for s in adj]
                    cand_adj[u_rem].discard(v_rem)
                    cand_adj[v_rem].discard(u_rem)
                    cand_edges = edges[:idx] + edges[idx+1:]
                    
                    # Try to add a random new edge
                    pairs_to_try = [(u, v) for u in range(n) for v in range(u+1, n) if v not in cand_adj[u]]
                    random.shuffle(pairs_to_try)
                    
                    swapped = False
                    for u_add, v_add in pairs_to_try[:30]:
                        if not elapsed_ok():
                            break
                        if not _can_add_edge(u_add, v_add, cand_adj):
                            continue
                        tmp_adj = [s.copy() for s in cand_adj]
                        tmp_adj[u_add].add(v_add)
                        tmp_adj[v_add].add(u_add)
                        if _is_3_colorable(tmp_adj):
                            continue
                        
                        # Valid swap
                        adj = tmp_adj
                        edges = cand_edges + [[u_add, v_add]]
                        swapped = True
                        break
                    
                    if swapped:
                        improved = True
                        break
        
        return adj, edges, len(edges)

    # Run multiple greedy+local search instances with different random seeds
    import random
    max_restarts = 100
    for i in range(max_restarts):
        if not elapsed_ok():
            break
        
        # Use a different random seed for shuffling the pair order
        seed = i * 1000 + 42
        adj, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]

    # Final validation
    if best_edges:
        final_adj = _build_adj(best_edges, n)
        assert _is_triangle_free(final_adj)
        assert not _is_3_colorable(final_adj)
        
        return best_edges
    else:
        # Fallback to Groetzsch if something went wrong
        base_edges = _groetzsch_edges()
        return [list(e) for e in base_edges]
