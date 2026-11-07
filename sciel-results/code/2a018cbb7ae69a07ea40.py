import time
import random

# EVOLVE-BLOCK-START
_N = 15
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _build_adj_deg(edges):
    """Build adjacency sets and degree array from a set of edges."""
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _is_valid(adj, deg, edges):
    """Check if the graph defined by adj, deg, edges is valid.
    Returns True if no violating paths exist.
    """
    for u, v in edges:
        # Path s-u-v-t
        # s in adj[u], s != v
        # t in adj[v], t != u, t != s
        # Check deg[s] == deg[t]
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _solve_once(rng):
    """
    Perform one run of the search.
    1. Start with complete graph.
    2. Greedily remove edges to become valid.
    3. Local search: Add edges, or Swap (remove one, add many) to increase edge count.
    """
    g = set(_ALL_EDGES)
    adj, deg = _build_adj_deg(g)
    
    # Phase 1: Remove edges until valid
    # We iterate until no violations exist.
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
            
        if _is_valid(adj, deg, g):
            break
            
        # Find one violation to guide removal
        found_violation = False
        candidate_edges = []
        
        for u, v in g:
            for s in adj[u]:
                if s == v: continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s: continue
                    if ds == deg[t]:
                        # Violation found: path s-u-v-t
                        candidate_edges = [
                            _norm((s, u)), 
                            _norm((u, v)), 
                            _norm((v, t))
                        ]
                        found_violation = True
                        break
                if found_violation:
                    break
            if found_violation:
                break
        
        if not found_violation:
            break
            
        # Evaluate which candidate edge removal results in the fewest total violations
        # Optimization: Instead of counting ALL violations (expensive), we can just pick one 
        # that seems "central" or use a quick heuristic. 
        # However, counting violations is O(E * d^2). With E~50, d~6, it's ~1800 ops. 
        # Doing this 1000 times is 1.8M ops. This is fast enough in Python? 
        # Let's stick to the previous logic but maybe limit the candidates evaluated.
        
        best_edge_to_remove = None
        best_violation_count = float('inf')
        
        rng.shuffle(candidate_edges)
        
        for e in candidate_edges:
            if e not in g:
                continue
                
            # Simulate removal
            new_g = g - {e}
            new_adj, new_deg = _build_adj_deg(new_g)
            
            # Count violations in the new graph
            # We can optimize this by only counting violations involving the changed degrees?
            # No, degrees change for u and v, which affects all paths involving them.
            # Full recount is safer.
            v_count = 0
            for u2, v2 in new_g:
                for s2 in new_adj[u2]:
                    if s2 == v2: continue
                    ds2 = new_deg[s2]
                    for t2 in new_adj[v2]:
                        if t2 == u2 or t2 == s2: continue
                        if ds2 == new_deg[t2]:
                            v_count += 1
                            # Early exit if we already exceed current best?
                            if v_count >= best_violation_count:
                                break
                    if v_count >= best_violation_count:
                        break
                if v_count >= best_violation_count:
                    break
            
            if v_count < best_violation_count:
                best_violation_count = v_count
                best_edge_to_remove = e
                
        if best_edge_to_remove is None:
            best_edge_to_remove = candidate_edges[0]
            
        g.discard(best_edge_to_remove)
        adj, deg = _build_adj_deg(g)
        
    # Phase 2: Local Search (Add and Swap)
    # We try to improve the edge count.
    # We use a loop that runs as long as we find an improvement or time runs out.
    # To avoid infinite loops on equal-size swaps, we only accept swaps if size increases.
    # (Swaps that keep size same might be useful for escaping local minima, but 
    #  increasing size is the primary goal. We can add a "kick" step later if needed.)
    
    improved = True
    while improved:
        if time.time() > _DEADLINE:
            break
        improved = False
        
        # 2a. Try to add any single edge
        candidates = [e for e in _ALL_EDGES if e not in g]
        rng.shuffle(candidates)
        
        for e in candidates:
            if time.time() > _DEADLINE:
                break
            new_g = g | {e}
            new_adj, new_deg = _build_adj_deg(new_g)
            if _is_valid(new_adj, new_deg, new_g):
                g = new_g
                adj, deg = new_adj, new_deg
                improved = True
                break # Restart the loop with the new state
        
        # 2b. If no single edge could be added, try a "Swap" move
        # Remove one edge, then greedily add as many as possible.
        # If the final size is larger than current, accept the move.
        if not improved and len(g) < _N * (_N - 1) // 2:
             # Pick a random edge to remove to "open up" space
             # We can bias towards edges that are "central" or just random.
             current_edges = list(g)
             if current_edges:
                 rng.shuffle(current_edges)
                 # Try a few random removals
                 for e_out in current_edges[:5]: # Limit to 5 attempts per iteration to save time
                     if time.time() > _DEADLINE:
                         break
                     
                     g_temp = g - {e_out}
                     adj_temp, deg_temp = _build_adj_deg(g_temp)
                     
                     # Now greedily add edges to g_temp
                     candidates_add = [e for e in _ALL_EDGES if e not in g_temp]
                     rng.shuffle(candidates_add)
                     
                     for e_in in candidates_add:
                         new_g_temp = g_temp | {e_in}
                         new_adj_temp, new_deg_temp = _build_adj_deg(new_g_temp)
                         if _is_valid(new_adj_temp, new_deg_temp, new_g_temp):
                             g_temp = new_g_temp
                             adj_temp, deg_temp = new_adj_temp, new_deg_temp
                     
                     # Check if this swap improved the size
                     if len(g_temp) > len(g):
                         g = g_temp
                         adj, deg = adj_temp, deg_temp
                         improved = True
                         break

    return g

def construct():
    start_time = time.time()
    deadline = 85.0 # Hard limit inside construct for safety
    
    best_g = None
    best_size = -1
    
    # Run multiple restarts with different seeds
    base_seed = 12345
    restart_count = 0
    
    while time.time() - start_time < deadline:
        rng = random.Random(base_seed + 1000 * restart_count)
        
        g = _solve_once(rng)
        
        # Verify final validity (paranoia check)
        if g:
            adj, deg = _build_adj_deg(g)
            if _is_valid(adj, deg, g):
                if len(g) > best_size:
                    best_size = len(g)
                    best_g = g
        
        restart_count += 1
        
        # If we've done a reasonable number of restarts, break to ensure return
        if restart_count > 100:
            break

    if best_g is None:
        # Fallback: Star graph is always valid
        best_g = {(0, i) for i in range(1, _N)}
        
    # Convert to required format: list of [u, v] with Python ints
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])
        
    return result
# EVOLVE-BLOCK-END
