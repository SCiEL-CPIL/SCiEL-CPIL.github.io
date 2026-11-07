import time
import random

# EVOLVE-BLOCK-START
_N = 15
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0  # Safety margin before 90s kill

def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _build_adj_deg(edges):
    """Build adjacency lists and degree array from a set of edges."""
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _is_valid_fast(adj, deg, edges):
    """
    Check if the graph is valid: no path s-u-v-t with deg[s] == deg[t].
    For each edge (u,v), for each neighbor s of u (s != v), for each neighbor t of v (t != u, t != s),
    check deg[s] != deg[t].
    """
    for u, v in edges:
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        for s in neighbors_u:
            if s == v:
                continue
            ds = deg[s]
            for t in neighbors_v:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _violation_edges(adj, deg, edges):
    """
    Return a dict mapping each edge to the number of violations it participates in.
    A violation is a path s-u-v-t with deg[s] == deg[t].
    """
    vc = {}
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    e1 = _norm((s, u))
                    e2 = _norm((u, v))
                    e3 = _norm((v, t))
                    vc[e1] = vc.get(e1, 0) + 1
                    vc[e2] = vc.get(e2, 0) + 1
                    vc[e3] = vc.get(e3, 0) + 1
    return vc

def _try_add_edge(g, adj, deg, e):
    """Try to add edge e to graph g. Return (new_g, new_adj, new_deg) if valid, else None."""
    u, v = e
    # Fast check: for the new edge (u,v), check all s in adj[u] \ {v} and t in adj[v] \ {u}
    # But adding (u,v) also changes degrees of u and v, which might affect other paths.
    # However, the condition only depends on deg[s] and deg[t] for s, t NOT being u or v in the path s-u-v-t.
    # Wait, the path is s-u-v-t. The degrees of s and t are checked.
    # If we add (u,v), the degrees of u and v increase by 1.
    # Does this affect the validity of existing paths?
    # Existing paths are s'-a-b-t'.
    # If s' or t' is u or v, their degree changes, so deg[s'] or deg[t'] changes.
    # This could create new violations or remove old ones (if they were equal before).
    # So we cannot just check the new edge. We must check the whole graph.
    # However, for a heuristic, we can first check if the new edge itself creates violations.
    # If it does, it's definitely invalid. If not, it might still be invalid due to degree changes.
    
    # Let's do a full check for correctness, but maybe we can optimize.
    # Since N=15, full check is fast enough for a few hundred iterations.
    
    new_g = g | {e}
    new_adj, new_deg = _build_adj_deg(new_g)
    if _is_valid_fast(new_adj, new_deg, new_g):
        return new_g, new_adj, new_deg
    return None

def _solve_once(rng, deadline):
    """
    Perform one run of the search.
    1. Start with a random subset of edges or complete graph and reduce.
    2. Local search: Add edges.
    """
    # Strategy: Start with a dense random graph, then remove violating edges.
    # Or start with complete graph and remove.
    
    g = set(_ALL_EDGES)
    adj, deg = _build_adj_deg(g)
    
    # Phase 1: Remove edges until valid
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        if _is_valid_fast(adj, deg, g):
            break
        
        vc = _violation_edges(adj, deg, g)
        if not vc:
            break
        
        # Remove the edge with the highest violation count
        # Tie-break randomly to escape local minima
        max_count = max(vc.values())
        candidates = [e for e, c in vc.items() if c == max_count]
        e_to_remove = rng.choice(candidates)
        
        g.discard(e_to_remove)
        adj, deg = _build_adj_deg(g)
    
    # Phase 2: Local Search - Add edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Get all missing edges
        missing = [e for e in _ALL_EDGES if e not in g]
        rng.shuffle(missing)
        
        # Try to add edges in random order
        for e in missing:
            if time.time() > deadline:
                break
            
            res = _try_add_edge(g, adj, deg, e)
            if res:
                g, adj, deg = res
                improved = True
                # Restart the loop to try more additions with the new state
                # This is important because adding one edge might enable adding others
                # or we might want to try different orders
                break
        
        # If no single edge could be added, try a swap: remove one, add many
        if not improved and len(g) < _N * (_N - 1) // 2:
            # Try removing a random edge and then adding as many as possible
            current_edges = list(g)
            rng.shuffle(current_edges)
            
            # Try a few random removals
            for e_out in current_edges[:5]:
                if time.time() > deadline:
                    break
                
                g_temp = g - {e_out}
                adj_temp, deg_temp = _build_adj_deg(g_temp)
                
                # Greedily add edges
                missing_temp = [e for e in _ALL_EDGES if e not in g_temp]
                rng.shuffle(missing_temp)
                
                for e_in in missing_temp:
                    if time.time() > deadline:
                        break
                    res = _try_add_edge(g_temp, adj_temp, deg_temp, e_in)
                    if res:
                        g_temp, adj_temp, deg_temp = res
                
                if len(g_temp) > len(g):
                    g, adj, deg = g_temp, adj_temp, deg_temp
                    improved = True
                    break

    return g

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    best_g = None
    best_size = -1
    
    restart_count = 0
    base_seed = 42
    
    while time.time() < deadline:
        rng = random.Random(base_seed + restart_count)
        
        g = _solve_once(rng, deadline)
        
        if g:
            adj, deg = _build_adj_deg(g)
            if _is_valid_fast(adj, deg, g):
                if len(g) > best_size:
                    best_size = len(g)
                    best_g = g
        
        restart_count += 1
        
        # Limit restarts to avoid infinite loop if one restart takes too long
        if restart_count > 50:
            break

    if best_g is None:
        # Fallback: Star graph
        best_g = {(0, i) for i in range(1, _N)}
    
    # Convert to required format
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])
        
    return result
# EVOLVE-BLOCK-END
