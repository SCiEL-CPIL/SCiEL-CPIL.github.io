import time
import random

_N = 15
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _build_adj_deg(edges):
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _count_violations(adj, deg):
    v_count = 0
    for u in range(_N):
        neighbors_u = adj[u]
        for v in neighbors_u:
            neighbors_v = adj[v]
            for s in neighbors_u:
                if s == v:
                    continue
                ds = deg[s]
                for t in neighbors_v:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        v_count += 1
    return v_count

def _is_valid(adj, deg):
    return _count_violations(adj, deg) == 0

def _local_search_add(g, adj, deg, deadline):
    """Try to add as many edges as possible."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [e for e in _ALL_EDGES if e not in g]
        random.shuffle(missing)
        for e in missing:
            u, v = e
            # Check if adding e keeps graph valid
            # We need to check if adding (u,v) creates any violation.
            # This requires a full check or incremental.
            # Full check is safe.
            new_g = g | {e}
            new_adj, new_deg = _build_adj_deg(new_g)
            if _is_valid(new_adj, new_deg):
                g = new_g
                adj = new_adj
                deg = new_deg
                improved = True
                break
    return g, adj, deg

def _solve_sa(rng, deadline):
    # Start with a random sparse graph or star
    # Let's start with a star to ensure validity
    g = set()
    for i in range(1, _N):
        g.add((0, i))
    
    adj, deg = _build_adj_deg(g)
    
    # Phase 1: Local Search Add
    g, adj, deg = _local_search_add(g, adj, deg, deadline)
    
    # Phase 2: Simulated Annealing to escape local minima
    # If we are stuck, we can try removing an edge and adding others
    # Or use SA on the whole graph
    
    # For now, return the result of local search
    # We can enhance this with restarts in the main loop
    
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_g = None
    best_size = -1
    
    restart_count = 0
    base_seed = 42
    
    while time.time() < deadline:
        rng = random.Random(base_seed + restart_count)
        
        # Strategy: Randomize the initial "star" center or start with random sparse graph
        # Let's try starting with a random subset of edges
        # Or just use the star with a random center
        center = rng.randint(0, _N - 1)
        g = set()
        for i in range(_N):
            if i != center:
                g.add((min(center, i), max(center, i)))
        
        # Local Search Add
        adj, deg = _build_adj_deg(g)
        g, adj, deg = _local_search_add(g, adj, deg, deadline)
        
        if _is_valid(adj, deg):
            if len(g) > best_size:
                best_size = len(g)
                best_g = g
        
        restart_count += 1

    if best_g is None:
        best_g = {(0, i) for i in range(1, _N)}
    
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])
        
    return result
