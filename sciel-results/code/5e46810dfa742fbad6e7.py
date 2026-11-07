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

def _count_violations(adj, deg, edges):
    """Count the number of violating paths."""
    count = 0
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    count += 1
    return count

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
            v_count = _count_violations(new_adj, new_deg, new_g)
            
            if v_count < best_violation_count:
                best_violation_count = v_count
                best_edge_to_remove = e
                
        if best_edge_to_remove is None:
            best_edge_to_remove = candidate_edges[0]
            
        g.discard(best_edge_to_remove)
        adj, deg = _build_adj_deg(g)
        
    # Phase 2: Local Search (Add and Swap)
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
        if not improved and len(g) < _N * (_N - 1) // 2:
             current_edges = list(g)
             if current_edges:
                 rng.shuffle(current_edges)
                 for e_out in current_edges[:5]:
                     if time.time() > _DEADLINE:
                         break
                     
                     g_temp = g - {e_out}
                     adj_temp, deg_temp = _build_adj_deg(g_temp)
                     
                     candidates_add = [e for e in _ALL_EDGES if e not in g_temp]
                     rng.shuffle(candidates_add)
                     
                     for e_in in candidates_add:
                         new_g_temp = g_temp | {e_in}
                         new_adj_temp, new_deg_temp = _build_adj_deg(new_g_temp)
                         if _is_valid(new_adj_temp, new_deg_temp, new_g_temp):
                             g_temp = new_g_temp
                             adj_temp, deg_temp = new_adj_temp, new_deg_temp
                     
                     if len(g_temp) > len(g):
                         g = g_temp
                         adj, deg = adj_temp, deg_temp
                         improved = True
                         break

    return g

def _construct_distinct_degree_graph():
    """
    Construct a graph where all vertices have distinct degrees.
    This automatically satisfies the constraint since no two vertices have equal degrees.
    
    Strategy: Use a "threshold" construction.
    For n vertices, we want degrees 0, 1, 2, ..., n-1 (or a subset).
    A simple way: vertex i has degree i, connected to vertices 0, 1, ..., i-1.
    But this gives degrees 0, 1, 2, ..., n-1 which are all distinct.
    However, vertex 0 has degree 0, vertex 1 has degree 1 (connected to 0), etc.
    Let's verify:
    Vertex i is connected to {0, 1, ..., i-1}.
    Degree of vertex i = i.
    Degrees are 0, 1, 2, ..., 14. All distinct.
    
    Number of edges = sum_{i=0}^{14} i = 14*15/2 = 105.
    Max possible = 15*14/2 = 105.
    Wait, that's the complete graph! 
    In the complete graph, all vertices have degree 14. They are NOT distinct.
    
    So the threshold construction above is:
    Edge (u, v) exists if u < v.
    This is the complete graph. Degrees are all 14.
    
    Let's rethink. We need distinct degrees.
    A graph with distinct degrees on n vertices:
    The degrees must be a subset of {0, 1, ..., n-1}.
    Since there are n vertices, the degrees must be exactly {0, 1, ..., n-1} or some permutation.
    But a graph cannot have both a vertex of degree 0 and a vertex of degree n-1 (connected to everyone).
    So the set of degrees cannot be {0, 1, ..., n-1}.
    The maximum number of distinct degrees is n-1.
    So we can have degrees {1, 2, ..., n-1} or {0, 1, ..., n-2}, etc.
    
    Let's aim for degrees {1, 2, ..., 14} or {0, 1, ..., 13}.
    If we use degrees {1, 2, ..., 14}, the sum of degrees is 1+2+...+14 = 105.
    Number of edges = 105/2 = 52.5. Not an integer. Impossible.
    
    If we use degrees {0, 1, ..., 13}, sum = 91. Edges = 45.5. Impossible.
    
    We need a set of 15 distinct integers from {0, ..., 14} with even sum.
    Sum of 0..14 = 105 (odd).
    We must remove one number to make the sum even.
    Remove an odd number: 1, 3, 5, 7, 9, 11, 13.
    Or add a number? No, we have 15 vertices, so 15 degrees.
    
    Let's try degrees: {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, x}
    We need 15 distinct values.
    Let's try to remove 1 from {0..14}.
    Degrees: {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}. That's 14 values.
    We need 15 values.
    
    Actually, we don't need ALL degrees to be distinct. We just need that for any path s-u-v-t, deg(s) != deg(t).
    If ALL degrees are distinct, this is trivially satisfied.
    So we want a graph with 15 distinct degrees.
    
    Possible degree sequences (sum must be even):
    {0, 1, 2, ..., 13, 15}? No, max degree is 14.
    
    Let's pick a set of 15 distinct integers from [0, 14] with even sum.
    Total sum of 0..14 is 105.
    We need to choose 15 numbers. But there are only 15 numbers in 0..14.
    So the only set is {0, 1, ..., 14}, which has odd sum.
    Therefore, it is IMPOSSIBLE to have 15 distinct degrees on 15 vertices!
    
    Why? Because the degrees must be distinct integers in [0, 14].
    There are exactly 15 such integers.
    Their sum is 105, which is odd.
    The sum of degrees must be even (2 * |E|).
    Contradiction.
    
    So we cannot have all distinct degrees.
    We must have at least one pair of vertices with the same degree.
    Let the degrees be d_1, ..., d_15.
    At least two are equal.
    
    However, the constraint is: for any path s-u-v-t, deg(s) != deg(t).
    This means that if two vertices have the same degree, they cannot be at distance 3 from each other.
    
    So the strategy is:
    1. Group vertices by degree.
    2. Ensure that vertices in the same degree group are not at distance 3.
    
    This is hard to construct directly.
    
    Alternative: Start with a graph that has many distinct degrees, and only a few collisions.
    
    Let's try a different approach.
    The seed program finds a graph with 3.1429 * seed_edges.
    The seed likely has around 10-20 edges.
    
    Let's stick to the search but make it much faster and smarter.
    """
    pass

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_g = None
    best_size = -1
    
    # Try a deterministic construction first:
    # We want to maximize edges.
    # Let's try to build a graph with degrees as distinct as possible.
    
    # Idea: Use a "near-threshold" graph.
    # Vertex i connects to {0, ..., i-1} except for some adjustments.
    
    # Let's just run the search multiple times with different seeds and keep the best.
    # But we need to make the search faster.
    
    # Optimization: Pre-compute all edges.
    # Use a faster validity check.
    
    base_seed = 12345
    restart_count = 0
    
    while time.time() - start_time < deadline:
        rng = random.Random(base_seed + 1000 * restart_count)
        
        g = _solve_once(rng)
        
        if g:
            adj, deg = _build_adj_deg(g)
            if _is_valid(adj, deg, g):
                if len(g) > best_size:
                    best_size = len(g)
                    best_g = g
        
        restart_count += 1
        
        if restart_count > 100:
            break

    if best_g is None:
        best_g = {(0, i) for i in range(1, _N)}
        
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])
        
    return result
# EVOLVE-BLOCK-END
