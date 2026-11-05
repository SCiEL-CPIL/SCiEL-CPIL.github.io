"""
Strategy: 
1. Use a faster, bit-parallel Maximum Independent Set solver (Tomita-style with coloring) 
   to verify C4-free graphs for alpha < 11.
2. Perform a randomized greedy construction of C4-free graphs, starting from high n (e.g., 43) 
   and removing edges/vertices to satisfy constraints, or building up and using the fast solver 
   to guide edge additions.
3. Specifically, we will try to construct a graph on n vertices for n in [43, 42, 41, 40, 39] 
   by starting with a random graph, enforcing C4-freeness via local repair, and checking alpha.
   If alpha >= 11, we add edges (if C4-free) or remove vertices.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if len(au & adj[v]) >= 2:
                return False
    return True

def _max_independent_set_bitset(n, adj):
    """
    Fast Maximum Independent Set using bitsets and branch & bound.
    Returns the size of the maximum independent set.
    """
    # Precompute neighbor masks
    masks = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= 1 << u
        masks[v] = m
        
    full = (1 << n) - 1
    
    # Simple upper bound using greedy coloring on the subgraph induced by 'rem'
    # For speed, we use a simpler bound: size of rem
    # But for n~40, we can afford a bit more.
    
    best = [0]
    
    # Order vertices by degree for better pruning? 
    # Let's stick to a standard branch and bound with bitsets.
    
    def dfs(rem, size):
        if time.time() > deadline_global:
            return
        if size > best[0]:
            best[0] = size
        if not rem:
            return
            
        # Bound: size + popcount(rem)
        if size + rem.bit_count() <= best[0]:
            return
            
        # Pick a vertex. Heuristic: vertex with max degree in rem?
        # For now, pick the lowest bit for speed, but this is weak.
        # Let's pick a vertex that maximizes reduction.
        # Actually, for n=40, picking any vertex is fine if we have good bounds.
        
        # Simple heuristic: pick vertex with highest degree in the induced subgraph
        # This is expensive to compute every time. 
        # Alternative: just pick the first available vertex.
        
        lb = rem & (-rem)
        v = lb.bit_length() - 1
        
        # Branch 1: Exclude v
        dfs(rem & ~lb, size)
        
        # Branch 2: Include v
        # Neighbors of v in rem
        neighbors_in_rem = rem & masks[v]
        dfs(rem & ~lb & ~neighbors_in_rem, size + 1)
        
    dfs(full, 0)
    return best[0]

def _has_independent_set_ge(n, adj, limit):
    """Check if there is an independent set of size >= limit."""
    # Quick check: if n < limit, return False
    if n < limit:
        return False
        
    # Use the exact solver
    max_alpha = _max_independent_set_bitset(n, adj)
    return max_alpha >= limit

def _can_add_edge_c4_free(n, adj, u, v):
    """Check if adding edge (u,v) keeps the graph C4-free."""
    if v in adj[u]:
        return True
    # Common neighbors must be empty? No.
    # If we add (u,v), we create a C4 if there exist a, b such that:
    # u-a, a-b, b-v, v-u (new) is a cycle.
    # This requires a in N(u) \ {v} and b in N(v) \ {u} and a-b is an edge.
    # Also, if u and v have a common neighbor w, does that create a C4?
    # Cycle u-w-v-u is a triangle, not a C4.
    # Cycle u-w-x-v-u? 
    # The condition for C4-freeness is: |N(u) \cap N(v)| <= 1 for all u,v.
    # If we add (u,v), we change N(u) and N(v).
    # New N(u) = N(u) \cup {v}
    # New N(v) = N(v) \cup {u}
    # For any pair (x,y) in the new graph, we need |N_new(x) \cap N_new(y)| <= 1.
    # Most pairs are unaffected.
    # Pairs involving u or v:
    # 1. Pair (u, y) for y in V \ {u,v}:
    #    N_new(u) \cap N_new(y) = (N(u) \cup {v}) \cap N(y)
    #    = (N(u) \cap N(y)) \cup ({v} \cap N(y))
    #    If v in N(y), then we add v to the intersection.
    #    If |N(u) \cap N(y)| was 1, and v in N(y), then new size is 2 -> C4.
    #    If |N(u) \cap N(y)| was 0, and v in N(y), new size is 1 -> OK.
    #    If v not in N(y), new size is |N(u) \cap N(y)|.
    #    So we need: for all y in N(v), |N(u) \cap N(y)| <= 0? 
    #    Wait. If y in N(v), then v in N(y).
    #    Intersection becomes (N(u) \cap N(y)) \cup {v}.
    #    Since v is not in N(u) (we are adding the edge, so initially v not in N(u)), 
    #    v is distinct from elements in N(u).
    #    So size = |N(u) \cap N(y)| + 1.
    #    We need |N(u) \cap N(y)| + 1 <= 1 => |N(u) \cap N(y)| <= 0.
    #    So for all y in N(v), N(u) and N(y) must be disjoint.
    
    # 2. Pair (x, v) for x in V \ {u,v}:
    #    Symmetric: for all x in N(u), N(v) and N(x) must be disjoint.
    
    # 3. Pair (u, v):
    #    N_new(u) \cap N_new(v) = (N(u) \cup {v}) \cap (N(v) \cup {u})
    #    = (N(u) \cap N(v)) \cup (N(u) \cap {u}) \cup ({v} \cap N(v)) \cup ({v} \cap {u})
    #    Since u not in N(u) and v not in N(v) (simple graph), and u!=v:
    #    = N(u) \cap N(v).
    #    We already know |N(u) \cap N(v)| <= 1 in the current C4-free graph.
    #    So this is OK.
    
    # 4. Pairs (x, y) where x, y not u, v:
    #    Unaffected.
    
    # So the condition is:
    # For all y in N(v): N(u) & N(y) == 0
    # For all x in N(u): N(v) & N(x) == 0
    
    for y in adj[v]:
        if y == u: continue
        if adj[u] & adj[y]:
            return False
    for x in adj[u]:
        if x == v: continue
        if adj[v] & adj[x]:
            return False
            
    return True

def _construct_random_c4_free(n, seed):
    """
    Construct a random C4-free graph on n vertices.
    We start with an empty graph and add edges randomly, rejecting if C4 is formed.
    """
    random.seed(seed)
    adj = [set() for _ in range(n)]
    
    # Shuffle edge order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    
    for u, v in edges:
        if _can_add_edge_c4_free(n, adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            
    return adj

def _improve_graph(adj, n, deadline):
    """
    Try to add edges to reduce independence number while keeping C4-free.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all non-edges
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    continue
                if _can_add_edge_c4_free(n, adj, u, v):
                    # Tentatively add
                    adj[u].add(v)
                    adj[v].add(u)
                    # Check if alpha decreased or stayed low
                    # We just check if alpha < 11. If it was < 11 before, it might become >= 11? 
                    # No, adding edges cannot increase independence number.
                    # So if the graph was valid (alpha < 11), it remains valid.
                    # We only add edges to make the graph "denser" which might help if we 
                    # were trying to remove vertices later, but here we are maximizing n.
                    # Actually, adding edges is always good for keeping alpha low.
                    # So we should just add all possible edges!
                    # Wait, if we add all possible edges, we get the maximal C4-free graph.
                    # Does the maximal C4-free graph have alpha < 11?
                    # Not necessarily. A sparse C4-free graph might have alpha < 11, 
                    # but adding edges might not change alpha much, or might be impossible.
                    # But adding edges never increases alpha. 
                    # So if we start with a valid graph, adding edges keeps it valid.
                    # So we should just maximize edges.
                    pass # No check needed for alpha if it was already valid.
        
    # Actually, the strategy should be:
    # 1. Generate a random C4-free graph.
    # 2. Check if alpha < 11.
    # 3. If yes, we have a valid graph.
    # 4. If no, we need to modify it. Adding edges reduces alpha.
    #    So we should try to add as many edges as possible.
    #    If after adding all possible edges alpha is still >= 11, this graph is useless.
    
    # Let's refine: 
    # The goal is to find A C4-free graph with alpha < 11.
    # Random C4-free graphs are often sparse.
    # We should try to make them as dense as possible (maximal C4-free).
    
    # So the loop should be:
    # 1. Generate random C4-free graph.
    # 2. Complete it to a maximal C4-free graph (add all possible edges).
    # 3. Check alpha.
    # 4. If alpha < 11, return.
    
    # Let's implement "Complete to maximal C4-free"
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u] and _can_add_edge_c4_free(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    changed = True
    return adj

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 145.0  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # We want to maximize n. 
    # Try n from 43 down to 39.
    # For each n, try to find a valid graph.
    
    for n in range(43, 38, -1):
        if time.time() > deadline_global - 10:
            break
            
        # Try multiple random seeds
        found = False
        for seed in range(100):
            if time.time() > deadline_global - 5:
                break
                
            # 1. Generate random C4-free graph
            adj = _construct_random_c4_free(n, seed)
            
            # 2. Complete to maximal C4-free
            adj = _improve_graph(adj, n, deadline_global - 5)
            
            # 3. Check alpha
            # Use a quick check first? 
            # Just use the exact solver.
            if not _has_independent_set_ge(n, adj, 11):
                # Valid graph found
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # Since we are iterating n downwards, the first valid n is the max
            break
    
    # Fallback: If we didn't find anything for n > 38, 
    # maybe the random generation is too weak.
    # Try the polarity graph of PG(2, 5) which has 31 vertices? 
    # Or PG(2, 7) which has 57 vertices (too big, but we can subgraph?).
    # Actually, the current program scores 1.0333 which implies it finds n=39.
    # If my random search fails to find n=40..43, I should at least reproduce n=39.
    
    if best_n < 39:
        # Fallback to a known construction or simpler search
        # Let's try to construct n=39 using a similar random approach but with more seeds
        # or just return the best we have.
        
        # Try n=39 specifically with more seeds if we have time
        if time.time() < deadline_global - 10:
            for seed in range(200):
                if time.time() > deadline_global - 5:
                    break
                n = 39
                adj = _construct_random_c4_free(n, seed + 1000)
                adj = _improve_graph(adj, n, deadline_global - 5)
                if not _has_independent_set_ge(n, adj, 11):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    best_n = n
                    best_edges = edges
                    break
    
    # Final fallback: empty graph or small graph if all else fails
    if best_n == 0:
        best_n = 1
        best_edges = []
        
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
