"""Construct a K4-free graph with no independent 6-set, maximizing n."""
import time
import random

# EVOLVE-BLOCK-START
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _find_k4(adj, n):
    """Return a K4 as a tuple of 4 vertices, or None."""
    # Precompute common neighbors for pairs
    # For speed, iterate over edges and check if they share a common neighbor
    # that is also connected to each other (forming triangle with the edge)
    # Then check if any two vertices in a triangle form a K4 with a 4th.
    
    # Strategy: find all triangles, then for each triangle check if any vertex
    # is connected to all three (making a K4)
    
    # For each edge (u,v), find common neighbors w where adj[u]&adj[v] has w
    # For each such w, we have triangle (u,v,w)
    # Check if any x is connected to u, v, w -> K4
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            # For each w in common, check if there's x connected to all of u,v,w
            # x must be in common (connected to u and v) and connected to w
            # So x in common & adj[w]
            for w_bit in range(n):
                if (common >> w_bit) & 1:
                    w = w_bit
                    # x must be connected to u, v, and w
                    # x in adj[u] & adj[v] & adj[w]
                    # But x != u,v,w
                    candidates = adj[u] & adj[v] & adj[w]
                    # Remove u, v, w from candidates
                    candidates &= ~(1 << u)
                    candidates &= ~(1 << v)
                    candidates &= ~(1 << w)
                    if candidates:
                        x = candidates.bit_length() - 1
                        return (u, v, w, x)
    return None

def _find_indep6(adj, n):
    """Return an independent set of size 6, or None."""
    # Complement graph: edge in complement = no edge in original
    # Find K6 in complement = independent 6-set in original
    # Use Bron-Kerbosch with pivot for speed
    # Actually, let's use a simpler approach:
    # For each 5-set that is independent, check if we can extend to 6
    
    # Build complement adjacency
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
    
    # Find K6 in comp_adj using backtracking
    # Use a simple recursive search with pruning
    best = [None]
    
    def search(current, candidates):
        if len(current) == 6:
            best[0] = tuple(current)
            return True
        if not candidates:
            return False
        # Pruning: if len(current) + popcount(candidates) < 6, can't reach 6
        if len(current) + bin(candidates).count('1') < 6:
            return False
        for v_bit in range(n):
            if (candidates >> v_bit) & 1:
                v = v_bit
                new_candidates = candidates & comp_adj[v]
                # Only consider candidates that come after v in some ordering
                # to avoid duplicates - actually for finding ANY independent set,
                # we can just pick any
                current.append(v)
                if search(current, new_candidates):
                    return True
                current.pop()
        return False
    
    if search([], (1 << n) - 1):
        return best[0]
    return None

def _is_valid(adj, n):
    if _find_k4(adj, n) is not None:
        return False
    if _find_indep6(adj, n) is not None:
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

def _paley17():
    """Return adjacency list for Paley graph on 17 vertices."""
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj

def _construct_35():
    """
    Construct a 35-vertex (4,6)-Ramsey graph.
    
    Strategy: Use two copies of the Paley graph on 17 vertices (P1, P2),
    plus one additional vertex. Connect them carefully.
    
    Actually, a known construction for R(4,6) > 35 uses a specific graph.
    Let me try a different approach: use the known 35-vertex graph from
    the literature, or construct one via combining smaller Ramsey graphs.
    
    Known: R(3,3)=6, R(3,4)=9, R(3,5)=14, R(3,6)=18, R(4,4)=18, R(4,5)=25, R(4,6)=36
    
    So we need a graph on 35 vertices with no K4 and no independent 6-set.
    
    One approach: Take the complement of a graph with no independent 4-set and
    no clique of 6. But that's the same problem.
    
    Let me try: combine two 17-vertex Paley graphs with a bipartite connection.
    P1 = Paley(17), P2 = Paley(17), plus vertex 34.
    
    For the bipartite edges between P1 and P2, we need to ensure:
    - No K4: any 4 vertices forming a clique must not exist
    - No independent 6: any 6 vertices with no edges between them must not exist
    
    This is complex. Let me try a simpler heuristic approach:
    Start with a known valid small graph and grow it.
    """
    # Start with Paley 17
    adj17 = _paley17()
    
    # Try to add vertices one by one
    # New vertex v connected to a subset S of existing vertices
    # For no K4: for any edge (u1,u2) in S, u1 and u2 must not both be in S
    #   (otherwise {v, u1, u2, u3} could be K4 if u3 connects to all)
    #   Actually: v connected to S. K4 containing v: need 3 vertices in S that form
    #   a triangle in the induced subgraph on S, and all connected to v (they are).
    #   So S must be K4-free in the sense that the subgraph induced by S has no triangle
    #   that, together with v, forms K4. Wait: v is connected to all of S.
    #   So {v} ∪ T where T is a triangle in G[S] gives K4.
    #   So G[S] must be triangle-free.
    #
    # For no independent 6: the non-neighbors of v (call it N) plus v form an independent
    #   set if N is an independent set in G. So G[N] must have no independent 5-set
    #   (since {v} ∪ I where I is independent 5-set in N gives independent 6).
    
    # This is getting complex. Let me just try a random/local search approach
    # starting from a good seed.
    
    n_start = 17
    adj = list(adj17)
    
    # Try adding vertices
    rng = random.Random(42)
    best_n = n_start
    best_adj = list(adj)
    
    for target_n in range(n_start + 1, 36):
        if _time_left() < 5:
            break
        # Try to add one vertex
        added = False
        for attempt in range(200):
            if _time_left() < 2:
                break
            # Choose a random subset S of current vertices to connect to
            # S should be triangle-free in the induced subgraph
            # Start with empty S and greedily add vertices
            S = set()
            order = list(range(target_n - 1))
            rng.shuffle(order)
            for v in order:
                # Check if adding v to S keeps S triangle-free
                # v must not form a triangle with any two vertices in S
                ok = True
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        u1, u2 = S_list[i], S_list[j]
                        if (adj[u1] >> u2) & 1:  # u1-u2 edge exists
                            # Check if v connects to both
                            if (adj[v] >> u1) & 1 and (adj[v] >> u2) & 1:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    S.add(v)
            
            # Now check the independent set condition
            # Non-neighbors of new vertex: all vertices not in S
            # We need: no independent 5-set in the non-neighborhood
            N = set(range(target_n - 1)) - S
            
            # Check if G[N] has an independent set of size 5
            # This is expensive for large N. Use a quick check.
            # If |N| < 5, automatically OK
            if len(N) >= 5:
                # Check if there's an independent 5-set in N
                # Use the _find_indep6 logic but for size 5 in the subgraph
                # Quick heuristic: if the density of G[N] is high enough,
                # independent sets are small
                # For now, do a limited search
                n_list = list(N)
                # Build adjacency within N
                found_indep5 = False
                # Try all 5-subsets (if N is small enough)
                if len(N) <= 20:
                    from itertools import combinations
                    for combo in combinations(n_list, 5):
                        # Check if independent
                        indep = True
                        for i in range(5):
                            for j in range(i + 1, 5):
                                if (adj[combo[i]] >> combo[j]) & 1:
                                    indep = False
                                    break
                            if not indep:
                                break
                        if indep:
                            found_indep5 = True
                            break
                else:
                    # For larger N, use a probabilistic check
                    # Sample some 5-subsets
                    for _ in range(100):
                        combo = rng.sample(n_list, 5)
                        indep = True
                        for i in range(5):
                            for j in range(i + 1, 5):
                                if (adj[combo[i]] >> combo[j]) & 1:
                                    indep = False
                                    break
                            if not indep:
                                break
                        if indep:
                            found_indep5 = True
                            break
                
                if found_indep5:
                    continue
            
            # Valid! Add the vertex
            new_adj = list(adj)
            new_adj.append(0)
            for v in S:
                new_adj[v] |= 1 << (target_n - 1)
                new_adj[target_n - 1] |= 1 << v
            adj = new_adj
            added = True
            break
        
        if not added:
            break
        
        if target_n > best_n:
            best_n = target_n
            best_adj = list(adj)
    
    return best_n, best_adj

def _local_search(adj, n, max_time=10):
    """Try to improve the graph by flipping edges."""
    start = time.time()
    best_adj = list(adj)
    best_n = n
    
    while time.time() - start < max_time:
        improved = False
        # Try flipping random edges
        for _ in range(500):
            if time.time() - start > max_time:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            # Flip edge (u,v)
            has_edge = (adj[u] >> v) & 1
            new_adj = list(adj)
            if has_edge:
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                new_adj[u] |= 1 << v
                new_adj[v] |= 1 << u
            
            if _is_valid(new_adj, n):
                adj = new_adj
                improved = True
                break
        
        if not improved:
            break
    
    return adj, n

def construct():
    _START_TIME_GLOBAL = time.time()
    
    # First try the 35-vertex construction
    try:
        n, adj = _construct_35()
        if _is_valid(adj, n):
            edges = _edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
    except Exception:
        pass
    
    # Fallback: local search from Paley 17
    adj17 = _paley17()
    if _is_valid(adj17, 17):
        edges = _edges_from_adj(adj17, 17)
        return {'n': 17, 'edges': edges}
    
    # Last resort: return the seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
# EVOLVE-BLOCK-END
