"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None

def _get_time_left(deadline):
    if _START_TIME is None:
        return deadline
    return deadline - (time.time() - _START_TIME)

def _is_k4_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_induced(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph size via branch and bound."""
    # A triangle-free induced subgraph is a set S where no three vertices in S form a triangle.
    # This is equivalent to: the induced subgraph on S has no K3.
    # We want the maximum size of such a set.
    
    best = 0
    deadline_time = start_time + deadline
    
    # Precompute for each vertex, the set of vertices that cannot be in the same triangle-free set
    # v and w conflict if there exists u such that adj[v][u] and adj[w][u] and adj[v][w]
    # Actually, v and w can both be in S as long as no u in S completes a triangle.
    # Better: for each pair (v,w) with adj[v][w], they conflict with any u adjacent to both.
    
    # For branch and bound:
    # State: (candidate_set, current_set)
    # A vertex v can be added to current_set if for all pairs (u1, u2) in current_set,
    # it's not the case that adj[v][u1] and adj[v][u2] and adj[u1][u2].
    
    # Precompute triangle_conflict[v] = set of vertices that cannot be in S with v
    # Actually, it's more subtle. Let me think again.
    
    # For a set S to be triangle-free, for every triple (a,b,c) in S, at least one edge is missing.
    # When adding v to S, we need to check: for every pair (u1, u2) in S, 
    # NOT (adj[v][u1] and adj[v][u2] and adj[u1][u2]).
    
    # Precompute: for each pair (u1, u2), is there an edge?
    # And for each v, which pairs (u1, u2) would form a triangle with v?
    
    # Let's use a different approach: 
    # For each vertex v, precompute the set of vertices that are "incompatible" with v in a triangle-free set.
    # v and w are incompatible if there exists u such that all three edges (v,u), (w,u), (v,w) exist.
    # But that's not quite right either, because u might not be in S.
    
    # Let me just do a straightforward branch and bound with good pruning.
    
    def dfs(available, current, current_size):
        nonlocal best
        if time.time() > deadline_time:
            return
        if current_size > best:
            best = current_size
        if not available:
            return
        
        # Upper bound: current_size + len(available)
        if current_size + len(available) <= best:
            return
        
        # Try each vertex in available
        for i in range(len(available)):
            v = available[i]
            # Check if v can be added to current
            can_add = True
            for j in range(len(current)):
                u1 = current[j]
                if not adj[v][u1]:
                    continue
                for k in range(j+1, len(current)):
                    u2 = current[k]
                    if adj[v][u2] and adj[u1][u2]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current = current + [v]
                new_available = available[:i] + available[i+1:]
                dfs(new_available, new_current, current_size + 1)
    
    dfs(list(range(n)), [], 0)
    return best

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 110  # seconds, leave margin for 120s limit
    n = 18
    
    best_edges = None
    best_score = 999
    
    # Try several structured constructions and random search
    
    def get_adjacency(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def edges_to_adj(edges, n):
        return get_adjacency(edges, n)
    
    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Construction 1: K6,6,6 (complete tripartite)
    # This has max triangle-free induced subgraph of size 12 (two parts)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    adj = get_adjacency(edges, n)
    if _is_k4_free(adj, n):
        mtf = _max_triangle_free_induced(adj, n, 5, _START_TIME)
        if mtf < best_score:
            best_score = mtf
            best_edges = edges[:]
    
    # Construction 2: Try to remove some edges from K6,6,6 to break triangles
    # In K6,6,6, any two vertices from different parts are adjacent.
    # A triangle requires vertices from all three parts (or two from one part and one from another, but there are no edges within a part).
    # Actually in K6,6,6, a triangle is formed by one vertex from each part.
    # To make it K4-free, we need to ensure no K4. In K6,6,6, any 4 vertices with at least one from each of 3 parts... 
    # Actually K6,6,6 contains K4: take 2 from part A, 1 from B, 1 from C? No, there are no edges within A.
    # A K4 in K6,6,6 would need 4 vertices where every pair is adjacent. Since vertices in the same part are not adjacent,
    # a K4 must have at most one vertex from each part. But there are only 3 parts, so no K4!
    # Wait, that means K6,6,6 is K4-free? Let me verify.
    # K4 requires 4 mutually adjacent vertices. In K6,6,6, two vertices in the same part are NOT adjacent.
    # So any set of 4 vertices must have at least two in the same part (by pigeonhole, since 3 parts),
    # and those two are not adjacent, so no K4. Yes, K6,6,6 is K4-free.
    
    # The max triangle-free induced subgraph in K6,6,6:
    # - One part: 6 vertices, no edges, triangle-free. Size 6.
    # - Two parts: 12 vertices, complete bipartite K6,6, triangle-free. Size 12.
    # - Three parts: 18 vertices, contains triangles (one from each part). Not triangle-free.
    # So max is 12.
    
    # Can we do better? We need to add edges within parts to break the bipartite structure,
    # but we must maintain K4-free.
    
    # Strategy: Start with K6,6,6 and add edges within parts, but carefully to avoid K4.
    # Adding an edge within a part creates triangles with vertices from the other two parts.
    # But it might break the triangle-free property of the two-part union.
    
    # Let's try: add a matching within each part.
    # Part A: add edges (0,3), (1,4), (2,5)
    # Part B: add edges (6,9), (7,10), (8,11)
    # Part C: add edges (12,15), (13,16), (14,17)
    
    edges2 = edges[:]  # Start with K6,6,6 edges
    # Add matching in part A
    for a, b in [(0,3), (1,4), (2,5)]:
        edges2.append([a, b])
    # Add matching in part B
    for a, b in [(6,9), (7,10), (8,11)]:
        edges2.append([a, b])
    # Add matching in part C
    for a, b in [(12,15), (13,16), (14,17)]:
        edges2.append([a, b])
    
    adj2 = get_adjacency(edges2, n)
    if _is_k4_free(adj2, n):
        mtf2 = _max_triangle_free_induced(adj2, n, 5, _START_TIME)
        if mtf2 < best_score:
            best_score = mtf2
            best_edges = edges2[:]
    
    # Construction 3: Random search
    # Try random K4-free graphs and evaluate
    
    def is_k4_free_fast(adj, n):
        """Fast K4-free check using triangle enumeration."""
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def random_k4_free_graph(n):
        """Generate a random K4-free graph by starting empty and adding edges."""
        adj = [[False]*n for _ in range(n)]
        # Start with a random tripartite structure
        random.shuffle(list(range(n)))
        perm = list(range(n))
        random.shuffle(perm)
        sizes = [6, 6, 6]
        parts = []
        idx = 0
        for s in sizes:
            parts.append(perm[idx:idx+s])
            idx += s
        
        # Add all edges between parts
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = True
                        adj[b][a] = True
        
        # Randomly add some edges within parts
        for part in parts:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if random.random() < 0.3:
                        adj[part[i]][part[j]] = True
                        adj[part[j]][part[i]] = True
        
        # Check K4-free, if not, remove random edges
        while not is_k4_free_fast(adj, n):
            # Find a K4 and remove a random edge from it
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4: a,b,c,d
                                # Remove a random edge
                                edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = random.choice(edges_in_k4)
                                adj[e[0]][e[1]] = False
                                adj[e[1]][e[0]] = False
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        return adj
    
    # Run random search
    time_budget = 60  # seconds for random search
    start_search = time.time()
    
    for iteration in range(10000):
        if time.time() - start_search > time_budget:
            break
        
        adj_rand = random_k4_free_graph(n)
        
        if not is_k4_free_fast(adj_rand, n):
            continue
        
        # Evaluate
        mtf_rand = _max_triangle_free_induced(adj_rand, n, 3, _START_TIME)
        if mtf_rand < best_score:
            best_score = mtf_rand
            best_edges = adj_to_edges(adj_rand, n)
            if best_score <= 10:
                break
    
    if best_edges is None:
        # Fallback to K6,6,6
        best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
