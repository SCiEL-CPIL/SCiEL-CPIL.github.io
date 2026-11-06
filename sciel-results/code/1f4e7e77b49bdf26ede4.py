"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    def try_add_edge(adj, common, u, v):
        """
        Attempts to add edge (u, v) to the graph represented by adj and common.
        Returns True if added, False otherwise.
        adj: n x n boolean matrix
        common: n x n integer matrix, common[i][j] = number of common neighbors of i and j
        """
        # Check 1: Pair (u, v)
        # If we add (u, v), u and v become neighbors.
        # Do they have 3 common neighbors?
        # The common neighbors are N(u) intersect N(v).
        # Note: The edge (u, v) is not in the graph yet, so u is not in N(v) and v is not in N(u).
        # So common[u][v] currently holds |N(u) intersect N(v)|.
        if common[u][v] >= 3:
            return False
            
        # Check 2: Pairs (u, w) for all w in N(v)
        # If w is in N(v), then adding (u, v) makes v a neighbor of u.
        # Since w is already a neighbor of v, v becomes a common neighbor of u and w.
        # So common[u][w] increases by 1.
        # We need new common[u][w] < 3, so old common[u][w] must be < 2.
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                if common[u][w] >= 2:
                    return False
            
        # Check 3: Pairs (v, w) for all w in N(u)
        # Similarly, if w is in N(u), u becomes a common neighbor of v and w.
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                if common[v][w] >= 2:
                    return False
                    
        # If all checks pass, add the edge
        adj[u][v] = True
        adj[v][u] = True
        
        # Update common counts
        # For each neighbor w of u, common[w][v] increases by 1 (because u is now neighbor of v? No.)
        # Let's re-verify the update logic.
        # common[a][b] = |N(a) intersect N(b)|
        # We added edge (u, v).
        # N(u) gains v. N(v) gains u.
        
        # For any pair (x, y):
        # common[x][y] changes if u or v is in N(x) or N(y) and the other is added.
        
        # 1. Pairs involving u and some w.
        #    common[w][u] = |N(w) intersect N(u)|.
        #    N(u) gained v.
        #    So if v is in N(w), then v is in N(w) intersect N(u) (new).
        #    So common[w][u] increases by 1 if adj[w][v] is True.
        #    This matches the check we did: if adj[v][w] (which is adj[w][v]), we checked common[u][w].
        #    So for all w where adj[w][v] is True:
        #       common[u][w] += 1
        #       common[w][u] += 1
        
        # 2. Pairs involving v and some w.
        #    common[w][v] = |N(w) intersect N(v)|.
        #    N(v) gained u.
        #    So if u is in N(w), then u is in N(w) intersect N(v) (new).
        #    So common[w][v] increases by 1 if adj[w][u] is True.
        #    This matches the check: if adj[u][w] (which is adj[w][u]), we checked common[v][w].
        #    So for all w where adj[w][u] is True:
        #       common[v][w] += 1
        #       common[w][v] += 1
        
        # 3. Pair (u, v).
        #    common[u][v] = |N(u) intersect N(v)|.
        #    N(u) gained v, N(v) gained u.
        #    Does u become a common neighbor of u and v? No, u is not in N(u).
        #    Does v become a common neighbor of u and v? No, v is not in N(v).
        #    So common[u][v] does NOT change due to the edge (u,v) itself being added?
        #    Wait. The definition is common neighbors.
        #    If we add edge (u,v), do u and v gain common neighbors?
        #    The neighbors of u are N(u) U {v}.
        #    The neighbors of v are N(v) U {u}.
        #    Intersection: (N(u) U {v}) intersect (N(v) U {u})
        #    = (N(u) intersect N(v)) U (N(u) intersect {u}) U ({v} intersect N(v)) U ({v} intersect {u})
        #    N(u) does not contain u. N(v) does not contain v.
        #    So intersection is just N(u) intersect N(v).
        #    So common[u][v] remains unchanged.
        
        # So the update is:
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1
                
        return True

    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [[False]*n for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        edges = []
        
        # We can try to add edges in a random order.
        # To be more effective, we can prioritize vertices with lower degrees?
        # Or just random.
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if try_add_edge(adj, common, u, v):
                edges.append([u, v])
        
        return edges

    # Run multiple restarts
    num_restarts = 200
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        edges = greedy_build(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        # Optional: Local search improvement on the best found so far
        # If we have time, we can try to improve the current best by removing an edge and refilling
        # But simple greedy with many restarts is often strong enough for n=20.
        # Let's add a simple local search step if we find a good candidate.
        
    # If we found a good graph, return it.
    # Theoretical max for n=20 is around 60-70 edges? 
    # ex(20, K2,3) is likely around 60-65.
    # The disjoint K4s give 5*6 = 30 edges.
    # Score 1.9 implies 30 * 1.9 = 57 edges.
    # So we are already finding ~57 edges.
    # We want to push higher.
    
    if best_edges:
        return best_edges
    
    # Fallback
    edges = []
    for k in range(5):
        edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    return edges
# EVOLVE-BLOCK-END
