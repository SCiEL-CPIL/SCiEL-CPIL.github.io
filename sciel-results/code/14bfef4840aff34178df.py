import time
import numpy as np
from itertools import combinations

DEADLINE = 120.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

def _has_c6(adj, u, v, n):
    """
    Check if adding edge (u, v) to the current graph (given by adj matrix)
    would create a 6-cycle (C6).
    
    A 6-cycle containing edge (u, v) is: u - a - b - c - d - v - u
    where a, b, c, d are distinct vertices from {0..n-1} \\ {u, v}
    
    Equivalently, we need a path of length 4 from u to v in the graph
    without using edge (u,v) directly, but wait - we're adding (u,v), so
    the cycle is u-a-b-c-d-v-u. The path u to v of length 5 (4 internal nodes)
    must exist in the current graph (before adding u-v), OR we can think:
    after adding (u,v), there exists a path of length 5 from u to v not using (u,v).
    
    Actually, let's reframe: adding edge (u,v) creates a C6 iff there exists
    a path of length 5 from u to v in the graph BEFORE adding (u,v), with all
    6 vertices distinct.
    
    Path of length 5: u -> a -> b -> c -> d -> v
    All of u, a, b, c, d, v must be distinct.
    
    We can check this by doing a BFS/DFS from u to find if there's a path of 
    exactly length 5 to v, avoiding v until the end.
    
    More efficient approach: 
    For each neighbor a of u (a != v), for each neighbor b of a (b != u, b != v, b != a),
    for each neighbor c of b (c != a, c != v, c != b, c != u),
    for each neighbor d of c (d != b, d != v, d != c, d != u, d != a),
    check if d is adjacent to v.
    """
    # adj is n x n adjacency matrix (symmetric), adj[i][j] = 1 if edge exists
    # We need to find a path u-a-b-c-d-v where all are distinct
    # and edges (u,a), (a,b), (b,c), (c,d), (d,v) all exist in current graph
    
    neighbors_u = np.where(adj[u] == 1)[0]
    # a must be neighbor of u, a != v
    for a in neighbors_u:
        if a == v:
            continue
        neighbors_a = np.where(adj[a] == 1)[0]
        for b in neighbors_a:
            if b == u or b == v or b == a:
                continue
            neighbors_b = np.where(adj[b] == 1)[0]
            for c in neighbors_b:
                if c == u or c == v or c == a or c == b:
                    continue
                neighbors_c = np.where(adj[c] == 1)[0]
                for d in neighbors_c:
                    if d == u or d == v or d == a or d == b or d == c:
                        continue
                    if adj[d][v] == 1:
                        return True
    return False

def _edge_count(adj):
    return int(np.sum(adj) // 2)

def construct():
    """
    Construct a C6-free graph on 40 vertices maximizing edges.
    Uses greedy edge addition with C6 checking, and local search with
    multi-edge perturbations to escape local optima.
    """
    start_time = time.time()
    n = 40
    
    # Initialize with empty graph
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Greedy phase: add edges that don't create C6
    # Sort edges by some heuristic - maybe by sum of degrees of endpoints
    # or just randomize to avoid bias
    import random
    random.seed(42)
    
    edge_order = list(all_edges)
    random.shuffle(edge_order)
    
    # Greedy addition
    for u, v in edge_order:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        if adj[u][v] == 1:
            continue
        # Check if adding this edge creates a C6
        if not _has_c6(adj, u, v, n):
            adj[u][v] = 1
            adj[v][u] = 1
    
    best_adj = adj.copy()
    best_edges = _edge_count(best_adj)
    
    # Local search phase
    # Try to swap edges or remove/add edges to improve
    stuck_count = 0
    perturb_count = 0
    
    while time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        improved = False
        
        # Try adding any missing edge
        missing_edges = [(u, v) for u, v in all_edges if adj[u][v] == 0]
        random.shuffle(missing_edges)
        
        added_any = False
        for u, v in missing_edges:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            if not _has_c6(adj, u, v, n):
                adj[u][v] = 1
                adj[v][u] = 1
                added_any = True
                improved = True
        
        if improved:
            best_adj = adj.copy()
            best_edges = _edge_count(best_adj)
            stuck_count = 0
            continue
        
        # No single edge can be added. Try perturbations.
        stuck_count += 1
        perturb_count += 1
        
        # Multi-edge perturbation: remove 2-4 random edges, then try to greedily add back
        if stuck_count >= 3:
            current_edges = [(u, v) for u, v in all_edges if adj[u][v] == 1]
            if len(current_edges) < 2:
                continue
            
            num_remove = min(4, len(current_edges))
            num_remove = random.randint(2, num_remove)
            to_remove = random.sample(current_edges, num_remove)
            
            # Save state
            saved_adj = adj.copy()
            
            for u, v in to_remove:
                adj[u][v] = 0
                adj[v][u] = 0
            
            # Now try to greedily add edges back
            all_edges_again = list(all_edges)
            random.shuffle(all_edges_again)
            
            for u, v in all_edges_again:
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                if adj[u][v] == 1:
                    continue
                if not _has_c6(adj, u, v, n):
                    adj[u][v] = 1
                    adj[v][u] = 1
            
            new_edges = _edge_count(adj)
            if new_edges > best_edges:
                best_adj = adj.copy()
                best_edges = new_edges
                stuck_count = 0
            else:
                # Revert to saved state
                adj = saved_adj
                stuck_count += 1
        
        # If stuck too long, do a bigger perturbation
        if stuck_count >= 10:
            # Reset to best_adj and try different randomization
            adj = best_adj.copy()
            stuck_count = 0
            # Add some random edges if possible
            missing = [(u, v) for u, v in all_edges if adj[u][v] == 0]
            random.shuffle(missing)
            for u, v in missing[:10]:
                if not _has_c6(adj, u, v, n):
                    adj[u][v] = 1
                    adj[v][u] = 1
            new_edges = _edge_count(adj)
            if new_edges > best_edges:
                best_adj = adj.copy()
                best_edges = new_edges
    
    # Return edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j] == 1:
                edges.append([i, j])
    
    return edges
