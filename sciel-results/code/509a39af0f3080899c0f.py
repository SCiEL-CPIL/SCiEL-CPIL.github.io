import random
import itertools
import time

def construct():
    n = 20
    vertices = list(range(n))
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def can_add_edge(adj, u, v):
        N_u = [x for x in range(n) if adj[u][x] and x != v]
        N_v = [x for x in range(n) if adj[v][x] and x != u]
        
        if len(N_u) < 2 or len(N_v) < 2:
            return True
            
        for a1, a2 in itertools.combinations(N_v, 2):
            common_count = 0
            for b in N_u:
                if adj[a1][b] and adj[a2][b]:
                    common_count += 1
                    if common_count >= 2:
                        return False
        return True

    def run_greedy_construction():
        adj = [[False] * n for _ in range(n)]
        
        edges = all_edges.copy()
        random.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
                
        # Local Search: Iteratively try to add missing edges
        # We do multiple passes because adding one edge might open up space for another?
        # Actually, adding an edge *reduces* the space for future additions (makes constraints tighter).
        # So if an edge couldn't be added, it will never be addable later in the same pass 
        # unless we *remove* an edge. 
        # Wait, if we add (u,v), it might create a K3,3 that blocks (w,x).
        # But if (w,x) was blocked before, adding (u,v) won't unblock it.
        # So a single pass of "try to add all missing edges" is sufficient?
        # No, the order matters. If we try (w,x) first and fail, then add (u,v), (w,x) is still blocked.
        # If we tried (w,x) after (u,v), it would still be blocked.
        # So the set of addable edges is monotonic decreasing as we add edges.
        # Thus, if we iterate through all missing edges once, and add all that are currently addable,
        # we reach a local optimum where no single missing edge can be added.
        
        # However, the *initial* greedy phase already did this for the shuffled order.
        # The local search just ensures that any edge skipped in the initial phase (because it was 
        # blocked by edges added *after* it in the shuffle? No, it was blocked by edges present *at that time*.
        # Since edges only get added, the set of present edges grows. 
        # If (u,v) was blocked at step i, it is blocked at step j > i.
        # So the initial greedy pass *is* the local search?
        # No. In the initial pass, we iterate through the shuffled list.
        # If edge E is at position 10, and it is blocked by edge F at position 20, 
        # F is not present when we check E. So E might be added.
        # If E is at position 20 and F at position 10, F is present when we check E. E might be blocked.
        # So the result depends on order.
        # The "Local Search" step described above (iterate all missing) is redundant 
        # if we already iterated all edges in the greedy phase?
        # No. In the greedy phase, we only add if it fits. 
        # After the greedy phase, we have a graph G.
        # We check all edges not in G. Can any be added?
        # If an edge E was not in G, it was either:
        # 1. Not considered (impossible, we consider all).
        # 2. Considered and rejected.
        # If it was rejected, it was because of edges present at that time.
        # Since we only added edges, the set of blocking edges is a superset of the original blocking set.
        # So if it was blocked before, it is blocked now.
        # Therefore, the graph produced by the single greedy pass is already a "maximal" 
        # K3,3-free graph with respect to single-edge additions.
        
        # So the local search loop `while changed` will never change anything 
        # if we don't remove edges.
        # To improve, we need to REMOVE edges.
        
        count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        return count, adj

    def improve_with_swaps(adj):
        # Try to remove 1 edge and add 2 edges.
        # This is complex. 
        # Alternative: Random Restart is the key.
        pass

    best_count = 0
    best_adj = None
    start_time = time.time()
    
    while time.time() - start_time < 100:
        count, adj = run_greedy_construction()
        if count > best_count:
            best_count = count
            best_adj = adj
            
    if best_adj is None:
        return []
        
    edges_list = [[u, v] for u in range(n) for v in range(u+1, n) if best_adj[u][v]]
    return edges_list
