"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Fallback: 10 disjoint triangles
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]

    def is_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    # Count common neighbors
                    common = np.sum(adj[u] & adj[v])
                    if common > 1:
                        return False
        return True

    def has_independent_set(adj, k):
        # Greedy heuristic for independent set
        # If we can find k non-adjacent vertices, return True
        # We use a simple randomized greedy to check for existence
        # If it finds one, it's valid. If it doesn't, it might still exist.
        # For safety, we only add edges if we are confident alpha < k?
        # Actually, we need to ensure NO independent set of size 11 exists.
        # Checking alpha < 11 is hard. 
        # However, if we construct a graph where the complement has no clique of size 11, 
        # i.e., alpha(G) < 11.
        
        # A faster necessary condition: 
        # If the graph is not C4-free, reject.
        # For alpha, we can use a greedy packing. If greedy finds 11, reject.
        # If greedy fails to find 11, it's a candidate.
        
        # Let's use a randomized greedy to try to find an independent set of size k.
        # If we find one, alpha >= k (bad).
        # If we don't, it's likely alpha < k, but not guaranteed.
        # To be safe, we should verify? 
        # For n<=43, checking alpha < 11 exactly is exponential but maybe feasible with pruning?
        # Or we can rely on the structure. 
        # The known 39-vertex graph is specific. 
        # Randomized search might find a valid one if we verify.
        
        # Let's try to verify alpha < 11 using a simple backtracking for small n?
        # n=43 is too big for naive backtracking.
        
        # Alternative: 
        # If we find a graph that is C4-free and has high minimum degree, alpha is bounded.
        # alpha <= n * (1 - d_min/n) ? No, that's not tight.
        # Caro-Wei bound: alpha >= sum(1/(d_i+1)).
        # If sum(1/(d_i+1)) < 11, then alpha might be < 11? No, it's a lower bound.
        
        # Let's use a randomized greedy to find an independent set.
        # If we find one of size 11, return True (has independent set).
        # If not, return False (assume no independent set).
        # This is unsound (false negatives possible), but we can verify the final candidate.
        
        # Actually, let's just do a randomized greedy construction and then verify the candidate.
        # Verification of "no independent set of size 11" is the hard part.
        
        # Heuristic: 
        # Try to find an independent set of size 11.
        # If found, invalid.
        # If not found after many tries, likely valid.
        
        # Let's implement a quick check: 
        # Randomly shuffle vertices, greedily pick non-adjacent ones.
        # If we can pick 11, return True.
        
        verts = list(range(len(adj)))
        random.shuffle(verts)
        independent = []
        for v in verts:
            if all(not adj[v][u] for u in independent):
                independent.append(v)
                if len(independent) >= k:
                    return True
        return False

    def get_edges_from_adj(adj):
        n = len(adj)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # Randomized Search
    # Start with n=39 (best known) or higher?
    # Let's try to construct a C4-free graph with n=39 and alpha < 11.
    # We can start with a random graph and add/remove edges.
    
    for n in range(43, 38, -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph of size n
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random edges? Or start empty and add?
            # C4-free constraint is local.
            # Let's start with a random regular-ish graph?
            # Or start with disjoint cliques and merge?
            
            # Strategy: Start with a random graph G(n, p).
            # p should be such that expected common neighbors is low.
            # E[common neighbors of u,v] = (n-2) * p^2.
            # We want this < 1 to avoid C4.
            # p ~ 1/sqrt(n).
            p = 1.0 / np.sqrt(n)
            
            # Generate random graph
            adj = np.zeros((n, n), dtype=bool)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u][v] = True
                        adj[v][u] = True
                        
            # Remove C4s: 
            # While there is a C4, remove a random edge from it.
            # This is slow.
            
            # Better: Greedy construction.
            # Add vertices one by one.
            # For each new vertex v, connect to existing vertices u such that
            # adding (v,u) doesn't create a C4.
            # Condition: For any w < u, if adj[u][w] is True, then adj[v][w] must be False.
            # i.e., N(v) and N(u) must be disjoint for all u in N(v)?
            # No, if u and v are connected, they can share at most 1 neighbor.
            # So if we connect v to u, we must ensure that for all w < u, 
            # adj[v][w] is not True if adj[u][w] is True.
            # Actually, if we build incrementally, N(v) is built.
            # If we add edge (v, u), we must check that |N(v) \cap N(u)| < 1?
            # No, |N(v) \cap N(u)| must be 0 if we want to be safe?
            # Wait, C4 is u-v-w-x-u.
            # If u and v are connected, and they share a neighbor w, that's a triangle, not C4.
            # C4 requires u-v-w-x-u.
            # So u and v can share neighbors, as long as those neighbors don't form a cycle of length 4.
            # Specifically, if u and v share neighbor w, then w is connected to u and v.
            # If there is another vertex x connected to u and v, then u-v-w-x-u is a C4?
            # No, u-v-w-x-u implies w and x are distinct.
            # So if u and v share TWO neighbors w and x, then u-w-v-x-u is a C4.
            # So condition: No two vertices can have 2 common neighbors.
            
            # So, when adding v, and considering connecting to u:
            # Let C = N(v) \cap N(u).
            # If |C| >= 1, it's okay as long as |C| < 2?
            # No, if |C| >= 2, we have a C4.
            # So we must ensure |N(v) \cap N(u)| < 2 for all u in N(v).
            
            # Let's try this greedy approach.
            adj = np.zeros((n, n), dtype=bool)
            for v in range(n):
                # Determine neighbors of v among 0..v-1
                # We can connect v to u if adding (v,u) doesn't create a C4.
                # Adding (v,u) creates a C4 if there exists w < u such that
                # adj[u][w] and adj[v][w] are both True.
                # So for each candidate u, we check if there is any w in N(v) \cap N(u).
                # If yes, we cannot connect v to u?
                # Wait, if we connect v to u, and they share w, that's a triangle.
                # The C4 is formed if they share TWO neighbors.
                # So we must ensure that for any u we connect to, 
                # the number of common neighbors with v is at most 1.
                
                # Let's collect candidates.
                candidates = []
                for u in range(v):
                    # Check if connecting v to u is safe
                    # Safe if |N(v) \cap N(u)| < 2
                    # N(v) is partially built.
                    # Common neighbors are w < u such that adj[v][w] and adj[u][w] are True.
                    common = 0
                    for w in range(u):
                        if adj[v][w] and adj[u][w]:
                            common += 1
                            if common >= 2:
                                break
                    if common < 2:
                        candidates.append(u)
                
                # Now, we have a list of candidates u that can be connected to v.
                # But connecting v to u1 and u2 might create a C4 involving u1 and u2?
                # If we connect v to u1 and u2, do u1 and u2 share a neighbor w?
                # If u1 and u2 share w, and v is connected to both, 
                # then v-u1-w-u2-v is a C4?
                # Yes, if u1 and u2 are not connected, and share w.
                # But u1 and u2 are both < v.
                # So we need to ensure that the set of neighbors we choose for v
                # does not contain two vertices u1, u2 that share a neighbor w in N(v)?
                # No, the C4 is v-u1-w-u2-v.
                # This requires u1 and u2 to share a neighbor w.
                # And w must be connected to v? No, w is just a neighbor of u1 and u2.
                # Wait, the cycle is v-u1-w-u2-v.
                # Edges: (v,u1), (u1,w), (w,u2), (u2,v).
                # So w is a neighbor of u1 and u2.
                # w does NOT need to be a neighbor of v.
                # So if we choose u1 and u2 as neighbors of v, and u1 and u2 share a neighbor w,
                # then we have a C4.
                
                # So the constraint is: 
                # The set of neighbors of v must be an independent set in the graph 
                # induced by "sharing a neighbor"?
                # Or simpler: 
                # For any two neighbors u1, u2 of v, they must not share a common neighbor w.
                # i.e., |N(u1) \cap N(u2)| == 0?
                # No, if they share w, and w is not v, then v-u1-w-u2-v is a C4.
                # So yes, neighbors of v must be pairwise non-adjacent in the "co-neighborhood" graph?
                # Actually, it means N(u1) and N(u2) must be disjoint (excluding v, which is not in N(u1) yet).
                # So N(u1) \cap N(u2) must be empty.
                
                # This is a very strong constraint. 
                # It implies that the graph is a "linear" graph in some sense?
                # Or that the neighborhoods are disjoint.
                
                # Let's just try to pick a subset of candidates that are pairwise "compatible".
                # Compatibility: u1 and u2 are compatible if they do not share a common neighbor.
                
                # To keep it simple, let's just pick one random candidate?
                # Or pick a few that are far apart?
                
                # Let's try picking a random subset of candidates, ensuring pairwise compatibility.
                # This is like finding an independent set in the conflict graph of candidates.
                
                # For now, let's just pick a random candidate to connect to, if any.
                # This will result in a sparse graph.
                # To maximize n, we need to be able to add vertices.
                # If we are too sparse, we might not reach n=39.
                
                # Let's try to connect to as many as possible.
                # Greedy: iterate candidates, add if compatible with all currently added.
                
                new_neighbors = []
                for u in candidates:
                    compatible = True
                    for u2 in new_neighbors:
                        # Check if u and u2 share a common neighbor
                        common = 0
                        for w in range(u2):
                            if adj[u][w] and adj[u2][w]:
                                common += 1
                                if common >= 1:
                                    compatible = False
                                    break
                        if not compatible:
                            break
                    if compatible:
                        new_neighbors.append(u)
                        adj[v][u] = True
                        adj[u][v] = True
                
                # If we couldn't add any neighbors, maybe we should skip this vertex?
                # But we need n vertices.
                # If we skip, we reduce n.
                # Let's just leave it isolated? 
                # An isolated vertex increases alpha by 1.
                # We need alpha < 11.
                # If we have many isolated vertices, alpha will be large.
                # So we should try to connect them.
                
                # If new_neighbors is empty, we might have a problem.
                # Let's try to connect to ANY candidate, even if it creates a C4?
                # No, C4 is forbidden.
                
                # If we can't connect, we have an isolated vertex.
                # This is bad for alpha.
                
                # Let's verify the graph at the end.
            
            # Verify C4-free
            if is_c4_free(adj):
                # Verify alpha < 11
                # Use randomized greedy to try to find independent set of size 11
                has_ind_set = False
                for _ in range(100):
                    if has_independent_set(adj, 11):
                        has_ind_set = True
                        break
                if not has_ind_set:
                    # Candidate found
                    edges = get_edges_from_adj(adj)
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    break # Move to smaller n? No, we want max n. 
                    # But if we found n=43, we are done.
                    # If we found n=39, we might try n=43 again?
                    # Let's just break and return if n==43.
                    if n == 43:
                        return {'n': 43, 'edges': edges}
                    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
