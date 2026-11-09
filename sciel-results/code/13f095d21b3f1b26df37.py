"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def construct():
    n = P['n']
    # Initialize graph
    # adj[u] is a set of neighbors
    adj = [set() for _ in range(n)]
    edges = []
    
    # Precompute all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Shuffle to break symmetry, maybe use a fixed seed for reproducibility in evolution?
    # Or just iterate in a specific order.
    # Let's try a greedy approach: try to add edges.
    # Order: maybe try edges that connect high-degree vertices?
    # Initially all degrees are 0.
    # Let's just iterate through all edges and add if valid.
    # To maximize edges, we should probably prioritize edges that are "easy" to add.
    
    # Helper: Check if adding edge (u,v) is valid
    def is_valid_add(u, v):
        # 1. Calculate new degrees
        # Current degrees
        deg = [len(adj[i]) for i in range(n)]
        new_deg_u = deg[u] + 1
        new_deg_v = deg[v] + 1
        
        # 2. Check existing paths affected by degree change of u or v
        # Paths are s-a-b-t.
        # If s is u: u-a-b-t. Need new_deg_u != deg[t] (if t not v) or new_deg_v (if t is v? No, t is endpoint)
        # Wait, if t is v, then deg[t] becomes new_deg_v.
        # So we need to check D_new(s) != D_new(t) for all valid paths s-a-b-t.
        
        # Let's iterate over all possible paths of length 3 in the CURRENT graph.
        # If the path doesn't involve u or v as an endpoint, its validity is unchanged.
        # If it involves u or v as an endpoint, we re-check with new degrees.
        # If the path involves u or v as an internal node, it's not affected by degree change of u/v?
        # No, the constraint is on ENDPOINTS s and t.
        # So only paths where s or t is u or v are affected by degree changes.
        
        # Iterate over all a, b such that (a,b) is an edge?
        # Path s-a-b-t requires edges (s,a), (a,b), (b,t).
        # We can iterate over middle edge (a,b).
        for a in range(n):
            for b in adj[a]:
                if a > b: continue # Each edge once
                # For each neighbor s of a (s != b)
                # For each neighbor t of b (t != a)
                # If s == t, skip (simple path)
                for s in adj[a]:
                    if s == b: continue
                    for t in adj[b]:
                        if t == a: continue
                        if s == t: continue
                        
                        # This is a path s-a-b-t in the current graph.
                        # Determine new degrees of s and t
                        ds = new_deg_u if s == u else (new_deg_v if s == v else deg[s])
                        dt = new_deg_u if t == u else (new_deg_v if t == v else deg[t])
                        
                        if ds == dt:
                            return False
                            
        # 3. Check new paths created by adding (u,v)
        # Path s-u-v-t
        for s in adj[u]:
            if s == v: continue
            for t in adj[v]:
                if t == u: continue
                if s == t: continue
                ds = new_deg_u if s == u else (new_deg_v if s == v else deg[s])
                # Wait, s is neighbor of u, so s != u.
                # t is neighbor of v, so t != v.
                # But s could be v? No, s in adj[u] and s != v.
                # t could be u? No, t in adj[v] and t != u.
                # However, s could be u? No.
                # So ds is just deg[s] UNLESS s is v? No s!=v.
                # Wait, what if s is v? We skipped s==v.
                # What if t is u? We skipped t==u.
                # So for new paths, s is not u or v, and t is not u or v?
                # No, s is neighbor of u. s can be any neighbor except v.
                # t is neighbor of v. t can be any neighbor except u.
                # Do s or t change degree?
                # s is an endpoint. Its degree changes ONLY if s is u or v.
                # But s != v (skipped) and s != u (neighbor of u).
                # So s is NOT u or v. Thus ds = deg[s].
                # Similarly, t is NOT u or v. Thus dt = deg[t].
                
                # Wait, is this correct?
                # If s is a neighbor of u, s is distinct from u.
                # We explicitly check s != v.
                # So s is distinct from u and v.
                # So deg[s] does not change.
                # Same for t.
                
                if deg[s] == deg[t]:
                    return False
        
        # Also path s-v-u-t
        for s in adj[v]:
            if s == u: continue
            for t in adj[u]:
                if t == v: continue
                if s == t: continue
                # s distinct from u, v. t distinct from u, v.
                if deg[s] == deg[t]:
                    return False

        return True

    # Greedy Construction
    # We want to maximize edges.
    # Try adding all possible edges.
    # To be deterministic and potentially find a dense graph, we can try multiple orders or just one.
    # Let's try adding edges in a random order, or sorted order.
    # Given the constraint is tight, order matters.
    # Let's try a few random restarts within the time limit.
    
    best_edges = 0
    best_edge_list = []
    
    start_time = time.time()
    deadline = start_time + 50 # 50s margin
    
    # Try a few random permutations of edge additions
    import random
    for _ in range(100): # Max 100 restarts
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        for u, v in all_edges:
            if is_valid_add(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_edge_list = edges[:]
            
        # If we reached a high number, we might stop early?
        # Max edges is 105.
        # If best_edges > 20, it's a good start.
        
    # Local Search / Improvement
    # Try to add edges that were skipped?
    # Or swap?
    # Given the time, a simple "try to add any missing edge" loop on the best solution found.
    
    if best_edge_list:
        adj = [set() for _ in range(n)]
        for u, v in best_edge_list:
            adj[u].add(v)
            adj[v].add(u)
            
        # Try to add any edge not in the list
        current_edge_set = set(map(tuple, best_edge_list))
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if (u, v) in current_edge_set or (v, u) in current_edge_set:
                    continue
                if is_valid_add(u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    best_edge_list.append([u, v])
                    current_edge_set.add((u, v))
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break

    # Convert to list of lists of ints
    # Ensure types are Python int
    final_edges = [[int(u), int(v)] for u, v in best_edge_list]
    
    return final_edges

# EVOLVE-BLOCK-END
