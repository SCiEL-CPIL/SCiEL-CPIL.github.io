import random
import time
import sys

def has_c4(adj, u, v):
    """Check if adding edge (u,v) creates a C4."""
    # A C4 is u-x-v-y-u.
    # This happens if there is a common neighbor x of u and v, and a common neighbor y of u and v?
    # No, if we add (u,v), a C4 is formed if there exists x, y such that
    # x is neighbor of u and v? No.
    # If we add (u,v), we look for paths of length 3 between u and v in the current graph.
    # Path u-x-y-v. If such a path exists, adding (u,v) closes the C4.
    # So we need to check if there are two distinct common neighbors of u and v?
    # No, if x is a common neighbor of u and v, then u-x-v is a path of length 2.
    # We need a path of length 3: u-x-y-v.
    # This means x is neighbor of u, y is neighbor of v, and x-y is an edge.
    # Also x != y.
    # Actually, simpler: Count common neighbors of u and v in the graph *without* (u,v).
    # If we add (u,v), a C4 is formed if there is a vertex x such that x is adjacent to u and v? No, that's a triangle.
    # C4: u-v-w-x-u.
    # If we add (u,v), we need w, x such that w~v, x~u, w~x.
    # This is equivalent to: Is there an edge (w,x) in the graph such that w is in N(v) and x is in N(u)?
    # Yes.
    
    # Efficient check:
    # Let N_u = neighbors of u
    # Let N_v = neighbors of v
    # We need to know if any edge connects a node in N_u to a node in N_v.
    # Wait, w must be neighbor of v. x must be neighbor of u.
    # And (w,x) must be an edge.
    # So we iterate over neighbors of v (w), and check if any neighbor of w (x) is in N_u.
    
    # Optimization:
    # If min(len(N_u), len(N_v)) is small, iterate over the smaller one.
    
    nu = adj[u]
    nv = adj[v]
    
    # We check if there is an edge between any node in nv and any node in nu.
    # Actually, w in nv, x in nu. Edge (w,x) exists?
    # Note: w cannot be u, x cannot be v (simple graph).
    # Also w != x.
    
    # Let's iterate over the smaller set to be safe, though degrees are ~7.
    if len(nv) < len(nu):
        for w in nv:
            if w == u: continue
            for x in adj[w]:
                if x in nu and x != v:
                    return True
    else:
        for x in nu:
            if x == v: continue
            for w in adj[x]:
                if w in nv and w != u:
                    return True
                    
    return False

def construct():
    # Target n=46, min_degree=7
    target_n = 46
    target_deg = target_n - 39 # 7
    
    # Try to construct for n=46
    # Use a randomized local search
    start_time = time.time()
    time_limit = 100 # Leave 20s for overhead
    
    # Initial random graph with avg degree ~8
    # We want to end up with min degree 7.
    
    best_n = 39
    best_edges = []
    
    # Fallback: A known valid graph?
    # The empty graph on 39 is valid.
    # Can we find a small valid graph with higher n?
    # Let's try to construct n=46.
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
            
        n = target_n
        adj = [set() for _ in range(n)]
        
        # Random initialization
        # Add random edges until we have enough
        # Target edges ~ n * deg / 2 = 46 * 7 / 2 = 161
        # Start with ~170 edges
        
        # Generate candidate edges
        candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(candidates)
        
        # Greedy add: if adding edge doesn't create C4, add it
        # But we also want high degree.
        # Let's add edges randomly, skipping if C4 created.
        
        added = 0
        for u, v in candidates:
            if added >= 170: # Stop early
                break
            if not has_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
        
        # Check min degree
        min_deg = min(len(adj[i]) for i in range(n))
        
        if min_deg >= target_deg:
            # Valid graph found
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
        # If min degree is low, try to repair
        # Identify low degree vertices
        # Try to add edges incident to them
        
        # Local Search / Simulated Annealing
        # For simplicity, just try a few random restarts in the loop above.
        # The greedy method with shuffle is effectively a random restart.
        
    # If we couldn't find n=46, try n=45?
    # The prompt says "Maximize n".
    # If 46 is hard, maybe 45 is easier?
    # But 46 is the target.
    
    # Fallback to a smaller known valid graph if search fails?
    # The empty graph on 39 is always valid.
    # But we can do better.
    # Let's try a deterministic construction for a smaller n if random fails.
    # Actually, the random search should find n=46 if it exists.
    # If it doesn't exist, we should return the best we found?
    # The function must return a VALID graph.
    # If we don't find one for 46, we must return a valid one for some n.
    # The empty graph on 39 is valid.
    
    # To be safe, if the search for 46 fails, return the 39-vertex empty graph.
    # This guarantees a valid output, though suboptimal if 46 exists and we just didn't find it.
    # However, 200 restarts with greedy is quite powerful for n=46, deg 7.
    
    return {'n': 39, 'edges': []}
