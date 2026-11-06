import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        # Target min degree
        min_deg_req = n - 39
        
        # Try multiple random restarts
        found = False
        attempts = 0
        while not found and time.time() < deadline:
            attempts += 1
            # Random graph initialization
            # p should be high enough to allow min degree, but not too high to create too many C4s
            # Expected degree = (n-1)*p
            # We want min degree >= min_deg_req
            # Let's aim for average degree ~ min_deg_req + 2
            p = (min_deg_req + 2) / (n - 1)
            
            adj = {i: set() for i in range(n)}
            
            # Generate random edges
            # To ensure connectivity and degree, we can just pick random pairs
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
                        
            # Remove C4s
            # We need to ensure no C4 exists.
            # Iteratively find C4s and remove edges.
            # A simple way: check all pairs of edges? No.
            # Check all quadruples? No.
            
            # Better: While there is a C4, remove an edge.
            # To find a C4:
            # For each pair of vertices (u, v) not connected, check if they have common neighbors.
            # If they have 2 or more common neighbors, say w1, w2, then u-w1-v-w2-u is a C4?
            # No, that's a diamond if (u,v) is not an edge.
            # Wait. If (u,v) is NOT an edge, and they share neighbors w1, w2,
            # then u-w1-v-w2-u is a C4.
            # So, any pair of non-adjacent vertices with >= 2 common neighbors forms a C4.
            
            # Also, if (u,v) IS an edge, and they share a neighbor w, that's a triangle, not C4.
            # But if there is a path of length 3, that's a C4.
            
            # So the condition for C4-freeness is:
            # 1. No triangles? No, C4-free allows triangles.
            # 2. For any non-edge (u,v), |N(u) \cap N(v)| <= 1.
            # 3. For any edge (u,v), there is no path of length 3 between u and v.
            #    i.e., no x,y such that u-x-y-v.
            
            # Let's just enforce: For all pairs (u,v), the number of common neighbors is <= 1?
            # If (u,v) is an edge, common neighbors are triangles.
            # If (u,v) is not an edge, common neighbors form C4s if count >= 2.
            
            # Actually, a simpler check:
            # A graph is C4-free iff for every pair of vertices u, v, |N(u) \cap N(v)| <= 1.
            # Is this true?
            # If |N(u) \cap N(v)| = 2, say w1, w2.
            # If (u,v) is an edge: u-w1-v and u-w2-v are triangles. The cycle u-w1-v-w2-u is length 4.
            # So yes, if (u,v) is an edge and they share 2 neighbors, we have a C4.
            # If (u,v) is NOT an edge and they share 2 neighbors, we have a C4 u-w1-v-w2-u.
            # So the condition |N(u) \cap N(v)| <= 1 for ALL pairs (u,v) is necessary and sufficient for C4-freeness.
            
            # So the algorithm is:
            # While exists pair (u,v) with |N(u) \cap N(v)| >= 2:
            #    Pick such a pair.
            #    Remove an edge to break the C4.
            #    Which edge? Removing (u, w1) or (v, w1) or (u, w2) or (v, w2).
            #    We should remove an edge that minimizes damage to min degree.
            
            c4_found = True
            while c4_found and time.time() < deadline:
                c4_found = False
                # Find a violating pair
                violating_pair = None
                violating_neighbors = []
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[u] & adj[v]
                        if len(common) >= 2:
                            violating_pair = (u, v)
                            violating_neighbors = list(common)[:2]
                            c4_found = True
                            break
                    if c4_found:
                        break
                
                if violating_pair:
                    u, v = violating_pair
                    w1, w2 = violating_neighbors
                    # Edges to consider removing:
                    # (u, w1), (u, w2), (v, w1), (v, w2)
                    # And potentially (u, v) if it exists?
                    # If (u,v) is an edge, removing it breaks the C4?
                    # C4 is u-w1-v-w2-u. Removing (u,v) doesn't break this cycle if (u,v) is not in the cycle.
                    # Wait, if (u,v) is an edge, the cycle is u-w1-v-w2-u.
                    # (u,v) is a diagonal. Removing it doesn't break the C4.
                    # We must remove one of the cycle edges.
                    
                    candidates = []
                    for w in [w1, w2]:
                        if w in adj[u]:
                            candidates.append((u, w))
                        if w in adj[v]:
                            candidates.append((v, w))
                    
                    # Pick one to remove
                    # Prefer removing edge from higher degree vertex?
                    # Or just random.
                    if candidates:
                        # Sort by degree of endpoints?
                        # Let's just pick the first one
                        eu, ev = candidates[0]
                        adj[eu].remove(ev)
                        adj[ev].remove(eu)
                        # Update edges list? We can rebuild at the end.
                        
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_req:
                found = True
                best_n = n
                # Rebuild edges list
                best_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append([u, v])
                # If we found n=47, we can stop.
                if n == 47:
                    break
            
        if found and n == 47:
            break

    return {'n': best_n, 'edges': best_edges}
