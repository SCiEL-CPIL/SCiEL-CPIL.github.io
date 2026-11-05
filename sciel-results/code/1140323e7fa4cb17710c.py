import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Try n from 36 down to 17
    for n in range(36, 16, -1):
        if time.time() > deadline - 10: # Need time to return
            break
            
        # Try to find a valid graph on n vertices
        # Multiple restarts for n=36, fewer for smaller n?
        # For n=35, it's known to exist, so should be easy.
        # For n=36, it's unknown, so might be hard.
        
        max_restarts = 5 if n > 35 else 10
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            success = False
            steps = 0
            while time.time() < deadline - 2:
                steps += 1
                # Check K4
                k4_found = False
                # Iterate 4-tuples
                for a in range(n):
                    for b in range(a+1, n):
                        if not (adj[a] & (1 << b)): continue
                        for c in range(b+1, n):
                            if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                            for d in range(c+1, n):
                                if not (adj[a] & (1 << d)) or not (adj[b] & (1 << d)) or not (adj[c] & (1 << d)): continue
                                # Found K4: a,b,c,d
                                k4_found = True
                                # Pick random edge to remove
                                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                u, v = random.choice(edges)
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                break
                            if k4_found: break
                        if k4_found: break
                    if k4_found: break
                
                if k4_found:
                    continue
                
                # Check Ind6
                ind6_found = False
                # Iterate 6-tuples
                # This is the slow part.
                # Optimization: If we just fixed a K4, we might have created Ind6s?
                # Removing an edge cannot create an Ind6 (it only makes graph sparser, more independent).
                # Wait. Removing an edge (u,v) means u and v are now non-adjacent.
                # If they were part of an Ind6, they were already non-adjacent.
                # So removing edges does NOT create new Ind6s. It only reduces the number of edges, potentially making it *easier* to have Ind6s?
                # No. An Ind6 is a set of 6 vertices with NO edges.
                # If we remove an edge, we are adding a non-edge.
                # This can only *create* new Ind6s or keep existing ones.
                # Example: {1,2,3,4,5,6} was not Ind6 because (1,2) was an edge.
                # Remove (1,2). Now {1..6} might be Ind6 if no other edges.
                # So removing edges for K4 repair can create Ind6s.
                
                # Adding edges for Ind6 repair can create K4s.
                
                # So we must alternate or check both.
                
                # Check Ind6
                # To speed up, we can sample?
                # No, we need exactness for the final answer, but for search, we can be heuristic.
                # But if we miss an Ind6, we might think we are valid when we are not.
                # So we must check all.
                
                # Pruning for Ind6 check:
                # Pick a vertex v. It must be non-adjacent to 5 others.
                # If degree of v in complement is < 5, it can't be in an Ind6.
                # Complement degree = n - 1 - degree(v).
                # So if degree(v) > n - 6, v cannot be in an Ind6.
                
                # Let's implement the full scan but with pruning.
                
                # Precompute candidates for Ind6?
                # A vertex v is a candidate for Ind6 if it has at least 5 non-neighbors.
                # non_neighbors(v) = ~adj[v] & mask_all
                # If popcount(non_neighbors(v)) < 5, skip v.
                
                # Let's just do the nested loops with early termination.
                
                for a in range(n):
                    # Check if a can be in Ind6
                    # Count non-neighbors of a
                    # non_neigh_a = [x for x in range(n) if x != a and not (adj[a] & (1<<x))]
                    # If len < 5, continue
                    # This is O(n) per a. Total O(n^2). 36^2 = 1296. Fast.
                    
                    non_neigh_a = []
                    for x in range(n):
                        if x == a: continue
                        if not (adj[a] & (1 << x)):
                            non_neigh_a.append(x)
                    if len(non_neigh_a) < 5:
                        continue
                    
                    # Now pick 5 from non_neigh_a that form Ind5 with each other and with a
                    # This is getting complex.
                    # Let's stick to the 6-tuple scan but break early if possible.
                    
                    # Actually, the previous K4 scan found nothing.
                    # So we proceed to Ind6 scan.
                    
                    # Standard 6-tuple scan
                    for b in range(a+1, n):
                        if adj[a] & (1 << b): continue
                        for c in range(b+1, n):
                            if adj[a] & (1 << c) or adj[b] & (1 << c): continue
                            for d in range(c+1, n):
                                if adj[a] & (1 << d) or adj[b] & (1 << d) or adj[c] & (1 << d): continue
                                for e in range(d+1, n):
                                    if adj[a] & (1 << e) or adj[b] & (1 << e) or adj[c] & (1 << e) or adj[d] & (1 << e): continue
                                    for f in range(e+1, n):
                                        if adj[a] & (1 << f) or adj[b] & (1 << f) or adj[c] & (1 << f) or adj[d] & (1 << f) or adj[e] & (1 << f): continue
                                        # Found Ind6: a,b,c,d,e,f
                                        ind6_found = True
                                        # Pick random pair to add edge
                                        pairs = [(a,b), (a,c), (a,d), (a,e), (a,f),
                                                 (b,c), (b,d), (b,e), (b,f),
                                                 (c,d), (c,e), (c,f),
                                                 (d,e), (d,f),
                                                 (e,f)]
                                        u, v = random.choice(pairs)
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                        break
                                    if ind6_found: break
                                if ind6_found: break
                            if ind6_found: break
                        if ind6_found: break
                    if ind6_found: break
                
                if not ind6_found and not k4_found:
                    # Valid graph found!
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}

    # Fallback: return the seed graph if nothing better found
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
