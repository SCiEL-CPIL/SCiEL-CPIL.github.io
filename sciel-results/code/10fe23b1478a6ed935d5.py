"""Construct a C4-free graph with high minimum degree using finite geometry / combinatorial designs."""
import time
import random
from itertools import combinations

def _is_c4_free(n, edges):
    """Check if graph is C4-free using bitsets for speed."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                # Check if i and j have a common neighbor k, and k and l share another common neighbor
                # Actually simpler: for each pair of vertices (i,j) that are NOT adjacent, check if they have 2+ common neighbors
                pass
    # Better: for each pair of non-adjacent vertices, count common neighbors
    # If any pair of non-adjacent vertices has >= 2 common neighbors, there's a C4
    # Actually C4: a-b-c-d-a. So a and c are non-adjacent (or adjacent), b and d non-adjacent (or adjacent).
    # Standard check: for every pair of vertices (u,v), the number of common neighbors should be <= 1
    # Wait, that's not right. C4 exists if there exist distinct a,b,c,d with a~b, b~c, c~d, d~a, and a!~c, b!~d.
    # Equivalently: for any two vertices a,c, if they have two common neighbors b,d, then if a!~c and b!~d, we have a C4.
    # Actually even if a~c or b~d, we might still have issues but the standard efficient check is:
    # For each pair (a,c), count common neighbors. If count >= 2, then pick two common neighbors b,d.
    # If a and c are not adjacent AND b and d are not adjacent, we have a C4.
    # But actually, if a and c are adjacent, and b,d are common neighbors, then a-b-c-d-a is not necessarily a C4 because a~c.
    # The cleanest check: for every pair of vertices (u,v), the number of common neighbors is at most 1.
    # This is a sufficient and necessary condition for C4-free? No.
    # Counterexample: K_{1,3} has center with 3 leaves, any two leaves have 1 common neighbor (the center). No C4.
    # But if two vertices have 2 common neighbors, say u and v have common neighbors w1, w2.
    # Then u-w1-v-w2-u is a 4-cycle if u!~v and w1!~w2.
    # If u~v, then u-w1-v-w2-u is not a simple cycle because u~v creates a triangle or chord.
    # Actually, u-w1-v-w2-u: edges are (u,w1), (w1,v), (v,w2), (w2,u). All distinct vertices if u!=v, w1!=w2, u!=w1, etc.
    # This IS a 4-cycle regardless of whether u~v or w1~w2. The cycle is u-w1-v-w2-u.
    # Wait, if u~v, then the subgraph has a chord, but the cycle u-w1-v-w2-u still exists as a cycle of length 4.
    # So YES: if any pair of vertices has >= 2 common neighbors, there is a C4.
    # Therefore, C4-free iff for all pairs (u,v), |N(u) ∩ N(v)| <= 1.
    
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Count bits in common
            c = common
            cnt = 0
            while c:
                cnt += 1
                c &= c - 1
            if cnt >= 2:
                return False
    return True

def _min_degree(n, edges):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return min(deg)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Strategy: Try to construct C4-free graphs with high minimum degree
    # Known constructions:
    # 1. Polar graphs of finite geometries
    # 2. Incidence graphs of projective planes (bipartite, C4-free)
    # 3. Random greedy with backtracking
    
    # For n=46, min degree >= 46-39 = 7
    # For n=47, min degree >= 47-39 = 8
    
    # Let's try random greedy construction with C4-free checking
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        # Random greedy: start with empty graph, add edges randomly
        # Check C4-free after each addition
        # Try multiple random seeds
        
        for seed in range(200):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            edges = []
            adj = [0] * target_n
            deg = [0] * target_n
            
            # Generate all possible edges
            all_edges = list(combinations(range(target_n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                
                # Try adding edge (u,v)
                # Check if this creates a C4
                # New C4 would involve u, v and two common neighbors of u and v after adding
                # Actually, adding (u,v) creates a C4 if there exist w1, w2 such that
                # u~w1, w1~v, v~w2, w2~u (the new edge is u-v)
                # So we need: w1 is common neighbor of u and v (before adding), and w2 is common neighbor of u and v (before adding)
                # Wait, the C4 is u-w1-v-w2-u. The edges are (u,w1), (w1,v), (v,w2), (w2,u).
                # (u,v) is NOT an edge in this C4. So adding (u,v) doesn't directly create this C4.
                # Adding (u,v) creates a C4 if there exist w1, w2 such that
                # (u,w1), (w1,v), (v,w2), (w2,u) exist AND (u,v) is the new edge... no.
                # A C4 involving the new edge (u,v): u-v-w-x-u.
                # So we need w adjacent to v, x adjacent to u, w adjacent to x, and w!=u, x!=v, w!=x.
                # So: for some w in N(v) and x in N(u), if w~x and w!=x and w!=u and x!=v, then u-v-w-x-u is a C4.
                
                # Check: for each w in N(v), for each x in N(u), if w~x and w!=x, then C4 exists
                # This is O(deg(v)*deg(u)) which is manageable
                
                creates_c4 = False
                for w in range(target_n):
                    if w == u or w == v:
                        continue
                    if not (adj[v] >> w) & 1:
                        continue
                    for x in range(target_n):
                        if x == u or x == v or x == w:
                            continue
                        if not (adj[u] >> x) & 1:
                            continue
                        if (adj[w] >> x) & 1:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                
                if not creates_c4:
                    # Add the edge
                    edges.append([u, v])
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            # Check minimum degree
            if min(deg) >= min_deg_needed:
                # Verify C4-free
                if _is_c4_free(target_n, edges):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = edges
                    break  # Found a valid graph for this n, move to next n
        
        # If we found a valid graph for this target_n, we can stop trying smaller n
        # Actually, we want the largest n, so if we found one for target_n, we're done
        if best_n >= target_n:
            break
    
    # If we didn't find anything better than 39, return the trivial one
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
