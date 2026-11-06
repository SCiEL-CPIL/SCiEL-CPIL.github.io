import random
import numpy as np

def construct():
    # Try to find a C4-free graph on n vertices with min degree >= n - 39
    # Maximize n.
    
    def is_valid(n, adj):
        # Check min degree
        degs = adj.sum(axis=1)
        if np.min(degs) < n - 39:
            return False
        # Check C4
        # A C4 exists if there are two non-adjacent vertices with >= 2 common neighbors?
        # No, that's a sufficient condition for a C4 if the common neighbors are not adjacent?
        # Actually, if u and v are not adjacent and have common neighbors a, b.
        # If a and b are not adjacent, u-a-v-b-u is a C4.
        # If a and b are adjacent, u-a-b-v-u is a C4? No, u-a, a-b, b-v are edges. u-v is not.
        # The cycle is u-a-b-v. We need edge v-u to close it? No, the cycle is u-a-b-v-...
        # Wait, a cycle of length 4 is u-v1-v2-v3-u.
        # If we have u, v non-adjacent. Common neighbors a, b.
        # Paths u-a-v and u-b-v.
        # If we combine them, we get a cycle u-a-v-b-u.
        # Edges: (u,a), (a,v), (v,b), (b,u).
        # This is a C4 if a != b.
        # So, any pair of non-adjacent vertices with at least 2 common neighbors forms a C4.
        # What if u and v are adjacent?
        # Then we look for a path of length 3 u-x-y-v.
        # Edges (u,x), (x,y), (y,v).
        # If we have edge (u,v), then u-x-y-v-u is a C4.
        # So, adjacent vertices with a common path of length 3 form a C4.
        
        # Efficient check:
        # For each pair (u, v):
        # If not adj[u][v]:
        #    if len(common_neighbors(u, v)) >= 2: return False
        # If adj[u][v]:
        #    # Check if there is x in N(u), y in N(v) such that adj[x][y] is True.
        #    # This is equivalent to checking if the subgraph induced by N(u) and N(v) has an edge.
        #    # Or simply: count paths of length 2 from u to v? No.
        #    # Number of C4s containing edge (u,v) is number of length 2 paths from u to v?
        #    # No, length 2 path is u-x-v. That's a triangle if (u,v) exists.
        #    # We need length 3 path u-x-y-v.
        #    # This is equivalent to: for each x in N(u), for each y in N(v), if adj[x][y]: C4 found.
        
        # This is O(n^4) or O(n^3) with bitsets. n=47 is small.
        
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    common = np.bitwise_and(adj[u], adj[v])
                    if np.sum(common) >= 2:
                        return False
                else:
                    # Check for C4 containing edge (u,v)
                    # Need x in N(u), y in N(v) such that adj[x][y]
                    # Note x != v, y != u
                    # N(u) includes v? adj[u][v] is 1.
                    # We should exclude v from N(u) and u from N(v) to avoid degenerate cases?
                    # If x=v, then path u-v-y-v? No.
                    # Standard definition: distinct vertices.
                    # So x != u, v. y != u, v.
                    # adj[u][v] is 1.
                    # Let's just check all x, y.
                    # If x=v, adj[v][y] must be 1. Path u-v-y-v? No.
                    # We require x in N(u) \ {v} and y in N(v) \ {u}.
                    nu = adj[u].copy()
                    nv = adj[v].copy()
                    nu[v] = 0
                    nv[u] = 0
                    # Check if there is an edge between any x in N(u) and y in N(v)
                    # This is equivalent to: (A @ A)[u][v] > 0?
                    # (A @ A)[u][v] counts paths of length 2.
                    # We need paths of length 3?
                    # No, u-x-y-v is length 3.
                    # (A @ A @ A)[u][v] counts paths of length 3.
                    # But we just need existence.
                    # We can check: for x in np.where(nu)[0]:
                    #    if np.dot(nu, adj[x]): # No, adj[x] is row x.
                    #    if np.any(np.bitwise_and(nv, adj[x])): return False
                    for x in np.where(nu)[0]:
                        if np.any(np.bitwise_and(nv, adj[x])):
                            return False
        return True

    def count_c4s(n, adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    common = np.bitwise_and(adj[u], adj[v])
                    c = np.sum(common)
                    if c >= 2:
                        # Number of pairs of common neighbors is c choose 2
                        count += c * (c - 1) // 2
                else:
                    nu = adj[u].copy()
                    nv = adj[v].copy()
                    nu[v] = 0
                    nv[u] = 0
                    for x in np.where(nu)[0]:
                        count += np.sum(np.bitwise_and(nv, adj[x]))
        return count

    def add_edge(n, adj, u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    def remove_edge(n, adj, u, v):
        adj[u][v] = 0
        adj[v][u] = 0

    for n in range(47, 38, -1):
        target_deg = n - 39
        # Randomized search
        best_adj = None
        best_score = float('inf')
        
        # Initialize with random graph
        p = max(0.1, (target_deg + 1) / (n - 1))
        
        for _ in range(100): # 100 restarts
            adj = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Simulated Annealing
            temp = 1.0
            for step in range(1000):
                # Calculate current score
                degs = adj.sum(axis=1)
                min_deg = np.min(degs)
                c4_count = count_c4s(n, adj)
                score = c4_count * 10 + max(0, target_deg - min_deg) * 5
                
                if score < best_score:
                    best_score = score
                    best_adj = adj.copy()
                
                if c4_count == 0 and min_deg >= target_deg:
                    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]}
                
                # Propose move
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                
                old_score = score
                if adj[u][v]:
                    remove_edge(n, adj, u, v)
                else:
                    add_edge(n, adj, u, v)
                
                # Calculate new score (approximate or exact)
                new_degs = adj.sum(axis=1)
                new_min_deg = np.min(new_degs)
                new_c4 = count_c4s(n, adj)
                new_score = new_c4 * 10 + max(0, target_deg - new_min_deg) * 5
                
                delta = new_score - old_score
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    pass # Accept
                else:
                    # Revert
                    if adj[u][v]:
                        remove_edge(n, adj, u, v)
                    else:
                        add_edge(n, adj, u, v)
                
                temp *= 0.999
            
            # Check best found in this restart
            if best_adj is not None:
                degs = best_adj.sum(axis=1)
                if np.min(degs) >= target_deg and count_c4s(n, best_adj) == 0:
                     return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i][j]]}

    # Fallback: return a small valid graph
    # n=39, empty graph is valid?
    # Min degree 0. n-39 = 0. Yes.
    return {'n': 39, 'edges': []}

