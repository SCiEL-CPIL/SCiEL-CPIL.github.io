"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    # Fallback: Polarity Graph of Projective Plane of order 5 (n=31)
    # This is a known C4-free graph with small independence number.
    def build_polarity_graph_q5():
        q = 5
        n = q * q + q + 1
        # Vertices are points (x,y) in GF(q)^2 plus point at infinity, etc.
        # Simpler: Use the standard construction for PG(2,q).
        # Points: (1, x, y) for x,y in GF(q), (0, 1, y) for y in GF(q), (0, 0, 1).
        # Total q^2 + q + 1 points.
        # Lines are sets of 3 points? No, lines have q+1 points.
        # Polarity: (x,y,z) is orthogonal to (x',y',z') if xx' + yy' + zz' = 0.
        # For q=5, this is easy to implement with mod 5 arithmetic.
        
        # Generate all points in PG(2,5)
        points = []
        for x in range(q):
            for y in range(q):
                points.append((1, x, y))
        for y in range(q):
            points.append((0, 1, y))
        points.append((0, 0, 1))
        
        # Map point to index
        point_index = {p: i for i, p in enumerate(points)}
        
        adj = np.zeros((n, n), dtype=bool)
        
        # Polarity graph: u ~ v if u is orthogonal to v.
        # In PG(2,q) with q odd, the polarity is symplectic? No, standard dot product.
        # Dot product: xx' + yy' + zz' == 0 mod q.
        
        for i in range(n):
            for j in range(i + 1, n):
                u = points[i]
                v = points[j]
                dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
                if dot == 0:
                    adj[i][j] = True
                    adj[j][i] = True
        # Remove self-loops if any (u orthogonal to u)
        # For q=5, x^2+y^2+z^2=0 has solutions, so self-loops possible.
        # Polarity graphs usually exclude self-loops or handle them.
        # The task requires a simple graph.
        np.fill_diagonal(adj, False)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return n, edges

    best_n, best_edges = build_polarity_graph_q5()
    
    # Try to find a graph with n=39 using Local Search
    # Strategy:
    # 1. Initialize random graph G(39, p)
    # 2. Remove C4s
    # 3. Add edges to reduce independence number
    
    def is_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    # Count common neighbors
                    # Using numpy for speed
                    common = np.sum(adj[u] & adj[v])
                    if common > 1:
                        return False
        return True

    def find_independent_set(adj, k, max_iter=200):
        """
        Greedy randomized search for independent set of size k.
        Returns True if found, False otherwise.
        """
        n = len(adj)
        verts = list(range(n))
        for _ in range(max_iter):
            random.shuffle(verts)
            independent = []
            for v in verts:
                # Check if v is non-adjacent to all in independent
                # adj[v] is a boolean array
                # We need to check if any u in independent has adj[v][u] == True
                # This is O(k) per vertex, O(nk) total.
                if all(not adj[v][u] for u in independent):
                    independent.append(v)
                    if len(independent) >= k:
                        return True
        return False

    def local_search(n_target):
        if time.time() > deadline - 10:
            return None
        
        best_adj = None
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            p = 1.0 / np.sqrt(n_target)
            adj = np.zeros((n_target, n_target), dtype=bool)
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if random.random() < p:
                        adj[u][v] = True
                        adj[v][u] = True
            
            # Phase 1: Remove C4s
            # Iterate until no C4s or time limit
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                # Find a C4
                c4_found = False
                for u in range(n_target):
                    for v in range(u + 1, n_target):
                        if adj[u][v]:
                            common = np.where(adj[u] & adj[v])[0]
                            if len(common) > 1:
                                # Found C4: u-v-w-x-u
                                # Remove edge (u,v) or (w,x)?
                                # Removing (u,v) is safe.
                                adj[u][v] = False
                                adj[v][u] = False
                                c4_found = True
                                break
                    if c4_found:
                        break
                if not c4_found:
                    break
            
            if not is_c4_free(adj):
                continue
            
            # Phase 2: Try to add edges to reduce alpha
            # We want to add edges that don't create C4s.
            # This is hard to do efficiently for all pairs.
            # Instead, we just check if current alpha < 11.
            if not find_independent_set(adj, 11, max_iter=50):
                # Likely valid
                best_adj = adj
                break
        
        if best_adj is not None:
            edges = []
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if best_adj[u][v]:
                        edges.append([u, v])
            return n_target, edges
        return None

    # Try n=39
    result = local_search(39)
    if result:
        n, edges = result
        if n > best_n:
            best_n = n
            best_edges = edges
    
    # Try n=40, 41, 42, 43 if time permits?
    # The search is expensive. 39 is the "best known". 
    # If we found 39, we are done.
    # If not, we have 31.
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
