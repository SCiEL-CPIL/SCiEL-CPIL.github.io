import random
import numpy as np

def construct():
    """
    Uses a stochastic local search on general graphs to find large (5,5)-free graphs.
    This approach is not restricted to Cayley graphs, allowing it to explore the
    full space of graphs on n vertices.
    """
    CLIQUE = 5
    INDEPENDENT = 5
    MAX_VERTICES = 46

    def is_valid_graph(n, adj):
        """
        Checks if the graph defined by adjacency matrix 'adj' (n x n boolean numpy array)
        contains no K5 and no Independent set of size 5.
        """
        # Check for K5
        # For each vertex v, look for a K4 in its neighborhood
        for v in range(n):
            nbs = np.where(adj[v])[0]
            nbs = nbs[nbs != v]
            if len(nbs) < 4:
                continue
            
            # Check all 4-combinations of neighbors
            # This is O(d^4). To speed up, we can use bitsets or smarter pruning.
            # For n=46, d~23, 23C4 ~ 8855. 46 * 8855 ~ 400k checks.
            # Each check is O(1) with bitsets or O(4) with sets.
            # Let's use a set for neighbors for O(1) lookup.
            nb_set = set(nbs)
            
            for i in range(len(nbs)):
                u1 = nbs[i]
                # Intersection of neighbors of u1 and current nb_set
                common1 = set(np.where(adj[u1])[0]) & nb_set
                common1.discard(u1)
                if len(common1) < 3:
                    continue
                common1_list = list(common1)
                
                for j in range(len(common1_list)):
                    u2 = common1_list[j]
                    if u2 <= u1: # Avoid duplicates, though set handles it
                        continue
                    # Intersection of neighbors of u2 and common1
                    common2 = set(np.where(adj[u2])[0]) & common1
                    common2.discard(u2)
                    common2.discard(u1)
                    if len(common2) < 2:
                        continue
                    common2_list = list(common2)
                    
                    for k in range(len(common2_list)):
                        u3 = common2_list[k]
                        # Intersection of neighbors of u3 and common2
                        common3 = set(np.where(adj[u3])[0]) & common2
                        common3.discard(u3)
                        common3.discard(u2)
                        common3.discard(u1)
                        if len(common3) >= 1:
                            return False # Found K5
        
        # Check for Independent 5
        # Similar logic: For each vertex v, look for Independent 4 in non-neighbors
        for v in range(n):
            non_nbs = np.where(~adj[v])[0]
            non_nbs = non_nbs[non_nbs != v]
            if len(non_nbs) < 4:
                continue
            
            non_nb_set = set(non_nbs)
            
            for i in range(len(non_nbs)):
                u1 = non_nbs[i]
                # Non-neighbors of u1 within non_nb_set
                # An independent set requires no edges between them.
                # So we look for vertices in non_nb_set that are NOT neighbors of u1.
                non_common1 = non_nb_set - set(np.where(adj[u1])[0])
                non_common1.discard(u1)
                if len(non_common1) < 3:
                    continue
                non_common1_list = list(non_common1)
                
                for j in range(len(non_common1_list)):
                    u2 = non_common1_list[j]
                    if u2 <= u1:
                        continue
                    non_common2 = non_common1 - set(np.where(adj[u2])[0])
                    non_common2.discard(u2)
                    non_common2.discard(u1)
                    if len(non_common2) < 2:
                        continue
                    non_common2_list = list(non_common2)
                    
                    for k in range(len(non_common2_list)):
                        u3 = non_common2_list[k]
                        non_common3 = non_common2 - set(np.where(adj[u3])[0])
                        non_common3.discard(u3)
                        non_common3.discard(u2)
                        non_common3.discard(u1)
                        if len(non_common3) >= 1:
                            return False # Found Independent 5

        return True

    def paley_graph(n):
        """Constructs Paley graph on n vertices (n must be prime, n % 4 == 1)."""
        Q = set()
        for i in range(1, n):
            Q.add((i * i) % n)
        S = Q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in S:
                    edges.append((i, j))
        return n, edges

    best_n = 0
    best_edges = []

    # Try to find a valid graph for n from MAX_VERTICES down to 37
    for n in range(MAX_VERTICES, 36, -1):
        found = False
        
        # Try multiple random restarts
        for restart in range(20):
            random.seed(42 + restart * 1000 + n)
            np.random.seed(42 + restart * 1000 + n)
            
            # Initialize random graph
            # p = 0.5 is a good starting point
            adj = np.random.rand(n, n) < 0.5
            np.fill_diagonal(adj, False)
            # Symmetrize
            adj = adj | adj.T
            
            # Local Search
            # We want to minimize violations.
            # Strategy: Sample random 5-tuples. If violation, fix it.
            # We also do periodic full checks to confirm validity.
            
            # Pre-compute neighbor sets for speed? 
            # Since adj changes, we update on the fly or recompute.
            # Recomputing sets is O(n^2). Flipping is O(1).
            # Let's just use the matrix and compute neighbors when needed in the sampling.
            
            for step in range(5000):
                # Sample 5 vertices
                verts = random.sample(range(n), 5)
                
                # Check if this 5-set is a K5 or Independent Set
                is_k5 = True
                is_indep = True
                for i in range(5):
                    for j in range(i+1, 5):
                        if adj[verts[i]][verts[j]]:
                            is_indep = False
                        else:
                            is_k5 = False
                
                if is_k5 or is_indep:
                    # Pick a random edge in this set to flip
                    i, j = random.sample(range(5), 2)
                    u, v = verts[i], verts[j]
                    adj[u][v] = not adj[u][v]
                    adj[v][u] = adj[u][v]
                else:
                    # Occasionally flip a random edge elsewhere to escape local minima
                    if random.random() < 0.1:
                        u, v = random.sample(range(n), 2)
                        if u != v:
                            adj[u][v] = not adj[u][v]
                            adj[v][u] = adj[u][v]
                
                # Periodic full check
                if step % 100 == 0:
                    if is_valid_graph(n, adj):
                        found = True
                        break
            
            if found:
                break
        
        if found:
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append((i, j))
            best_edges = edges
            break

    # Fallback to Paley 37 if no better found
    if best_n == 0:
        best_n, best_edges = paley_graph(37)

    return best_n, best_edges
