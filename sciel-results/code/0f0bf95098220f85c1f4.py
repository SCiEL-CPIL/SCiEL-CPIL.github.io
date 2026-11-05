import time
from random import Random

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave 20s margin for 120s limit

    def generate_graph(n, p, seed):
        rng = Random(seed)
        # Use bitmasks for adjacency for faster operations
        adj = [0] * n
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
        return adj, edges

    def max_clique(adj, n, time_limit, start_time):
        """
        Finds a maximum clique in the graph defined by adj (list of bitmasks).
        Uses Bron-Kerbosch with pivot selection and branch-and-bound.
        Returns the clique as a list of vertices.
        """
        best_clique = []
        
        def bk(R, P, X, best_size):
            nonlocal best_clique
            if time.time() - start_time > time_limit:
                return
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            # If |R| + |P| <= current best, prune
            if len(R) + len(P) <= len(best_clique):
                return

            # Pivot selection
            # Choose u from P ∪ X with maximum |N(u) ∩ P|
            P_list = list(P)
            X_list = list(X)
            union = P_list + X_list
            
            if not union:
                return

            best_u = None
            max_inter = -1
            for u in union:
                # Count neighbors of u in P
                # adj[u] is a bitmask of neighbors
                # P is a set of vertices
                # We need to count how many vertices in P are in adj[u]
                # This is (adj[u] & mask(P)).bit_count()
                # But creating mask(P) every time is expensive.
                # Let's stick to iterating P for simplicity first, or optimize.
                # Given n <= 64, iterating P is O(n).
                inter = 0
                for v in P_list:
                    if (adj[u] >> v) & 1:
                        inter += 1
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            
            if best_u is None:
                return

            # Candidates are vertices in P not adjacent to best_u
            # candidates = [v for v in P if not (adj[best_u] >> v) & 1]
            
            # Iterate over candidates
            # To maintain P and X correctly in recursive calls, we usually iterate and remove
            # But since we are doing branch and bound for MAX clique, we just need to explore.
            
            # Optimization: Order candidates by degree in P? Or just iterate.
            # Let's iterate over a copy of P to avoid modification issues during iteration if we don't remove.
            # Actually, standard BK removes v from P and adds to X.
            # For max clique, we can just iterate over P.
            
            for v in P_list:
                if (adj[best_u] >> v) & 1:
                    continue # v is adjacent to pivot, so N(v) subset of N(pivot), skip if we want? 
                    # No, the property is: if v is in P and adjacent to pivot u, then any clique containing v and R is also a clique containing u and R? No.
                    # The property is: If we pick pivot u, then any maximal clique containing R must contain some vertex from P \ N(u).
                    # So we only need to recurse on v in P \ N(u).
                    pass

            # Correct logic:
            candidates = [v for v in P_list if not ((adj[best_u] >> v) & 1)]
            
            for v in candidates:
                # R' = R ∪ {v}
                # P' = P ∩ N(v)
                # X' = X ∩ N(v)
                
                R.add(v)
                
                P_new = set()
                for w in P:
                    if (adj[v] >> w) & 1:
                        P_new.add(w)
                
                X_new = set()
                for w in X:
                    if (adj[v] >> w) & 1:
                        X_new.add(w)
                
                bk(R, P_new, X_new, len(best_clique))
                
                P.remove(v)
                X.add(v)
                R.remove(v)
                
        R = set()
        P = set(range(n))
        X = set()
        bk(R, P, X, 0)
        return best_clique

    def get_clique_edges(clique):
        k = len(clique)
        edges = set()
        for i in range(k):
            for j in range(i+1, k):
                u, v = clique[i], clique[j]
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        adj, edges = generate_graph(n, p, seed)
        
        if time.time() - start_time > time_limit:
            return [[u, v] for u, v in edges]

        uncovered = set(edges)
        partition = []
        
        while uncovered:
            if time.time() - start_time > time_limit:
                break
                
            # Build residual adjacency
            # We only care about vertices that are in uncovered edges
            # And edges that are in uncovered
            
            # Optimization: If the number of edges is small, just add pairs?
            # No, we want max cliques.
            
            # Create a subgraph of the original graph induced by the uncovered edges
            # We can reuse the original adj matrix but filter by uncovered edges?
            # No, the "residual graph" has vertices 0..n-1, but an edge (u,v) exists in residual 
            # only if (u,v) is in uncovered.
            
            # So we need a new adjacency structure or filter in BK.
            # Let's create a boolean matrix or bitmask for the residual graph.
            
            res_adj = [0] * n
            active_vertices = set()
            for u, v in uncovered:
                res_adj[u] |= (1 << v)
                res_adj[v] |= (1 << u)
                active_vertices.add(u)
                active_vertices.add(v)
                
            if not active_vertices:
                break

            # Find max clique in this residual graph
            # Note: The residual graph is a subgraph of the original graph.
            # We can use the same max_clique function, but it needs to know which edges exist.
            # My max_clique function uses `adj`. So I should pass `res_adj`.
            
            max_c = max_clique(res_adj, n, time_limit, start_time)
            
            if not max_c or len(max_c) < 2:
                # No clique of size >= 2 found? 
                # This should not happen if there are edges, unless time limit hit.
                # If time limit hit, break.
                break
            
            partition.append(max_c)
            
            # Remove edges of this clique from uncovered
            c_edges = get_clique_edges(max_c)
            uncovered -= c_edges
            
        # Add remaining edges as cliques of size 2
        for u, v in uncovered:
            partition.append([u, v])
            
        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
