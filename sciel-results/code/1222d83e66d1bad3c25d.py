import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # The task requires a triangle-free graph with no independent set of size 10.
    # This is equivalent to a graph G such that G is triangle-free and alpha(G) < 10.
    # Or, considering the complement H: H has no independent set of size 3 (triangle-free) 
    # and no clique of size 10.
    # Wait, the task says "Triangle-free graphs without independent 10-sets".
    # So G is triangle-free. And G has no independent set of size 10.
    # This is a standard Ramsey number R(3, 10) problem.
    # R(3, 10) > 39 is known. The largest known graph is 39 vertices.
    # The graph is the complement of the 39-vertex strongly regular graph with parameters (39, 14, 5, 6).
    # Let's verify this.
    # If H is SRG(39, 14, 5, 6), then G = complement(H) is SRG(39, 39-1-14, 39-1-2*14-6+5, 39-1-2*14+6) = SRG(39, 24, 13, 14).
    # Wait, parameters for complement of SRG(n, k, lambda, mu):
    # n' = n
    # k' = n - 1 - k
    # lambda' = n - 2 - 2k + lambda
    # mu' = n - 2k + mu
    # For H = SRG(39, 14, 5, 6):
    # k' = 39 - 1 - 14 = 24
    # lambda' = 39 - 2 - 28 + 5 = 14
    # mu' = 39 - 28 + 6 = 17
    # So G is SRG(39, 24, 14, 17).
    # Is G triangle-free?
    # A triangle in G corresponds to an independent set of size 3 in H.
    # Does H have independent sets of size 3?
    # In an SRG, the independence number alpha is related to the eigenvalues.
    # Eigenvalues of SRG(39, 14, 5, 6):
    # k=14.
    # r, s are roots of x^2 + (mu - lambda)x + (mu - k) = 0
    # x^2 + (6-5)x + (6-14) = 0 => x^2 + x - 8 = 0
    # Roots are (-1 +/- sqrt(1+32))/2 = (-1 +/- sqrt(33))/2.
    # This doesn't look like integer eigenvalues, which is typical for non-integral SRGs, but SRG parameters must yield integer eigenvalues?
    # Wait, the characteristic polynomial of the adjacency matrix has integer coefficients, so eigenvalues are algebraic integers.
    # But for SRGs, the eigenvalues r and s are roots of that quadratic.
    # If the discriminant is not a perfect square, the eigenvalues are irrational.
    # However, the multiplicities must be integers.
    # m_r + m_s = n - 1 = 38.
    # k*m_r + s*m_s = 0.
    # 14 m_r + s m_s = 0.
    # m_r = -s * 38 / (14 - s).
    # If s = (-1 - sqrt(33))/2, this is not an integer.
    # So SRG(39, 14, 5, 6) does not exist?
    # Let's re-check the parameters for the 39-vertex Ramsey graph.
    
    # The known 39-vertex graph for R(3,10) is often cited.
    # According to Radziszowski's survey, R(3,10) is between 39 and 41.
    # The 39-vertex graph is the complement of the graph with parameters...
    # Actually, let's look for a known valid graph.
    # The "Clebsch graph" is n=16, k=5, lambda=0, mu=2. Triangle-free. Alpha=5.
    # If we take the disjoint union of two Clebsch graphs, n=32, alpha=10. Not good.
    # We need alpha < 10.
    
    # There is a known 39-vertex graph.
    # It is the complement of the 39-vertex graph formed by the icosahedron?
    # The icosahedron has 12 vertices.
    # Maybe the graph is related to the 5-dimensional hypercube? No.
    
    # Let's try to construct a graph using the "Blow-up" of a smaller graph.
    # If we blow up C5 (5-cycle) with bags of size k, n=5k.
    # For n=35, k=7. Alpha = 7+7=14. Too big.
    # We need a denser graph.
    
    # What if we use the line graph of a bipartite graph?
    # Line graphs are triangle-free if the original graph is bipartite? No, line graphs of bipartite graphs are not necessarily triangle-free.
    # A line graph is triangle-free iff the original graph is a matching or a star? No.
    # Line graph L(G) has a triangle iff G has a triangle or a claw (K_{1,3}).
    # So if G is triangle-free and claw-free, L(G) is triangle-free.
    # Bipartite graphs are triangle-free. If we choose a bipartite graph G with no claw, L(G) is triangle-free.
    # A bipartite graph with no claw means max degree <= 2?
    # If max degree <= 2, G is a collection of paths and cycles.
    # L(G) is a collection of paths and cycles.
    # Alpha of L(G) is related to the matching number of G.
    # This seems too small.
    
    # Let's go back to the SRG idea.
    # Maybe the parameters are different.
    # Search for "39 vertex triangle free graph independence number 9".
    # The graph is the complement of the 39-vertex graph with parameters (39, 24, 13, 14)?
    # Let's check if SRG(39, 24, 13, 14) exists.
    # k=24.
    # x^2 + (14-13)x + (14-24) = 0 => x^2 + x - 10 = 0.
    # Roots (-1 +/- sqrt(41))/2. Not integer.
    
    # What about SRG(39, 14, 4, 6)?
    # x^2 + 2x - 8 = 0 => x=2, -4.
    # m_2 + m_-4 = 38.
    # 14 m_2 - 4 m_-4 = 0 => 7 m_2 = 2 m_-4.
    # m_2 = 8, m_-4 = 28.
    # 8+28=36 != 38. No.
    
    # What about SRG(39, 14, 5, 6) again?
    # Maybe I made a mistake in the discriminant.
    # x^2 + (6-5)x + (6-14) = x^2 + x - 8.
    # Discriminant 1 + 32 = 33.
    
    # Is there a 39-vertex SRG?
    # The only known 39-vertex SRGs are...
    # Actually, there is a unique SRG(39, 14, 5, 6)? No, it doesn't exist.
    
    # Let's try a different known construction.
    # The graph is the "Hoffman-Singleton" graph? n=50.
    # The "Clebsch" graph? n=16.
    
    # Maybe the 39-vertex graph is not an SRG.
    # It is often constructed from the icosahedron.
    # Vertices: 12 (icosahedron vertices) + 30 (icosahedron edges) = 42?
    # If we remove 3 vertices?
    
    # Let's use a robust local search with bitmask optimization.
    # The previous program used numpy arrays which are slow for bit operations.
    # Using Python integers as bitmasks is much faster.
    
    best_result = None
    best_n = 0
    
    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with a random graph
            # We want it to be close to triangle-free.
            # Start with a random bipartite graph?
            # Or a random graph and then fix triangles.
            
            # Let's start with a random graph with edge probability 0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Phase 1: Make triangle-free
            # Find a triangle and remove an edge.
            # To find a triangle efficiently:
            # For each edge (u, v), check if there is a common neighbor w.
            # common = adj[u] & adj[v]
            # If common is not 0, pick a bit w.
            
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                tri_found = False
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            common = adj[u] & adj[v]
                            if common:
                                # Find a set bit in common
                                # lsb = common & -common
                                # w = lsb.bit_length() - 1
                                # Remove edge (u, v)
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                tri_found = True
                                break
                    if tri_found:
                        break
                if not tri_found:
                    break
            
            # Phase 2: Reduce Alpha
            # We want alpha < 10.
            # Loop:
            #   Find an independent set of size 10.
            #   If not found, valid.
            #   If found, try to add an edge within this set.
            #   Adding edge (u, v) is safe if N(u) and N(v) are disjoint.
            #   N(u) = adj[u].
            #   Disjoint means (adj[u] & adj[v]) == 0.
            
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                iso = find_independent_set_bitmask(adj, n, 10)
                
                if iso is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    break
                else:
                    # iso is a list of 10 vertices
                    improved = False
                    for i in range(10):
                        for j in range(i+1, 10):
                            u = iso[i]
                            v = iso[j]
                            if not ((adj[u] >> v) & 1):
                                # Check if adding (u, v) creates a triangle
                                if (adj[u] & adj[v]) == 0:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                    improved = True
                                    break
                        if improved:
                            break
                    
                    if not improved:
                        break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph from current program
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def find_independent_set_bitmask(adj, n, k):
    """
    Find an independent set of size k in the graph defined by adj (list of bitmasks).
    Returns a list of vertices if found, None otherwise.
    """
    # Precompute non-neighbors (including self? No, independent set vertices must be non-adjacent to each other).
    # non_neighbors[i] should contain j if i != j and not adj[i][j].
    # Actually, for the search, we maintain a set of candidates.
    # Initially, candidates = all vertices.
    # When we pick v, new candidates = candidates & non_neighbors[v] & ~v.
    
    non_neighbors = [0] * n
    for i in range(n):
        # ~adj[i] will have 1s for non-neighbors.
        # But we need to mask to n bits and exclude i itself.
        mask = ((1 << n) - 1) ^ adj[i]
        mask &= ~(1 << i)
        non_neighbors[i] = mask
        
    best_set = []
    found = [False]
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            best_set.extend(current_set)
            found[0] = True
            return
        
        # Pruning: if not enough candidates left
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        # Heuristic: pick a vertex from candidates.
        # To speed up, we can pick a vertex with fewest non-neighbors in candidates?
        # Or just the first one.
        # Let's try a simple heuristic: pick the vertex with the smallest index.
        # This is fast.
        
        if candidates_mask == 0:
            return
            
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        new_candidates = candidates_mask & non_neighbors[v] & ~lsb
        dfs(new_candidates, current_set + [v])
        
        if found[0]:
            return
        
        # Branch 2: Exclude v
        dfs(candidates_mask & ~lsb, current_set)
        
    dfs((1 << n) - 1, [])
    
    if found[0]:
        return best_set
    return None
