import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 320.0  # 300s limit with buffer
    
    def time_left():
        return deadline - time.time()
    
    # Helper: Count triangles in a set of vertices given adjacency bitsets
    # vertices: list of ints
    # adj: list of ints (bitsets)
    def count_triangles(vertices, adj):
        if len(vertices) < 3:
            return 0
        # Map vertex to index in list for faster lookup? No, use bitsets directly.
        # vertices are global indices.
        count = 0
        # To speed up, we can just iterate over the list
        # But we need to check if (a,b) is an edge.
        # adj[a] & (1 << b)
        for i in range(len(vertices)):
            a = vertices[i]
            for j in range(i + 1, len(vertices)):
                b = vertices[j]
                if adj[a] & (1 << b):
                    # Count common neighbors in vertices
                    common = adj[a] & adj[b]
                    # Mask to only include vertices in the set
                    # We can precompute a mask for the set
                    # But doing it every time is slow.
                    # Let's just check each vertex in the list?
                    # Or use bit_count on (common & mask)
                    pass
        # This is getting complicated to optimize inside the delta calc.
        # Let's stick to the list iteration for the small sets involved in delta.
        
        # Actually, for the delta calculation, the set of candidates is 
        # common neighbors of i and j. This set is small.
        # So we can just iterate.
        pass

    def count_triangles_in_set(candidates, adj):
        """Count triangles in the induced subgraph on 'candidates'."""
        n_cand = len(candidates)
        if n_cand < 3:
            return 0
        count = 0
        for i in range(n_cand):
            a = candidates[i]
            for j in range(i + 1, n_cand):
                b = candidates[j]
                if adj[a] & (1 << b):
                    # Find common neighbors of a and b within candidates
                    common = adj[a] & adj[b]
                    # Check each candidate k > j
                    # Optimization: precompute mask?
                    # For small sets, simple loop is fine.
                    for k in range(j + 1, n_cand):
                        c = candidates[k]
                        if common & (1 << c):
                            count += 1
        return count

    def count_ind_triangles_in_set(candidates, adj):
        """Count independent triangles (triangles in complement) in 'candidates'."""
        n_cand = len(candidates)
        if n_cand < 3:
            return 0
        count = 0
        for i in range(n_cand):
            a = candidates[i]
            for j in range(i + 1, n_cand):
                b = candidates[j]
                # In complement, (a,b) is edge if NOT (a,b) in original
                if not (adj[a] & (1 << b)):
                    # Find common non-neighbors of a and b within candidates
                    # common_non = ~(adj[a] | adj[b])
                    # But we must restrict to candidates.
                    # And a,b are not in candidates? No, they are.
                    # We need c such that c not connected to a and c not connected to b.
                    for k in range(j + 1, n_cand):
                        c = candidates[k]
                        if not (adj[a] & (1 << c)) and not (adj[b] & (1 << c)):
                            count += 1
        return count

    def total_cost(adj, n):
        """Calculate total number of K5 and Independent 5-sets."""
        k5_count = 0
        ind5_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    # Common neighbors
                    common = adj[i] & adj[j]
                    # Get list of common neighbors
                    # Optimization: we can just iterate bits?
                    # Or use a helper to get list.
                    # For n=42, we can just loop k
                    common_list = []
                    for k in range(n):
                        if k != i and k != j and (common & (1 << k)):
                            common_list.append(k)
                    k5_count += count_triangles_in_set(common_list, adj)
                else:
                    # Common non-neighbors
                    common_non = 0
                    for k in range(n):
                        if k != i and k != j:
                            if not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                                common_non |= (1 << k)
                    common_non_list = []
                    for k in range(n):
                        if common_non & (1 << k):
                            common_non_list.append(k)
                    ind5_count += count_ind_triangles_in_set(common_non_list, adj)
        
        # Each K5 is counted 10 times (5*4/2 pairs, each K5 has 10 pairs? No.)
        # A K5 has 10 edges. For each edge, we count the triangles in the remaining 3 vertices.
        # The remaining 3 vertices form a K3 (triangle).
        # So each K5 is counted 10 times.
        # Similarly for Independent 5-sets.
        return k5_count // 10 + ind5_count // 10

    def delta_cost(i, j, adj, n, is_edge):
        """Calculate change in cost if we flip edge (i,j).
        is_edge: True if (i,j) is currently an edge (will be removed).
                 False if (i,j) is not an edge (will be added).
        """
        if is_edge:
            # Removing edge (i,j)
            # Destroys K5s containing (i,j)
            # Creates Ind5s containing (i,j) (which means (i,j) was non-edge in Ind graph, i.e., edge in G)
            
            # Common neighbors of i and j
            common = adj[i] & adj[j]
            common_list = []
            for k in range(n):
                if k != i and k != j and (common & (1 << k)):
                    common_list.append(k)
            
            # Triangles in common_list are the K5s destroyed
            delta_k5 = -count_triangles_in_set(common_list, adj)
            
            # Common non-neighbors of i and j
            # These form Ind5s in the complement?
            # If we remove edge (i,j) from G, it becomes an edge in G_bar.
            # An Ind5 in G is a K5 in G_bar.
            # So we are adding an edge to G_bar.
            # This creates new K5s in G_bar (which are Ind5s in G).
            # The new K5s in G_bar must contain the new edge (i,j) in G_bar.
            # So we need common neighbors of i and j in G_bar.
            # Neighbors in G_bar are non-neighbors in G.
            
            common_non = 0
            for k in range(n):
                if k != i and k != j:
                    if not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        common_non |= (1 << k)
            
            common_non_list = []
            for k in range(n):
                if common_non & (1 << k):
                    common_non_list.append(k)
            
            # Triangles in common_non_list (in G_bar)
            # Which are Ind triangles in G.
            delta_ind5 = count_ind_triangles_in_set(common_non_list, adj)
            
            return delta_k5 + delta_ind5
        else:
            # Adding edge (i,j)
            # Creates K5s containing (i,j)
            # Destroys Ind5s containing (i,j)
            
            common = adj[i] & adj[j]
            common_list = []
            for k in range(n):
                if k != i and k != j and (common & (1 << k)):
                    common_list.append(k)
            
            delta_k5 = count_triangles_in_set(common_list, adj)
            
            common_non = 0
            for k in range(n):
                if k != i and k != j:
                    if not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        common_non |= (1 << k)
            
            common_non_list = []
            for k in range(n):
                if common_non & (1 << k):
                    common_non_list.append(k)
            
            delta_ind5 = -count_ind_triangles_in_set(common_non_list, adj)
            
            return delta_k5 + delta_ind5

    best_n = 37
    best_edges = None
    
    # Try to find a valid graph for N = 42, 41, ... down to 38
    for N in range(42, 37, -1):
        if time_left() < 30:
            break
            
        # Initialize random graph
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = total_cost(adj, N)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        steps = 0
        max_steps = 50000
        
        while T > min_T and steps < max_steps:
            if time_left() < 5:
                break
            
            # Pick random edge
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            is_edge = bool(adj[i] & (1 << j))
            
            d_cost = delta_cost(i, j, adj, N, is_edge)
            
            # Accept if better or with probability exp(-d/T)
            if d_cost < 0 or random.random() < np.exp(-d_cost / T):
                # Flip edge
                if is_edge:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                current_cost += d_cost
                if current_cost == 0:
                    # Found a valid graph!
                    edges = []
                    for u in range(N):
                        for v in range(u + 1, N):
                            if adj[u] & (1 << v):
                                edges.append((u, v))
                    return N, edges
            
            T *= cooling
            steps += 1
            
            # If cost is stuck, try a few random restarts or perturbations
            if steps % 1000 == 0 and current_cost > 0:
                # Perturb: flip 10 random edges
                for _ in range(10):
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                current_cost = total_cost(adj, N)
                
    # If we didn't find a valid graph > 37, return the seed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges
