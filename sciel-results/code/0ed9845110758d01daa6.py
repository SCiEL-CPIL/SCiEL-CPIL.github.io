import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 325.0  # 300s limit with buffer
    
    def time_left():
        return deadline - time.time()
    
    def count_triangles_in_mask(mask, adj, n):
        """Count triangles in the induced subgraph on vertices in 'mask'."""
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        n_cand = len(verts)
        if n_cand < 3:
            return 0
        
        count = 0
        # Iterate over pairs
        for i in range(n_cand):
            a = verts[i]
            adj_a = adj[a]
            for j in range(i + 1, n_cand):
                b = verts[j]
                if adj_a & (1 << b):
                    # Count common neighbors in verts[j+1:]
                    common = adj_a & adj[b]
                    # Mask out i and j (already done by common if no self loops, but ensure)
                    # We only care about k > j in the list
                    # Create a mask of verts[k] for k > j
                    # Instead of building a mask, just iterate
                    # Optimization: precompute mask of suffix?
                    # For now, simple loop
                    for k in range(j + 1, n_cand):
                        c = verts[k]
                        if common & (1 << c):
                            count += 1
        return count

    def count_ind_triangles_in_mask(mask, adj, n):
        """Count independent triangles (triangles in complement) in 'mask'."""
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        n_cand = len(verts)
        if n_cand < 3:
            return 0
        
        count = 0
        for i in range(n_cand):
            a = verts[i]
            adj_a = adj[a]
            for j in range(i + 1, n_cand):
                b = verts[j]
                if not (adj_a & (1 << b)):
                    for k in range(j + 1, n_cand):
                        c = verts[k]
                        if not (adj_a & (1 << c)) and not (adj[b] & (1 << c)):
                            count += 1
        return count

    def total_cost(adj, n):
        """Calculate total number of K5 and Independent 5-sets."""
        k5_count = 0
        ind5_count = 0
        
        # Iterate over all pairs (i, j)
        for i in range(n):
            adj_i = adj[i]
            for j in range(i + 1, n):
                if adj_i & (1 << j):
                    # Common neighbors of i and j, excluding i and j
                    common = adj_i & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    k5_count += count_triangles_in_mask(common, adj, n)
                else:
                    # Common non-neighbors of i and j
                    # Non-neighbor of i: ~(adj_i | (1<<i))
                    # Non-neighbor of j: ~(adj_j | (1<<j))
                    non_i = ~(adj_i | (1 << i))
                    non_j = ~(adj_j | (1 << j))
                    common_non = non_i & non_j
                    # Mask to n vertices
                    mask_n = (1 << n) - 1
                    common_non &= mask_n
                    ind5_count += count_ind_triangles_in_mask(common_non, adj, n)
        
        # Each K5 is counted 10 times (5 choose 2 pairs)
        # Each Ind5 is counted 10 times
        return k5_count // 10 + ind5_count // 10

    def delta_cost(i, j, adj, n, is_edge):
        """Calculate change in cost if we flip edge (i,j)."""
        adj_i = adj[i]
        adj_j = adj[j]
        
        if is_edge:
            # Removing edge (i,j)
            # This destroys K5s containing edge (i,j)
            # K5s containing edge (i,j) are in common neighbors of i and j
            common = adj_i & adj_j
            common &= ~(1 << i)
            common &= ~(1 << j)
            delta_k5 = -count_triangles_in_mask(common, adj, n)
            
            # This creates Ind5s containing edge (i,j) in the complement?
            # No, Ind5s in the graph are independent sets.
            # An independent set of size 5 containing i and j requires i and j to be non-adjacent.
            # Currently they are adjacent, so no Ind5 contains them.
            # After removal, they become non-adjacent.
            # New Ind5s containing i and j are formed by common non-neighbors of i and j.
            non_i = ~(adj_i | (1 << i))
            non_j = ~(adj_j | (1 << j))
            common_non = non_i & non_j
            mask_n = (1 << n) - 1
            common_non &= mask_n
            delta_ind5 = count_ind_triangles_in_mask(common_non, adj, n)
            
            return delta_k5 + delta_ind5
        else:
            # Adding edge (i,j)
            # Creates K5s containing edge (i,j)
            common = adj_i & adj_j
            common &= ~(1 << i)
            common &= ~(1 << j)
            delta_k5 = count_triangles_in_mask(common, adj, n)
            
            # Destroys Ind5s containing edge (i,j) (which were non-adjacent pairs)
            non_i = ~(adj_i | (1 << i))
            non_j = ~(adj_j | (1 << j))
            common_non = non_i & non_j
            mask_n = (1 << n) - 1
            common_non &= mask_n
            delta_ind5 = -count_ind_triangles_in_mask(common_non, adj, n)
            
            return delta_k5 + delta_ind5

    # Try to find a valid graph for N = 42, 41, ... down to 38
    # We will use a random initialization biased towards the Paley graph structure if possible,
    # or just random.
    
    best_result = None
    
    for N in range(42, 37, -1):
        if time_left() < 30:
            break
        
        # Multiple restarts for this N
        # Reduce restarts if time is tight
        max_restarts = 20
        for restart in range(max_restarts):
            if time_left() < 10:
                break
            
            # Initialize random graph with p=0.5
            # For N=42, p=0.5 is a good start.
            adj = [0] * N
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_cost = total_cost(adj, N)
            
            # Simulated Annealing
            T = 5.0
            cooling = 0.995
            min_T = 0.01
            steps = 0
            max_steps = 50000 # Increased steps due to faster delta_cost
            
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
                if d_cost < 0 or (d_cost == 0) or (d_cost > 0 and random.random() < np.exp(-d_cost / T)):
                    if is_edge:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    
                    current_cost += d_cost
                    if current_cost == 0:
                        edges = []
                        for u in range(N):
                            for v in range(u + 1, N):
                                if adj[u] & (1 << v):
                                    edges.append((u, v))
                        return N, edges
                
                T *= cooling
                steps += 1
                
                # If cost is stuck, perturb
                if steps % 1000 == 0 and current_cost > 0:
                    for _ in range(10):
                        u = random.randint(0, N - 1)
                        v = random.randint(0, N - 1)
                        if u == v:
                            continue
                        if u > v:
                            u, v = v, u
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    current_cost = total_cost(adj, N)

    # Return the seed Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges
